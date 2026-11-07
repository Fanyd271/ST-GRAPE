"""Crop white PDF margins using raster detection while preserving vector content."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PixelBox = tuple[int, int, int, int]
PdfBox = tuple[float, float, float, float]


class PdfCropError(RuntimeError):
    """Raised when a PDF cannot be analyzed or cropped safely."""


@dataclass(frozen=True)
class PageGeometry:
    """Physical size and rotation reported by pdfinfo."""

    width_points: float
    height_points: float
    rotation_degrees: int


@dataclass(frozen=True)
class PageCrop:
    """Detected crop information for one PDF page."""

    page: int
    geometry: PageGeometry
    crop_box: PdfBox

    @property
    def cropped_width_points(self) -> float:
        """Width of the cropped page in PDF points."""

        left, _, right, _ = self.crop_box
        return right - left

    @property
    def cropped_height_points(self) -> float:
        """Height of the cropped page in PDF points."""

        _, bottom, _, top = self.crop_box
        return top - bottom

    @property
    def removed_margins_points(self) -> tuple[float, float, float, float]:
        """Removed margins as (left, top, right, bottom) in PDF points."""

        left, bottom, right, top = self.crop_box
        return (
            left,
            self.geometry.height_points - top,
            self.geometry.width_points - right,
            bottom,
        )


_PAGE_COUNT_RE = re.compile(r"^Pages:\s+(\d+)\s*$", re.MULTILINE)
_PAGE_SIZE_RE = re.compile(
    r"^Page\s+\d+\s+size:\s+([0-9.eE+-]+)\s+x\s+([0-9.eE+-]+)\s+pts",
    re.MULTILINE,
)
_PAGE_ROTATION_RE = re.compile(r"^Page\s+\d+\s+rot:\s+(-?\d+)\s*$", re.MULTILINE)


def find_content_box(
    image: Image.Image,
    *,
    white_threshold: int = 250,
) -> PixelBox | None:
    """Return the smallest box containing pixels that are not nearly white.

    A pixel is treated as white when all three RGB channels are greater than or
    equal to white_threshold. The returned right and bottom coordinates are
    exclusive, following the Pillow crop convention.
    """

    if not 0 <= white_threshold <= 255:
        raise ValueError("white_threshold must lie between 0 and 255")

    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    content = np.any(pixels < white_threshold, axis=2)
    rows, columns = np.nonzero(content)
    if rows.size == 0:
        return None

    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def pixel_box_to_pdf_box(
    pixel_box: PixelBox,
    *,
    image_size: tuple[int, int],
    page_size_points: tuple[float, float],
    padding_points: float = 0.75,
) -> PdfBox:
    """Convert a top-left-origin pixel box to a PDF crop box."""

    if padding_points < 0:
        raise ValueError("padding_points must be nonnegative")

    image_width, image_height = image_size
    page_width, page_height = page_size_points
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if page_width <= 0 or page_height <= 0:
        raise ValueError("PDF page dimensions must be positive")

    left_px, top_px, right_px, bottom_px = pixel_box
    if not (0 <= left_px < right_px <= image_width and 0 <= top_px < bottom_px <= image_height):
        raise ValueError("pixel_box lies outside the rendered page")

    x_scale = page_width / image_width
    y_scale = page_height / image_height

    left = max(0.0, left_px * x_scale - padding_points)
    right = min(page_width, right_px * x_scale + padding_points)
    bottom = max(0.0, page_height - bottom_px * y_scale - padding_points)
    top = min(page_height, page_height - top_px * y_scale + padding_points)
    return left, bottom, right, top


def crop_pdf(
    input_path: Path | str,
    output_path: Path | str | None = None,
    *,
    dpi: int = 300,
    white_threshold: int = 250,
    padding_points: float = 0.75,
    in_place: bool = False,
    force: bool = False,
) -> tuple[Path, list[PageCrop]]:
    """Crop white margins and return the output path and per-page reports.

    Pages are rasterized only for boundary detection. The output is produced
    from the original pages with pdfcrop, preserving vector graphics and
    searchable text.
    """

    input_pdf = Path(input_path).expanduser().resolve()
    if not input_pdf.is_file():
        raise PdfCropError(f"Input PDF does not exist: {input_pdf}")
    if input_pdf.suffix.lower() != ".pdf":
        raise PdfCropError(f"Input must be a PDF file: {input_pdf}")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if not 0 <= white_threshold <= 255:
        raise ValueError("white_threshold must lie between 0 and 255")
    if padding_points < 0:
        raise ValueError("padding_points must be nonnegative")
    if in_place and output_path is not None:
        raise PdfCropError("Do not provide an output path together with --in-place")

    if in_place:
        output_pdf = input_pdf
    elif output_path is None:
        output_pdf = input_pdf.with_name(f"{input_pdf.stem}_cropped.pdf")
    else:
        output_pdf = Path(output_path).expanduser().resolve()

    if output_pdf.suffix.lower() != ".pdf":
        raise PdfCropError(f"Output must use the .pdf extension: {output_pdf}")
    if output_pdf == input_pdf and not in_place:
        raise PdfCropError("Use --in-place to overwrite the input PDF")
    if output_pdf.exists() and output_pdf != input_pdf and not force:
        raise PdfCropError(f"Output already exists; pass --force to replace it: {output_pdf}")

    for executable in ("pdfinfo", "pdftoppm", "pdfcrop"):
        _require_executable(executable)

    page_count = _read_page_count(input_pdf)
    if page_count > 1:
        for executable in ("pdfseparate", "pdfunite"):
            _require_executable(executable)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    reports: list[PageCrop] = []

    with tempfile.TemporaryDirectory(prefix="stgrape-pdf-crop-") as temporary_directory:
        temporary = Path(temporary_directory)

        for page in range(1, page_count + 1):
            geometry = _read_page_geometry(input_pdf, page)
            if geometry.rotation_degrees % 360 != 0:
                raise PdfCropError(
                    f"Page {page} is rotated by {geometry.rotation_degrees} degrees; "
                    "rotated pages are not currently supported"
                )

            rendered_page = _render_page(input_pdf, page, dpi=dpi, directory=temporary)
            with Image.open(rendered_page) as image:
                content_box = find_content_box(
                    image,
                    white_threshold=white_threshold,
                )
                if content_box is None:
                    crop_box: PdfBox = (
                        0.0,
                        0.0,
                        geometry.width_points,
                        geometry.height_points,
                    )
                else:
                    crop_box = pixel_box_to_pdf_box(
                        content_box,
                        image_size=image.size,
                        page_size_points=(
                            geometry.width_points,
                            geometry.height_points,
                        ),
                        padding_points=padding_points,
                    )
            reports.append(PageCrop(page=page, geometry=geometry, crop_box=crop_box))

        if page_count == 1:
            assembled_pdf = temporary / "cropped.pdf"
            _crop_vector_page(input_pdf, assembled_pdf, reports[0].crop_box)
        else:
            split_pattern = temporary / "page-%d.pdf"
            _run(["pdfseparate", str(input_pdf), str(split_pattern)])
            cropped_pages: list[Path] = []
            for report in reports:
                source_page = temporary / f"page-{report.page}.pdf"
                cropped_page = temporary / f"cropped-{report.page}.pdf"
                _crop_vector_page(source_page, cropped_page, report.crop_box)
                cropped_pages.append(cropped_page)

            assembled_pdf = temporary / "cropped.pdf"
            _run(["pdfunite", *(str(path) for path in cropped_pages), str(assembled_pdf)])

        _replace_output(assembled_pdf, output_pdf, mode_source=input_pdf if in_place else None)

    return output_pdf, reports


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise PdfCropError(
            f"Required executable '{name}' was not found. "
            "Install Poppler and a TeX distribution providing pdfcrop."
        )


def _run(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise PdfCropError(f"Required executable was not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip() or "no diagnostic output"
        raise PdfCropError(f"Command failed: {' '.join(command)}\n{details}") from error
    return completed.stdout


def _read_page_count(input_pdf: Path) -> int:
    information = _run(["pdfinfo", str(input_pdf)])
    match = _PAGE_COUNT_RE.search(information)
    if match is None:
        raise PdfCropError(f"Could not determine the page count of {input_pdf}")
    return int(match.group(1))


def _read_page_geometry(input_pdf: Path, page: int) -> PageGeometry:
    information = _run(["pdfinfo", "-f", str(page), "-l", str(page), str(input_pdf)])
    size_match = _PAGE_SIZE_RE.search(information)
    rotation_match = _PAGE_ROTATION_RE.search(information)
    if size_match is None or rotation_match is None:
        raise PdfCropError(f"Could not determine the geometry of page {page}")

    return PageGeometry(
        width_points=float(size_match.group(1)),
        height_points=float(size_match.group(2)),
        rotation_degrees=int(rotation_match.group(1)),
    )


def _render_page(input_pdf: Path, page: int, *, dpi: int, directory: Path) -> Path:
    prefix = directory / f"rendered-{page}"
    _run(
        [
            "pdftoppm",
            "-f",
            str(page),
            "-l",
            str(page),
            "-singlefile",
            "-r",
            str(dpi),
            "-png",
            str(input_pdf),
            str(prefix),
        ]
    )
    rendered_page = prefix.with_suffix(".png")
    if not rendered_page.is_file():
        raise PdfCropError(f"Renderer did not produce the expected file: {rendered_page}")
    return rendered_page


def _crop_vector_page(input_pdf: Path, output_pdf: Path, crop_box: PdfBox) -> None:
    left, bottom, right, top = crop_box
    box_argument = f"{left:.6f} {bottom:.6f} {right:.6f} {top:.6f}"
    _run(
        [
            "pdfcrop",
            "--quiet",
            "--clip",
            "--bbox",
            box_argument,
            str(input_pdf),
            str(output_pdf),
        ]
    )


def _replace_output(source: Path, destination: Path, *, mode_source: Path | None) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=".pdf",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_output = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary_output)
        if mode_source is not None:
            os.chmod(temporary_output, mode_source.stat().st_mode)
        os.replace(temporary_output, destination)
    finally:
        temporary_output.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stgrape-crop-pdf",
        description=(
            "Detect nearly white PDF margins from rendered pixels and crop the "
            "original vector pages to the detected content."
        ),
    )
    parser.add_argument("input", type=Path, help="input PDF")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="output PDF; defaults to <input>_cropped.pdf",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="rendering resolution used for edge detection (default: 300)",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=250,
        metavar="0..255",
        help=("a pixel is white when every RGB channel is at least this value (default: 250)"),
    )
    parser.add_argument(
        "--padding-pt",
        type=float,
        default=0.75,
        help="safe padding retained around detected content, in points (default: 0.75)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="atomically replace the input PDF",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the crop report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the PDF whitespace cropper."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        output, reports = crop_pdf(
            arguments.input,
            arguments.output,
            dpi=arguments.dpi,
            white_threshold=arguments.white_threshold,
            padding_points=arguments.padding_pt,
            in_place=arguments.in_place,
            force=arguments.force,
        )
    except (PdfCropError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")

    if arguments.quiet:
        return

    for report in reports:
        left, top, right, bottom = report.removed_margins_points
        print(
            f"Page {report.page}: "
            f"{report.geometry.width_points:.2f} x "
            f"{report.geometry.height_points:.2f} pt -> "
            f"{report.cropped_width_points:.2f} x "
            f"{report.cropped_height_points:.2f} pt; "
            f"removed L/T/R/B = "
            f"{left:.2f}/{top:.2f}/{right:.2f}/{bottom:.2f} pt"
        )
    print(f"Saved cropped PDF to {output}")


if __name__ == "__main__":
    main()
