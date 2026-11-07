"""Plotting implementation transcribed from the final paper notebook.

Run through stgrape.analysis; all outputs are directed outside the data archive.
"""

import csv
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCALING_TASK = globals().get("TASK", "ghz")
DATA_ROOT = Path(globals()["DATA_ROOT"])
BLOCK_RESULT_ROOT = DATA_ROOT / "paper/scaling" / SCALING_TASK / "blocks"
DIMENSION_RESULT_ROOT = DATA_ROOT / "paper/scaling" / SCALING_TASK / "dimension"
INCLUDE_EXPM_MULTIPLY = bool(globals().get("SUPPLEMENTARY", False))
EXPM_BLOCK_RESULT_ROOT = DATA_ROOT / "supplementary/exponential_action" / SCALING_TASK / "blocks"
EXPM_DIMENSION_RESULT_ROOT = (
    DATA_ROOT / "supplementary/exponential_action" / SCALING_TASK / "dimension"
)
SAVE_FIGURES = True
FIGURE_DIRECTORY = Path(globals()["OUTPUT_DIR"])

FIT_MIN_BLOCKS = 66
FIT_MIN_DIMENSION = 64
FIT_MAX_BLOCKS = None
FIT_MAX_DIMENSION = None
FIT_COMMON_RANGE = False
MIN_FIT_POINTS = 5
FIT_LINE_BACKWARD_EXTENSION = 3.0
FIT_LINE_FORWARD_EXTENSION = 2.0

# Publication typography. X_LABEL_FONT_SIZE controls xlabel text;
# TICK_LABEL_FONT_SIZE controls the numerical labels along both axes.
X_LABEL_FONT_SIZE = 7.0
Y_LABEL_FONT_SIZE = 7.0
TICK_LABEL_FONT_SIZE = 7.0
LEGEND_FONT_SIZE = 5.2
PANEL_LABEL_FONT_SIZE = 7.0
EXPORT_DPI = 360

COLORS = {
    "ST": "#0072B2",
    "dense": "#D55E00",
    "block-sparse": "#009E73",
    "expm_multiply": "#7A3E9D",
}
MARKERS = {"ST": "o", "dense": "o", "block-sparse": "o", "expm_multiply": "D"}
METHOD_ORDER = ("ST", "dense", "block-sparse", "expm_multiply")

if SAVE_FIGURES:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)


def load_timing_records(result_root, *, coordinate_key, context_key, extra_roots=()):
    raw_paths = sorted(result_root.glob("*/raw_data.npz"))
    for extra_root in extra_roots:
        extra_paths = sorted(extra_root.glob("*/raw_data.npz"))
        if not extra_paths:
            warnings.warn(f"No optional expm_multiply data below {extra_root}.", stacklevel=2)
        raw_paths.extend(extra_paths)
    if not raw_paths:
        raise FileNotFoundError(f"No method raw_data.npz files below {result_root}")

    bundles = {}
    for path in raw_paths:
        with np.load(path) as saved:
            bundle = {name: saved[name] for name in saved.files}
        method = str(np.asarray(bundle["method"]).item())
        if method in bundles:
            raise ValueError(f"Duplicate raw result for {method}: {path}")
        bundles[method] = bundle

    methods = [method for method in METHOD_ORDER if method in bundles]
    methods.extend(sorted(set(bundles) - set(methods)))
    records = {}
    for method in methods:
        raw = bundles[method]
        coordinate = np.asarray(raw[coordinate_key], dtype=int)
        order = np.argsort(coordinate)
        samples = np.asarray(raw["objective_gradient_times_s"], dtype=float)[:, 0, :][order]
        valid = np.all(np.isfinite(samples), axis=1)
        if "status" in raw:
            valid &= np.asarray(raw["status"]).astype(str)[order] == "ok"
        coordinate = coordinate[order][valid]
        samples = samples[valid]
        if coordinate.size == 0:
            warnings.warn(f"No successful timing points for {method}; skipping it.", stacklevel=2)
            continue
        records[method] = {
            "coordinate": coordinate,
            "context": np.asarray(raw[context_key], dtype=int)[order][valid],
            "median": np.median(samples, axis=1),
            "lower": np.quantile(samples, 0.25, axis=1),
            "upper": np.quantile(samples, 0.75, axis=1),
        }
    return [method for method in methods if method in records], records


def fit_timing_records(
    methods, records, *, minimum, maximum=None, common_range=False, minimum_points=5
):
    common_coordinates = set(records[methods[0]]["coordinate"].tolist())
    for method in methods[1:]:
        common_coordinates.intersection_update(records[method]["coordinate"].tolist())

    fits = {}
    details = {}
    for method in methods:
        record = records[method]
        coordinate = record["coordinate"]
        median = record["median"]
        mask = (coordinate >= minimum) & (median > 0) & np.isfinite(median)
        if maximum is not None:
            mask &= coordinate <= maximum
        if common_range:
            mask &= np.isin(coordinate, sorted(common_coordinates))
        if mask.sum() < minimum_points:
            warnings.warn(
                f"{method} has only {mask.sum()} fit points; need at least "
                f"{minimum_points}. Its measured curve is retained without a fit.",
                stacklevel=2,
            )
            details[method] = {
                "status": "insufficient-data",
                "n_points": int(mask.sum()),
                "minimum_required": int(minimum_points),
            }
            continue
        exponent, intercept = np.polyfit(np.log(coordinate[mask]), np.log(median[mask]), 1)
        fits[method] = float(exponent)
        record["fit_mask"] = mask
        record["fit_intercept"] = float(intercept)
        details[method] = {
            "exponent": float(exponent),
            "intercept": float(intercept),
            "minimum_coordinate": int(coordinate[mask][0]),
            "maximum_coordinate": int(coordinate[mask][-1]),
            "n_points": int(mask.sum()),
        }
    return fits, details


def draw_scaling_row(
    time_axis,
    speedup_axis,
    *,
    methods,
    records,
    fits,
    coordinate_label,
    exponent_symbol,
    panel_labels,
):
    fit_handles = []
    fit_labels = []
    for method in methods:
        record = records[method]
        coordinate = record["coordinate"]
        measured_handle = time_axis.plot(
            coordinate,
            record["median"],
            marker=MARKERS.get(method, "o"),
            linestyle="-",
            color=COLORS.get(method),
            linewidth=1.0,
            markersize=3.1,
            alpha=0.6,
            zorder=3,
        )[0]
        time_axis.fill_between(
            coordinate,
            record["lower"],
            record["upper"],
            color=COLORS.get(method),
            alpha=0.06,
            linewidth=0,
            zorder=1,
        )
        if method not in fits:
            fit_handles.append(measured_handle)
            fit_labels.append(f"{method}: fit unavailable")
            continue
        selected = coordinate[record["fit_mask"]]
        fit_coordinate = np.geomspace(
            selected[0] / FIT_LINE_BACKWARD_EXTENSION,
            selected[-1] * FIT_LINE_FORWARD_EXTENSION,
            200,
        )
        fitted_time = np.exp(record["fit_intercept"]) * fit_coordinate ** fits[method]
        handle = time_axis.plot(
            fit_coordinate,
            fitted_time,
            color=COLORS.get(method),
            linewidth=1.25,
            linestyle=(0, (5, 2.5)),
            alpha=0.98,
            zorder=5,
        )[0]
        fit_handles.append(handle)
        fit_labels.append(rf"{method}: ${exponent_symbol}={fits[method]:.2f}$")

    if "ST" in records:
        st_lookup = dict(zip(records["ST"]["coordinate"], records["ST"]["median"], strict=True))
        for method in methods:
            if method == "ST":
                continue
            record = records[method]
            overlap = np.asarray(
                [value for value in record["coordinate"] if value in st_lookup], dtype=int
            )
            if overlap.size:
                method_lookup = dict(zip(record["coordinate"], record["median"], strict=True))
                ratio = np.asarray([method_lookup[value] / st_lookup[value] for value in overlap])
                speedup_axis.plot(
                    overlap,
                    ratio,
                    marker=MARKERS.get(method, "o"),
                    linestyle="-",
                    color=COLORS.get(method),
                    linewidth=1.2,
                    markersize=3.3,
                    alpha=0.9,
                )

    time_axis.set_xscale("log")
    time_axis.set_yscale("log")
    speedup_axis.set_xscale("log")
    speedup_axis.set_yscale("log")
    speedup_axis.axhline(1.0, color="0.45", linewidth=0.8, linestyle=":")
    time_axis.set_xlabel(coordinate_label, fontsize=X_LABEL_FONT_SIZE)
    time_axis.set_ylabel("Time (s)", fontsize=Y_LABEL_FONT_SIZE)
    speedup_axis.set_xlabel(coordinate_label, fontsize=X_LABEL_FONT_SIZE)
    speedup_axis.set_ylabel("Speedup", fontsize=Y_LABEL_FONT_SIZE)

    for label, axis in zip(panel_labels, (time_axis, speedup_axis), strict=True):
        axis.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONT_SIZE)
        axis.grid(True, which="major", alpha=0.24, linewidth=0.6)
        axis.grid(True, which="minor", alpha=0.10, linewidth=0.4)
        axis.set_title(
            label,
            loc="left",
            pad=2.0,
            fontsize=PANEL_LABEL_FONT_SIZE,
            fontweight="bold",
        )

    legend = time_axis.legend(
        fit_handles,
        fit_labels,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        facecolor="white",
        edgecolor="0.82",
        framealpha=1.0,
        borderpad=0.30,
        handlelength=2.3,
        labelspacing=0.25,
    )
    legend.set_zorder(20)


block_methods, block_records = load_timing_records(
    BLOCK_RESULT_ROOT,
    coordinate_key="block_counts",
    context_key="source_counts",
    extra_roots=(EXPM_BLOCK_RESULT_ROOT,) if INCLUDE_EXPM_MULTIPLY else (),
)
block_fits, block_fit_details = fit_timing_records(
    block_methods,
    block_records,
    minimum=FIT_MIN_BLOCKS,
    maximum=FIT_MAX_BLOCKS,
    common_range=FIT_COMMON_RANGE,
    minimum_points=MIN_FIT_POINTS,
)

dimension_methods, dimension_records = load_timing_records(
    DIMENSION_RESULT_ROOT,
    coordinate_key="hilbert_dimensions",
    context_key="n_qubits",
    extra_roots=(EXPM_DIMENSION_RESULT_ROOT,) if INCLUDE_EXPM_MULTIPLY else (),
)
dimension_fits, dimension_fit_details = fit_timing_records(
    dimension_methods,
    dimension_records,
    minimum=FIT_MIN_DIMENSION,
    maximum=FIT_MAX_DIMENSION,
    common_range=FIT_COMMON_RANGE,
    minimum_points=MIN_FIT_POINTS,
)

fit_summary = {
    "block_scaling": block_fit_details,
    "dimension_scaling": dimension_fit_details,
}
if SAVE_FIGURES:
    summary_name = (
        f"{SCALING_TASK}_fit_summary_with_expm_multiply.json"
        if INCLUDE_EXPM_MULTIPLY
        else "fit_summary.json"
    )
    csv_name = (
        f"{SCALING_TASK}_fit_summary_with_expm_multiply.csv"
        if INCLUDE_EXPM_MULTIPLY
        else "fit_summary.csv"
    )
    (FIGURE_DIRECTORY / summary_name).write_text(
        json.dumps(fit_summary, indent=2), encoding="utf-8"
    )
    with (FIGURE_DIRECTORY / csv_name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "analysis",
                "method",
                "exponent",
                "intercept",
                "minimum_coordinate",
                "maximum_coordinate",
                "n_points",
            ),
        )
        writer.writeheader()
        for analysis, method_details in fit_summary.items():
            for method, details in method_details.items():
                if details.get("status") != "insufficient-data":
                    writer.writerow({"analysis": analysis, "method": method, **details})

fig, axes = plt.subplots(4, 1, figsize=(3.4, 5.7))

draw_scaling_row(
    axes[0],
    axes[1],
    methods=block_methods,
    records=block_records,
    fits=block_fits,
    coordinate_label="Blocks $N$",
    exponent_symbol=r"\alpha",
    panel_labels=("(a)", "(b)"),
)
draw_scaling_row(
    axes[2],
    axes[3],
    methods=dimension_methods,
    records=dimension_records,
    fits=dimension_fits,
    coordinate_label="Dimension $d$",
    exponent_symbol=r"\gamma",
    panel_labels=("(c)", "(d)"),
)

fig.subplots_adjust(left=0.17, right=0.995, bottom=0.07, top=0.995, hspace=0.48)
if SAVE_FIGURES:
    figure_stem = (
        f"{SCALING_TASK}_scaling_with_expm_multiply"
        if INCLUDE_EXPM_MULTIPLY
        else "computational_scaling"
    )
    pdf_path = FIGURE_DIRECTORY / f"{figure_stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    if globals().get("CROP_PDF", False):
        from stgrape.pdf_crop import crop_pdf

        crop_pdf(pdf_path, in_place=True)
    fig.savefig(
        FIGURE_DIRECTORY / f"{figure_stem}.png",
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        pad_inches=0.01,
    )
plt.close("all")
