from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from stgrape.pdf_crop import find_content_box, pixel_box_to_pdf_box


class PdfCropGeometryTests(unittest.TestCase):
    def test_find_content_box_uses_exclusive_right_and_bottom_edges(self) -> None:
        pixels = np.full((8, 10, 3), 255, dtype=np.uint8)
        pixels[2:7, 3:9] = 0
        image = Image.fromarray(pixels)

        self.assertEqual(find_content_box(image), (3, 2, 9, 7))

    def test_find_content_box_ignores_nearly_white_pixels(self) -> None:
        pixels = np.full((5, 6, 3), 255, dtype=np.uint8)
        pixels[0, 0] = 252
        pixels[3, 4] = 249
        image = Image.fromarray(pixels)

        self.assertEqual(
            find_content_box(image, white_threshold=250),
            (4, 3, 5, 4),
        )

    def test_find_content_box_returns_none_for_white_page(self) -> None:
        image = Image.new("RGB", (20, 30), "white")

        self.assertIsNone(find_content_box(image))

    def test_pixel_box_to_pdf_box_converts_origin_and_adds_padding(self) -> None:
        crop_box = pixel_box_to_pdf_box(
            (10, 20, 90, 70),
            image_size=(100, 80),
            page_size_points=(200.0, 160.0),
            padding_points=2.0,
        )

        expected = (18.0, 18.0, 182.0, 122.0)
        for actual, target in zip(crop_box, expected, strict=True):
            self.assertAlmostEqual(actual, target)

    def test_pixel_box_to_pdf_box_clamps_padding_to_page(self) -> None:
        crop_box = pixel_box_to_pdf_box(
            (0, 0, 100, 80),
            image_size=(100, 80),
            page_size_points=(200.0, 160.0),
            padding_points=5.0,
        )

        expected = (0.0, 0.0, 200.0, 160.0)
        for actual, target in zip(crop_box, expected, strict=True):
            self.assertAlmostEqual(actual, target)


if __name__ == "__main__":
    unittest.main()
