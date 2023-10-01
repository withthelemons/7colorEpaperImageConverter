# SPDX-FileCopyrightText: 2022 Kode Munkie <https://github.com/KodeMunkie>
# PDX-FileCopyrightText: 2023 Eliška Szopová / withthelemons <https://github.com/withthelemons>
# SPDX-License-Identifier: MIT
# https://github.com/KodeMunkie/inky-impression-slideshow/blob/main/image_processor.py


import sys
from typing import TypeAlias

from PIL.Image import Image

from mypy_extensions import u8, i64
pixel_type: TypeAlias = tuple[u8, u8, u8]


class ImageProcessor:
    TARGET_PALETTE: tuple[pixel_type, pixel_type, pixel_type, pixel_type, pixel_type, pixel_type, pixel_type] = (
        (u8(0x0C), u8(0x0C), u8(0x0E)),  # Black
        (u8(0xD2), u8(0xD2), u8(0xD0)),  # White
        (u8(0x1E), u8(0x60), u8(0x1F)),  # Green
        (u8(0x1D), u8(0x1E), u8(0xAA)),  # Blue
        (u8(0x8C), u8(0x1B), u8(0x1D)),  # Red
        (u8(0xD3), u8(0xC9), u8(0x3D)),  # Yellow
        (u8(0xC1), u8(0x71), u8(0x2A)),  # Orange
    )

    @staticmethod
    def euclidean_distance(source_colour: pixel_type, target_colour: pixel_type) -> i64:
        red_diff: i64 = i64(source_colour[0]) - i64(target_colour[0])
        green_diff: i64 = i64(source_colour[1]) - i64(target_colour[1])
        blue_diff: i64 = i64(source_colour[2]) - i64(target_colour[2])
        return (red_diff * red_diff) + (blue_diff * blue_diff) + (green_diff * green_diff)

    def get_closest_colour(self, old_pixel: pixel_type) -> pixel_type:
        best_candidate = self.TARGET_PALETTE[0]
        best_distance: i64 = sys.maxsize
        for candidate in self.TARGET_PALETTE:
            candidate_distance: i64 = self.euclidean_distance(old_pixel, candidate)
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_candidate = candidate
        return best_candidate

    @staticmethod
    def is_in_bounds(size: tuple[i64, i64], x: i64, y: i64) -> bool:
        return 0 <= x < size[0] and 0 <= y < size[1]

    @staticmethod
    def clamp(value: float) -> u8:
        if value > 255:
            return 255
        if value < 0:
            return 0
        return u8(value)

    def calculate_adjusted_rgb(self, old_rgb: pixel_type, new_rgb: pixel_type, diffused_rgb: pixel_type) -> pixel_type:
        red = self.clamp(float(diffused_rgb[0]) + ((float(old_rgb[0]) - float(new_rgb[0])) / 8))
        green = self.clamp(float(diffused_rgb[1]) + ((float(old_rgb[1]) - float(new_rgb[1])) / 8))
        blue = self.clamp(float(diffused_rgb[2]) + ((float(old_rgb[2]) - float(new_rgb[2])) / 8))
        return red, green, blue

    def distribute_error(self, output: Image, old_pixel: pixel_type, new_pixel: pixel_type, x: i64, y: i64) -> None:
        x_plus_1 = self.is_in_bounds(output.size, x + 1, y)
        y_plus_1 = self.is_in_bounds(output.size, x, y + 1)
        if x_plus_1:
            adj_pixel = self.calculate_adjusted_rgb(old_pixel, new_pixel, output.getpixel((x + 1, y)))
            output.putpixel((x + 1, y), adj_pixel)
        if self.is_in_bounds(output.size, x + 2, y):
            adj_pixel = self.calculate_adjusted_rgb(old_pixel, new_pixel, output.getpixel((x + 2, y)))
            output.putpixel((x + 2, y), adj_pixel)
        if self.is_in_bounds(output.size, x - 1, y + 1):
            adj_pixel = self.calculate_adjusted_rgb(old_pixel, new_pixel, output.getpixel((x - 1, y + 1)))
            output.putpixel((x - 1, y + 1), adj_pixel)

        if y_plus_1:
            adj_pixel = self.calculate_adjusted_rgb(old_pixel, new_pixel, output.getpixel((x, y + 1)))
            output.putpixel((x, y + 1), adj_pixel)
        if self.is_in_bounds(output.size, x, y + 2):
            adj_pixel = self.calculate_adjusted_rgb(old_pixel, new_pixel, output.getpixel((x, y + 2)))
            output.putpixel((x, y + 2), adj_pixel)
        if x_plus_1 and y_plus_1:
            adj_pixel = self.calculate_adjusted_rgb(old_pixel, new_pixel, output.getpixel((x + 1, y + 1)))
            output.putpixel((x + 1, y + 1), adj_pixel)

    def diffuse_pixel(self, output: Image, x: i64, y: i64) -> None:
        old_pixel: pixel_type = output.getpixel((x, y))
        new_pixel = self.get_closest_colour(old_pixel)
        output.putpixel((x, y), new_pixel)
        self.distribute_error(output, old_pixel, new_pixel, x, y)

    def diffuse_image(self, source_image: Image) -> None:
        height_range = range(0, source_image.height)
        width_range = range(0, source_image.width)
        for y in height_range:
            for x in width_range:
                self.diffuse_pixel(source_image, x, y)
