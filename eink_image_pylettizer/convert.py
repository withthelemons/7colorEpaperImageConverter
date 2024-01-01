import functools
import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from statistics import mean
from time import perf_counter
from io import BytesIO
from pathlib import Path
import argparse
from typing import Collection, Literal

from PIL import Image, ImageOps, ImageStat
from wand.image import Image as WandImage

vipsbin = "C:/libvips/vips-dev-8.15/bin"
os.environ["PATH"] = vipsbin + ";" + os.environ["PATH"]

use_libvips = False
try:
    import pyvips
    from pyvips.enums import Interesting
except OSError as e:
    print("Not using libvips")
else:
    major, minor, patch = pyvips.base.version(0), pyvips.base.version(1), pyvips.base.version(2)
    print(f"Using libvips {major}.{minor}.{patch}")
    use_libvips = True
    del major, minor, patch

logger = logging.getLogger(__name__)
IMAGE_PROCESSOR = None

# available palettes
OG_BW_PALETTE = (0, 0, 0, 255, 255, 255)
LIMITED_BW_PALETTE = (16, 16, 16, 240, 240, 240)
OLD_SATURATED_PALETTE = (57, 48, 57, 255, 255, 255, 58, 91, 70, 61, 59, 94, 156, 72, 75, 208, 190, 71, 177, 106, 73)
SAT_PAL_TUNED_FOR_25 = (57, 48, 57, 255, 255, 255, 58, 91, 70, 25, 70, 100, 156, 72, 75, 208, 190, 71, 177, 106, 73)
SAT_PAL_TUNED_FOR_50 = (57, 48, 57, 255, 255, 255, 58, 91, 70, 39, 66, 98, 156, 72, 75, 208, 190, 71, 177, 106, 73)
FROM_PHOTO_PALETTE = (42, 45, 63, 227, 227, 227, 77, 111, 86, 57, 69, 107, 168, 85, 81, 222, 206, 95, 195, 104, 86)
DATASHEET_PALETTE = (50, 39, 56, 173, 173, 173, 45, 101, 67, 63, 62, 105, 144, 61, 63, 167, 161, 72, 157, 83, 65)
ALMOST_DATASHEET_PALETTE = (0, 0, 0, 235, 235, 235, 45, 101, 67, 63, 62, 105, 144, 61, 63, 167, 161, 72, 157, 83, 65)
OG_PALETTE = (0, 0, 0, 255, 255, 255, 0, 255, 0, 0, 0, 255, 255, 0, 0, 255, 255, 0, 255, 128, 0)

# selected palettes
BW_PALETTE = LIMITED_BW_PALETTE
SATURATED_PALETTE = FROM_PHOTO_PALETTE
PALETTE_BLEND_RATIO = 1 / 3


def split_palette(palette: Collection[int]) -> list[Collection[int]]:
    return [palette[x : x + 3] for x in range(0, len(palette), 3)]


@functools.cache
def blend_palette(saturation: float) -> list[int]:
    palette = []
    for i in range(7):
        rs, gs, bs = [c * saturation for c in split_palette(SATURATED_PALETTE)[i]]
        rd, gd, bd = [c * (1 - saturation) for c in split_palette(OG_PALETTE)[i]]
        palette.extend((round(rs + rd), round(gs + gd), round(bs + bd)))
    return palette


def get_target_size(display_direction: Literal["landscape", "portrait"]) -> tuple[int, int]:
    if display_direction == "landscape":
        target_size = 800, 480
    else:
        target_size = 480, 800
    return target_size


def pillow_resize(
    input_filename: Path,
    display_direction: Literal["landscape", "portrait"],
    display_mode: Literal["fit", "pad"],
) -> Image:
    input_image = Image.open(input_filename)
    target_size = get_target_size(display_direction)
    if target_size == input_image.size:
        resized_image = input_image
    elif display_mode == "fit":
        resized_image = ImageOps.fit(input_image, target_size, method=Image.Resampling.LANCZOS)
    else:
        stats = ImageStat.Stat(input_image)
        mean_brightness = mean(stats.mean)
        image_is_dark = mean_brightness < 100
        logger.debug(f"{mean_brightness=} {image_is_dark}")
        pad_color = (0, 0, 0) if image_is_dark else (255, 255, 255)
        resized_image = ImageOps.pad(input_image, target_size, method=Image.Resampling.LANCZOS, color=pad_color)
    return resized_image


def libvips_resize(
    input_filename: Path,
    display_direction: Literal["landscape", "portrait"],
    display_mode: Literal["fit", "pad"],
) -> Image:
    filename = str(input_filename)
    width, height = get_target_size(display_direction)
    if display_mode == "fit":
        image = pyvips.Image.thumbnail(filename, width, height=height, crop=Interesting.CENTRE, no_rotate=True)
    else:
        image = pyvips.Image.new_from_file(filename, access="sequential")
        mean_brightness = image.avg()
        image_is_dark = mean_brightness < 100
        logger.debug(f"{mean_brightness=} {image_is_dark}")
        pad_color = (0, 0, 0) if image_is_dark else (255, 255, 255)

        image = pyvips.Image.thumbnail(filename, width, height=height, no_rotate=True)
        image = image.gravity("centre", width, height, background=pad_color)

    image.flatten()
    data = image.write_to_memory()
    return Image.frombytes(mode="RGB", size=(image.width, image.height), data=data)


def convert(
    input_filename: Path,
    display_direction: Literal["landscape", "portrait"],
    display_mode: Literal["fit", "pad"],
    bw: bool,
    output_dir: Path,
    use_km: bool,
    skip_existing: bool,
) -> None:
    start = perf_counter()
    output_path = make_output_path(bw, display_mode, input_filename, output_dir, use_km)
    if skip_existing and output_path.exists():
        print(f"Skipping {output_path}")
        return
    if use_libvips:
        resized_image = libvips_resize(input_filename, display_direction, display_mode)
        print(f"libvips resize took {(perf_counter()-start)*1000:.3f} ms")
    else:
        resized_image = pillow_resize(input_filename, display_direction, display_mode)
        print(f"pillow_resize resize took {(perf_counter()-start)*1000:.3f} ms")

    # Create a palette object
    use_km = use_km if not bw else False
    if use_km:
        global IMAGE_PROCESSOR
        if IMAGE_PROCESSOR is None:
            from image_processor import ImageProcessor

            IMAGE_PROCESSOR = ImageProcessor()
        IMAGE_PROCESSOR.diffuse_image(resized_image)
        quantized_image = resized_image
    else:
        pal_image = Image.new("P", (1, 1))
        if bw:
            palette = BW_PALETTE
        else:
            palette = blend_palette(PALETTE_BLEND_RATIO)
        pal_image.putpalette(palette)
        quantized_image = resized_image.quantize(palette=pal_image)

    # save to buffer
    buffer = BytesIO()
    quantized_image.save(buffer, format="bmp")
    buffer.seek(0)

    # Save output image
    wand_image = WandImage(file=buffer, format="bmp")
    wand_image.flop()
    wand_image.type = "palette"
    wand_image.save(filename=output_path)
    print(f"Successfully converted {input_filename} to {output_path}")


def make_output_path(bw: bool, display_mode: str, input_filename: Path, output_dir: Path, use_km: bool) -> Path:
    if use_km:
        extra_string = "_km"
    elif bw:
        extra_string = "_bw"
    else:
        extra_string = f"_{int(PALETTE_BLEND_RATIO * 100)}"
    image_name = f"{input_filename.stem}_{display_mode}{extra_string}_converted.bmp"
    return output_dir / image_name


def main() -> None:
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        prog="python -m eink_image_pylettizer", description="Converts images to bmp format for 7 color eink displays"
    )

    # Add orientation parameter
    parser.add_argument("image_path", help="Path to image file or folder")
    parser.add_argument(
        "--orientation",
        choices=["landscape", "portrait"],
        default="landscape",
        help="Image orientation (landscape or portrait)",
    )
    parser.add_argument("--mode", choices=["fit", "pad"], default="pad")
    parser.add_argument("--bw", action=argparse.BooleanOptionalAction, help="Black and white mode")
    parser.add_argument(
        "--use-km", action=argparse.BooleanOptionalAction, help="Use Kode Munkie's algorithm for dithering"
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        help="Skip images that already exist in the target folder",
    )

    args = parser.parse_args()
    input_filename = Path(args.image_path)
    # Check whether the input file exists
    if not input_filename.exists():
        print(f"Error: file {input_filename} does not exist")
        sys.exit(1)

    bw_string = "_BW" if args.bw else ""
    output_dir = Path(f"converted{bw_string}/")
    output_dir.mkdir(exist_ok=True)

    if input_filename.is_dir():
        print("Input is a directory")
        filelist_png = input_filename.glob("**/*.png")
        filelist_jpg = input_filename.glob("**/*.jpg")
        start = perf_counter()
        with ThreadPoolExecutor() as executor:
            for input_file in chain(filelist_png, filelist_jpg):
                arguments = (
                    input_file,
                    args.orientation,
                    args.mode,
                    args.bw,
                    output_dir,
                    args.use_km,
                    args.skip_existing,
                )
                executor.submit(convert, *arguments)
        print(f"Processing all images took {perf_counter()-start:.3f} seconds")
    else:
        print("Input is a single file")
        convert(input_filename, args.orientation, args.mode, args.bw, output_dir, args.use_km, args.skip_existing)


if __name__ == "__main__":
    main()
