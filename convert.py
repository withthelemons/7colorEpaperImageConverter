import functools
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from time import perf_counter
from io import BytesIO
from pathlib import Path
import argparse
from typing import Collection, Literal

from PIL import Image, ImageOps, ImageStat
from wand.image import Image as WandImage

from image_processor import ImageProcessor

logger = logging.getLogger(__name__)
IMAGE_PROCESSOR = ImageProcessor()

# available palettes
OG_BW_PALETTE = (0, 0, 0, 255, 255, 255)
LIMITED_BW_PALETTE = (16, 16, 16, 235, 235, 235)
OLD_SATURATED_PALETTE = (57, 48, 57, 255, 255, 255, 58, 91, 70, 61, 59, 94, 156, 72, 75, 208, 190, 71, 177, 106, 73)
SATURATED_PALETTE_TUNED_FOR_25 = (57, 48, 57, 255, 255, 255, 58, 91, 70, 25, 70, 100, 156, 72, 75, 208, 190, 71, 177, 106, 73)
SATURATED_PALETTE_TUNED_FOR_50 = (57, 48, 57, 255, 255, 255, 58, 91, 70, 39, 66, 98, 156, 72, 75, 208, 190, 71, 177, 106, 73)
FROM_PHOTO_PALETTE = (42, 45, 63, 227, 227, 227, 77, 111, 86, 57, 69, 107, 168, 85, 81, 222, 206, 95, 195, 104, 86)
DATASHEET_PALETTE = (50, 39, 56, 173, 173, 173, 45, 101, 67, 63, 62, 105, 144, 61, 63, 167, 161, 72, 157, 83, 65)
ALMOST_DATASHEET_PALETTE = (0, 0, 0, 255, 255, 255, 45, 101, 67, 63, 62, 105, 144, 61, 63, 167, 161, 72, 157, 83, 65)
OG_PALETTE = (0, 0, 0, 255, 255, 255, 0, 255, 0, 0, 0, 255, 255, 0, 0, 255, 255, 0, 255, 128, 0)

# selected palettes
BW_PALETTE = OG_BW_PALETTE
SATURATED_PALETTE = ALMOST_DATASHEET_PALETTE


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


def get_target_size(
    display_direction: Literal["landscape", "portrait"] | None, input_image_size: tuple[int, int]
) -> tuple[int, int]:
    if display_direction:
        if display_direction == "landscape":
            target_size = 800, 480
        else:
            target_size = 480, 800
    else:
        if input_image_size[0] < input_image_size[1]:
            target_size = 480, 800
        else:
            target_size = 800, 480
    return target_size


def convert(
    input_filename: Path,
    display_direction: Literal["landscape", "portrait"] | None,
    display_mode: Literal["fit", "pad"],
    bw: bool,
    output_dir: Path,
    palette_blend_ratio: float,
    use_km: bool
) -> None:
    use_km = use_km if not bw else False
    input_image = Image.open(input_filename)
    stats = ImageStat.Stat(input_image)
    mean_brightness = mean(stats.mean)
    image_is_dark = mean_brightness < 100
    logger.debug(f"{mean_brightness=} {image_is_dark}")
    target_size = get_target_size(display_direction, input_image.size)
    pad_color = (0, 0, 0) if image_is_dark else (255, 255, 255)

    if target_size == input_image.size:
        resized_image = input_image
    elif display_mode == "fit":
        resized_image = ImageOps.fit(input_image, target_size, method=Image.Resampling.LANCZOS)
    else:
        resized_image = ImageOps.pad(input_image, target_size, method=Image.Resampling.LANCZOS, color=pad_color)

    # Create a palette object
    if use_km:
        IMAGE_PROCESSOR.diffuse_image(resized_image)
        quantized_image = resized_image
    else:
        pal_image = Image.new("P", (1, 1))
        if bw:
            palette = BW_PALETTE
        else:
            palette = blend_palette(palette_blend_ratio)
        pal_image.putpalette(palette)
        quantized_image = resized_image.quantize(palette=pal_image)

    # make filename
    output_dir.mkdir(exist_ok=True)
    if use_km:
        extra_string = "_km"
    elif bw:
        extra_string = "_bw"
    else:
        extra_string = f"_{int(palette_blend_ratio * 100)}"
    image_name = f"{input_filename.stem}_{display_mode}{extra_string}_converted.bmp"
    output_filename = output_dir / image_name

    if bw:
        quantized_image.save(output_filename, format="bmp")
    else:
        # save to buffer
        buffer = BytesIO()
        quantized_image.save(buffer, format="bmp")
        buffer.seek(0)

        # Save output image
        wand_image = WandImage(file=buffer, format="bmp")
        wand_image.flop()
        wand_image.type = "palette"
        wand_image.save(filename=output_filename)

    print(f"Successfully converted {input_filename} to {output_filename}")


def main() -> None:
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process some images.")

    # Add orientation parameter
    parser.add_argument("image_file", help="Input image file")
    parser.add_argument(
        "--dir",
        choices=["landscape", "portrait"],
        default="landscape",
        help="Image orientation (landscape or portrait)",
    )
    parser.add_argument("--mode", choices=["fit", "pad"], default="pad")
    parser.add_argument("--bw", action=argparse.BooleanOptionalAction, help="Black and white mode")
    parser.add_argument(
        "--use-km", action=argparse.BooleanOptionalAction, help="Use Kode Munkie's algorithm for dithering"
    )

    args = parser.parse_args()
    input_filename = Path(args.image_file)
    output_dir = Path("converted/")

    # Check whether the input file exists
    if not input_filename.exists():
        print(f"Error: file {input_filename} does not exist")
        sys.exit(1)

    is_dir = input_filename.is_dir()
    print(f"{is_dir=}")
    palette_blend_ratio = 1/3
    if is_dir:
        filelist = input_filename.glob("**/*.png")
        start = perf_counter()
        with ThreadPoolExecutor() as executor:
            for input_file in filelist:
                arguments = (input_file, args.dir, args.mode, args.bw, output_dir, palette_blend_ratio, args.use_km)
                executor.submit(convert, *arguments)
        print(f"Processing all images took {perf_counter()-start:.3f} seconds")
    else:
        convert(input_filename, args.dir, args.mode, args.bw, output_dir, palette_blend_ratio, args.use_km)


if __name__ == "__main__":
    main()
