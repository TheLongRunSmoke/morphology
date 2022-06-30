import argparse
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Morphology.")
parser.add_argument(
    'files',
    nargs='+',
    type=lambda p: Path(p).absolute(),
    help='Path to a file. *.jpg only.')
args = parser.parse_args()


def to_odd(num):
    return num if num % 2 != 0 else num + 1


def process_image(image):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Resize image to be smaller than 1000px in any direction to keep kernel size reasonable.
    width, height = image.size
    scale = ceil(max(width, height) / 1000)
    cv_image = cv2.resize(cv_image, (int(width / scale), int(height / scale)))

    # Apply CLAHE to adaptively equalize hist.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cv_image = clahe.apply(cv_image)

    # Apply morphology to remove small features. Blur...kind of.
    kernel = np.ones((3, 3), np.uint8)
    cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_OPEN, kernel)
    cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_OPEN, kernel)

    # Separate horizontal and vertical lines to filter out spots.
    kernel = np.ones((9, 3), np.uint8)
    vert = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 9), np.uint8)
    horiz = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)

    # Combine
    comb = cv2.add(horiz, vert)

    return Image.fromarray(comb)


for file in args.files:
    print('Process: %s ...' % file)
    path = Path(file)
    with Image.open(path).convert(mode='RGB') as raw:
        original_size = raw.size
        result = process_image(raw).resize(raw.size, resample=Image.Resampling.LANCZOS)
        result.save(
            format='JPEG',
            fp=path.with_stem("%s_morph" % path.stem)  # Use original file name with appends.
        )
        # Tel user that we done with this image.
        print('Process: %s - OK' % file)
