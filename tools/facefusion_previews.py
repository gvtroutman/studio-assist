"""Normalize profile thumbnails in the FaceFusion environment, never in Tk."""
import json
from pathlib import Path
import sys
from PIL import Image, ImageOps

for source, destination in json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')):
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert('RGBA')
            image.thumbnail((220, 220))
            canvas = Image.new('RGBA', (220, 220), (255, 255, 255, 255))
            canvas.alpha_composite(image, ((220-image.width)//2, (220-image.height)//2))
            canvas.convert('RGB').save(destination)
    except (OSError, ValueError):
        continue
