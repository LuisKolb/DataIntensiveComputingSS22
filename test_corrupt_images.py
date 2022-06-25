#
# used to test if PIL can open all the images in a directory, specify the path below 
#

import PIL
from pathlib import Path
from PIL import UnidentifiedImageError, Image

path = Path("/app/images/").rglob("*.jpg")
for img_p in path:
    try:
        img = Image.open(img_p)
    except PIL.UnidentifiedImageError:
            print(img_p)