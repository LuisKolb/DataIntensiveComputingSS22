# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Unused imports
#import detect
#import tflite_runtime.interpreter as tflite
#import platform
#import cv2
#import time
#import io
#import random

# For the flask app.
from flask import Flask, request, Response, jsonify

# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

# utils
import os
from datetime import datetime
import json
import re


# Print Tensorflow version
print(tf.__version__)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.compat.v1.enable_eager_execution()  # necessary feature!

#
# Initialize the flask app
#
app = Flask(__name__)

print('[INFO] Started Flask App.')

#
# setup
#
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']


#
# function definitions for image processing
#

def save_image(image, path):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imsave(path, image)
    plt.close()
    print("Annotated Image saved to %s" % path)
    return path


def resize_image(path, new_width=256, new_height=256, display=False):
    _, new_filename = tempfile.mkstemp(suffix=".jpg")
    pil_image = Image.open(path)
    pil_image = ImageOps.fit(
        pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(new_filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % new_filename)
    if display:
        # display_image(pil_image)
        print("Displaying images is not supported! called with path: %s" % path)
    return new_filename


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                           int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image

#
# function definitions for object detection
#


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def run_detector(detector, path, save_annotated_img=''):
    img = load_img(path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[
        tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    #print("Found %d objects." % len(result["detection_scores"]))
    #print("Inference time: ", end_time-start_time)

    image_with_boxes = draw_boxes(
        img.numpy(), result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"])

    if save_annotated_img:
        save_image(image_with_boxes, save_annotated_img)

    output_dict = {
        'objects_found': len(result["detection_scores"]),
        'detection_class_entities': [e.decode('ASCII') for e in result["detection_class_entities"].tolist()],
        'detection_scores': result["detection_scores"].tolist(),
        'inference_time': end_time-start_time,
        'annotated_image_path': save_annotated_img,
    }

    return output_dict

# params
# ------
# filename_image: name of the file to run detection on
# path: local folder where the filename_image is located
# output: boolean, wheter annotated images should be saved to local storage


def detection_loop(filename_image, path, output):
    start_time = time.time()
    resized_img_path = resize_image(path + filename_image, 640, 480)

    output_dir = path + 'output/'  # where results are saved
    if output:
        save_path = output_dir + filename_image
    else:
        save_path = ''

    result = run_detector(detector, resized_img_path, save_path)
    end_time = time.time()

    result['total_det_loop_time'] = end_time-start_time

    # save the results dict for the image
    with open(output_dir + filename_image + '.json', 'w', encoding='utf-8') as res_file:
        json.dump(result, res_file, ensure_ascii=False, indent=4)

    return result

#
# Routing http posts to this method
#


@app.route('/api/detect', methods=['POST', 'GET'])
def main():
    #img = request.files["image"].read()
    #image = Image.open(io.BytesIO(img))
    #data_input = request.args['input']
    #output = request.form.get('output')

    print("call received")
    # endpoint will be called like http://url:port/api/detect -d "input=/images/filename.jpg&output=1"

    # path+name of the image, which is saved locally
    data_input = request.values.get('input')
    if not data_input:
        return Response(status=400)


    # (is a flag) if not empty annotated images should be saved to local storage
    if request.values.get('output'):
        output = True
    else:
        output = False

    print("File      Path:", Path(__file__).absolute())
    print("Directory Path:", Path().absolute())
    path = data_input
    print("Given path:", path)
    filename = ''

    input_format = ["jpg", "png", "jpeg"]
    if data_input.find(".") != -1:
        print(data_input + " is a file")
        split_data_input = data_input.split(".", 1)
        if data_input.endswith(tuple(input_format)):
            print("INPUT FORMAT: %s IS VALID" % split_data_input[1])
            path_splitted = []
            path_splitted = re.split('/', data_input)
            # in the exmaple above, would be 'filename.jpg'
            filename = path_splitted[len(path_splitted)-1]
            # in the exmaple above, would be '/images/'
            path = os.path.dirname(data_input) + '/'
    else:
        print(data_input + " is a path with the following files: ")
        for filename in os.listdir(data_input):
            print("  " + filename)
        # return 400 BAD REQUEST since path is a directory, not an image
        return Response(status=400)

    res_dir = detection_loop(filename, path, output)

    return jsonify(res_dir), 200  # return results and 200 OK


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
