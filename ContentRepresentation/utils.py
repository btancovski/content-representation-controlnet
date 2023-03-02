import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from PIL import Image
from os.path import isdir, exists, join, abspath, splitext, basename
import glob

def tf_read_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def tf_resize_image(img, height, width):
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max(height, width) / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def preprocess_image(path, height = 128, width = 128):
    img = tf_read_image(path)    
    img = tf_resize_image(img, height, width)
    return tf.Variable(img)


def create_random(path, height, width, seed = 42):
    np.random.seed(seed)
    img = tf_read_image(path)
    img = tf.Variable(np.random.uniform(0, 1, tf.Variable(img).shape))
    img = tf_resize_image(img, height, width)
    return tf.Variable(img)


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)


def get_generated_image_path(path, suffix):

    path = splitext(abspath(path))
    path = path[0] + suffix + path[1]
    return path


def save_image(image, path, suffix):
    image.save(get_generated_image_path(path, suffix))


def validate_file(file):
    try:
        Image.open(file)
        tf.io.read_file(file)
    except:
        raise


def validate_directory(dir):
    if not exists(abspath(dir)):
        raise FileNotFoundError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILES = 2000
FROM = 0
TO = 5000
def process_directory(dir):
    glob_expr = join(abspath(dir), '**')
    files = []
    for filepath in glob.glob(glob_expr, recursive=True):
        if not isdir(filepath) and str.lower(splitext(filepath)[1]) in IMAGE_EXTENSIONS:
            validate_file(filepath)
            files.append(filepath)
    # return files
    return sorted(files, key=lambda file: int(file.split("_")[-1].split(".")[0]))[FROM:TO]


def process_input(file, dir):
    if not len(file) and not len(dir):
        print(f"No input images given.")
        exit(0)
    if not len(file) and len(dir):
        try:
            validate_directory(dir)
            return process_directory(dir)
        except Exception as e:
            print(f"Failed to process directory {dir}. {e}")
    if not len(dir) and len(file):
        try:
            validate_file(file)
            return [file]
        except Exception as e:
            print(f"Failed to process file {file}. {e}")
    

PROMPT_MAP = 'data/input/tiny-imagenet-200/words.txt'
MAP = None
import pandas as pd

def read_map():
    global MAP
    MAP = pd.read_csv(PROMPT_MAP, sep='\t', header = None, names = ["w_id", "prompt"])


def extract_prompt(file_name):
    global MAP
    if MAP is None:
        read_map()
    w_id = splitext(basename(file_name))[0].split("_")[0]
    return MAP[MAP['w_id'] == w_id]['prompt'].values
    
import json
JSON_FILE = "data/output/prompt.json"
def create_json_file(dir):
    json_data = []
    glob_expr = join(abspath(dir), '**')
    for dir2 in glob.glob(glob_expr, recursive=False):
        if isdir(dir2):
            img_data = {"source": "", "target": "", "prompt": ""}
            glob_expr2 = join(abspath(dir2), '**')
            for file in glob.glob(glob_expr2, recursive=False):
                basefile = basename(file)
                if "_original" in basefile:
                    img_data['target'] = file
                elif "_final" in basefile:
                    img_data['source'] = file
                elif "prompt" in basefile:
                    with open(file) as f:
                        img_data['prompt'] = f.readlines()[0]

            json_data.append(img_data)
            # print(img_data)
    with open(JSON_FILE, "w") as f:
        f.writelines((str(i) + '\n' for i in json_data))

if __name__ == "__main__":
    create_json_file('data/output/tiny-imagenet/generated_i400_v5w1/')
