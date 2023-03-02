import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from PIL import Image
from os import path, makedirs
import time
import traceback
# custom
from vgg19 import get_vgg19
from utils import *
from tensorflow_config import init_tensorflow

DEBUG_MODE = False
SEED = 42
EXTRACT_PROMPTS = False

MODEL = None
HEIGHT = 256
WIDTH = 256
CHANNELS = 3
ITERATIONS = 600

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate = 0.02, beta_1 = 0.99, epsilon = 1e-1)

CONTENT_LAYERS = ["block5_conv1", "block5_conv2"]
CONTENT_WEIGHTS = [1, 5]
CONTENT_WEIGHTS = CONTENT_WEIGHTS / np.sum(CONTENT_WEIGHTS)
CONTENT_WEIGHT = 1
VARIATION_WEIGHT = 5
SUFFIX = '_generated'
NAME = 'generated'
SAVE_LOCATION = f'data/output/{NAME}/generated_i{ITERATIONS}_v{VARIATION_WEIGHT}w{CONTENT_WEIGHT}/'

def content_loss(content, generated):
    return 0.5 * tf.reduce_sum(tf.square(content - generated))


def variation_loss(generated):
    
    x = generated[:, :, 1:, :] - generated[:, :, :-1, :]
    y = generated[:, 1:, :, :] - generated[:, :-1, :, :]
    return tf.reduce_sum(tf.abs(x)) + tf.reduce_sum(tf.abs(y))


def total_loss(content_image, generated_image):

    features = FEATURE_EXTRACTOR(tf.concat([content_image, generated_image], axis = 0))

    # CONTENT LOSS
    c_loss = tf.zeros(shape=())
    for i, content_layer in enumerate(CONTENT_LAYERS):

        layer = features[content_layer]
        content_features = layer[0, :, :, :]
        generated_features = layer[-1, :, :, :]
        c_loss += content_loss(content_features, generated_features) * CONTENT_WEIGHT * CONTENT_WEIGHTS[i]

    # VARIATION LOSS
    v_loss = variation_loss(generated_image) * VARIATION_WEIGHT
    
    return c_loss + v_loss


def get_model():
    model = get_vgg19()
    model.trainable = False
    return model


def get_extractor():

    outputs_dict = dict([(layer.name, layer.output) for layer in MODEL.layers])
    feature_extractor = keras.Model(inputs=MODEL.inputs, outputs=outputs_dict)
    
    return feature_extractor


def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

@tf.function()
def train_step(content_image, generated_image):
    with tf.GradientTape() as tape:
        loss = total_loss(content_image, generated_image)
    grads = tape.gradient(loss, generated_image)
    OPTIMIZER.apply_gradients([(grads, generated_image)])
    generated_image.assign(clip_0_1(generated_image))

    return loss


def train(content_image, generated_image, save_loc = None, file = None):

    start_time = time.time()
    for i in range(1, ITERATIONS + 1):

        loss = train_step(content_image, generated_image)

        if i % 100 == 0:
            end_time = time.time()
            print(f"Iteration {i} - Loss (thousands) {(loss / 1e3):.2f} - Time: {np.round(end_time - start_time, 2)}s")
            if save_loc is not None:
                save_image(tensor_to_image(generated_image), save_loc + file, SUFFIX + f"_{i}i")
            start_time = time.time()

    return generated_image


def generate_image(file):
    file_name = path.basename(file)
    save_loc = path.join(SAVE_LOCATION, splitext(file_name)[0] + '/')
    makedirs(save_loc, exist_ok=True)

    content_image = preprocess_image(file, HEIGHT, WIDTH)
    random_image = create_random(file, HEIGHT, WIDTH, SEED)

    print("Image shapes:", content_image.shape, random_image.shape)
    save_image(tensor_to_image(random_image), save_loc + file_name, SUFFIX + f"_start")
    save_image(tensor_to_image(content_image), save_loc + file_name, SUFFIX + f"_original")

    generated_image = train(content_image, random_image, save_loc, file_name)
    save_image(tensor_to_image(generated_image), save_loc + file_name, SUFFIX + f"_final")

    if EXTRACT_PROMPTS:
        with open(f"{save_loc}prompt.txt", "w") as f:
            prompt = extract_prompt(file_name)
            if isinstance(prompt, np.ndarray):
                prompt = prompt[0]
            f.write(prompt)

def init_args(args):
    global HEIGHT, WIDTH, CHANNELS, SEED, DEBUG_MODE, ITERATIONS, SAVE_LOCATION, VARIATION_WEIGHT, CONTENT_WEIGHT, NAME, EXTRACT_PROMPTS
    if not len(args.file) and not len(args.directory):
        print(f"[ERROR] Input file or directory required.")
        exit(1)
    if args.height:
        if args.height %32 != 0 or args.height > 768 or args.height < 32:
            HEIGHT = (min((max(args.height, 32), 768)) // 32 ) * 32
            print(f"Height not valid, setting width to {HEIGHT}.")
        else:
            HEIGHT = args.height
    if args.width:
        if args.width %32 != 0 or args.width > 768 or args.width < 32:
            WIDTH = (min((max(args.width, 32), 768)) // 32 ) * 32
            print(f"Width not valid, setting width to {WIDTH}.")
        else:
            WIDTH = args.width
    if args.iterations:
        if 0 < args.iterations < 10000:
            ITERATIONS = args.iterations
        else:
            print(f"Iteration amount not valid, setting to {ITERATIONS}.")
    if args.variation:
        VARIATION_WEIGHT = args.variation
    if args.content:
        CONTENT_WEIGHT = args.content
    if len(args.name):
        NAME = args.name
    SEED = args.seed
    DEBUG_MODE = args.print
    EXTRACT_PROMPTS = args.extract
    SAVE_LOCATION = f'data/output/{NAME}/generated_i{ITERATIONS}_v{VARIATION_WEIGHT}w{CONTENT_WEIGHT}/'


def parse_args():
    parser = argparse.ArgumentParser(description="Create content representation of a given image.")
    parser.add_argument("-f", "--file", type=str, help="The location to the image file.", default="")
    parser.add_argument("-d", "--directory", type=str, help="Location to a directory containing images", default="")
    parser.add_argument("-p", "--print", action=argparse.BooleanOptionalAction, help="A flag to set DEBUG_MODE to true.")
    parser.add_argument("-y", "--height", type=int, help="Resulting image height.", default=HEIGHT)
    parser.add_argument("-x", "--width", type=int, help="Resulting image width.", default=WIDTH)
    parser.add_argument("-v", "--variation", type=int, help="Variation weight.", default=VARIATION_WEIGHT)
    parser.add_argument("-c", "--content", type=int, help="Content weight.", default=CONTENT_WEIGHT)
    parser.add_argument("-s", "--seed", type=int, help="Seed for the random image.", default=SEED)
    parser.add_argument("-i", "--iterations", type=int, help="Number of iterations for image recreation.", default=ITERATIONS)
    parser.add_argument("-o", "--name", type=str, help="(Base) Name of the output directory inside /data.", default=NAME)
    parser.add_argument("-e", "--extract", action=argparse.BooleanOptionalAction, help="Flag to extract classes for generated images from given words.txt file. Refer to the words.txt file from tiny-imagenet-200. Changes to extract_prompts.py required.")
    return parser.parse_args()


def main():
    global MODEL, FEATURE_EXTRACTOR, OPTIMIZER
    start_time = time.time()
    init_tensorflow()
    args = parse_args()
    init_args(args)
    MODEL = get_model()
    FEATURE_EXTRACTOR = get_extractor()
    files = process_input(args.file, args.directory)
    makedirs(SAVE_LOCATION, exist_ok = True)
    try:
        total_files = len(files)
        for i, file in enumerate(files):
            OPTIMIZER = tf.keras.optimizers.Adam(learning_rate = 0.02, beta_1 = 0.99, epsilon = 1e-1)
            print("="*50)
            print(f"[INFO] ({i+1}/{total_files}) Starting to process {path.basename(file)} ...")
            image_start_time = time.time()
            generate_image(file)
            print(f"[SUCCESS] Finished processing {path.basename(file)} in {np.round(time.time() - image_start_time, 2)} seconds.")

        print("="*50)
        print(f"[SUCCESS] Finished processing all images in {np.round(time.time() - start_time, 2)} seconds.")
        print("="*50)
        if EXTRACT_PROMPTS:
            print(f"Creating prompt.json ...")
            create_json_file(SAVE_LOCATION)
    except Exception as e:
        print(f"[ERROR] Failed to process all files. {e}")
        if DEBUG_MODE:
            traceback.print_exc()


if __name__ == "__main__":
    main()