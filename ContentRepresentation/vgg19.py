import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

def get_vgg19():
    return vgg19.VGG19(include_top = False,
                       weights = 'imagenet')
    