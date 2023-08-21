import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

import pathlib

# display the version of tensorflow
print(tf.__version__)

# display current directory
# current_directory = os.path.dirname(os.path.abspath(__file__))
# print("Current directory:", current_directory)

# set the download directory
download_dir = pathlib.Path("/Users/tiff/Projects/tensorflow-practice")

# download and prepare the dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True, cache_dir=download_dir)
data_dir = pathlib.Path(archive).with_suffix('')

# display the dataset directory
print("Dataset directory:", data_dir)

# display the number of total images: 3670
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# load and display an image
roses = list(data_dir.glob('roses/*'))
img = PIL.Image.open(str(roses[1]))
img.show('Image')

# create a dataset: define parameters for the loader
batch_size = 32
img_height = 180
img_width = 180

# load data using a Keras utility
# use validation split when developing your model
# 80% of the images for trianing and 20% for validation
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)