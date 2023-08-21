import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

import pathlib

import matplotlib.pyplot as plt

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

# find the class names in the class_names attributes on these datasets
class_names = train_ds.class_names
print(class_names)

# visualize the data - create visualizations and plots using matplotlib
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

# manually iterate over the dataset and retrieve batches of images
# image_batch is a tensor of the shape (32, 180, 180, 3)
# this is a batch of 32 images of shape 180x180x3
# label_batch is a tensor of the shape (32, ), thse are corresponding labels to the 32 images
# you can call .numpy() on either of these tensors to convert them to a numpy.ndarray
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# standardize the data
# for neural network, seek to make RGB channel input values small
# standardize values to be in the [0, 1] range by using tf.keras.layers.Rescaling
normalization_layer = tf.keras.layers.Rescaling(1./255)

# There are two ways to use this layer. You can apply it to the dataset by calling Dataset.map:
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# or you can include the layer inside your model definition to simplify deployment