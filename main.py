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

# configure the dataset for performance: use buffered prefetching
# when loading data, use: Dataset.cache and Dataset.prefetchAUTOTUNE = tf.data.AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# train a simple model using the datasets we just prepared
# the Sequential model
num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# Choose the tf.keras.optimizers.Adam optimizer and tf.keras.losses.SparseCategoricalCrossentropy loss function
# To view training and validation accuracy for each training epoch, pass the metrics argument to Model.compile.
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# # train model by passing datasets to model.fit
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

# Keras preprocessing utility - tf.keras.utils.image_dataset_from_directory is a convenient way to create a tf.data.Dataset from a directory of images
# use tf.data to write your own input pipeline for finer grain control
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

for f in list_ds.take(5):
  print(f.numpy())

# the tree structure of the files can be used to compile a class_names list
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

# split the dataset into training and validation sets
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

# print the length of each dataset
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

# write a function that converts a file path to an (img, label) pair
def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# uset Dataset.map to create a dataset of image, label pairs
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

# configure dataset for performance
# to train a model with this dataset you want the data to be: well shuffled, be batched, batches to be available asap
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

# visualize this dataset similarly to the one we created previously
image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(class_names[label])
  plt.axis("off")
plt.show()

# continue training the model using the tf.data.Dataset
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)