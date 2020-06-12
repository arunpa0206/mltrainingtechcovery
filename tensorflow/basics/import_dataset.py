import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


# Construct a tf.daatta.Dataset
ds = tfds.load('mnist', split='train', shuffle_files=True)

# Build your input pipeline
ds = ds.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
for example in ds.take(1):
  image, label = example["image"], example["label"]

  print("label:",label)
