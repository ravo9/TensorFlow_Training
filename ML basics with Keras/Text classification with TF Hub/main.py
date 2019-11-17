from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

# !pip install -q tensorflow-hub
# !pip install -q tensorflow-datasets
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

# Let's print first 10 examples.
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
#print(train_examples_batch)

# Let's also print the first 10 labels.
#print(train_labels_batch)

# Let's first create a Keras layer, and try it out on a couple of input examples.
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
# hub_layer(train_examples_batch[:3])

# Let's now build the full model:
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# Now, configure the model to use an optimizer and a loss function.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
