import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import image_dataset_utils as utils

# KMP_DUPLICATE_LIB_OK to avoid errors due to multiple libraries loading
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Set the logging level to suppress TensorFlow warnings
# tf.get_logger().setLevel("ERROR")


class Config:
    OUTPUT_CLASS_COUNT = 3  # Number of target classes
    AUGMENTATIONS = 1  # Number of augmentations per image
    BATCH_SIZE = 32  # Number of images per batch



print("Loading training data...")
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=(128, 128),
    batch_size=Config.BATCH_SIZE,
    labels="inferred",
    label_mode="int",
)
class_names = train_data.class_names  # type: ignore

preprocessed_train_data = train_data.map(lambda x, y: (utils.preprocess_tensorflow(x, y)))  # type: ignore
preprocessed_train_data.class_names = train_data.class_names  # type: ignore
print("")

print("\n\n== Training data ==")
utils.show_dataset_info(train_data)  # type: ignore

print("\n\n== Preprocessed training data ==")
utils.show_dataset_info(preprocessed_train_data)  # type: ignore




def generate_augmented_dataset(dataset):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            # tf.keras.layers.RandomRotation(0.5),
            # tf.keras.layers.RandomZoom(0.5),
        ]
    )
    
    augmented_images = []
    augmented_labels = []

    for image, label in dataset.unbatch():
        for _ in range(Config.AUGMENTATIONS):
            augmented_image = data_augmentation(tf.expand_dims(image, axis=0), training=True)
            augmented_images.append(tf.squeeze(augmented_image, axis=0))
            augmented_labels.append(label)  # label is a scalar

    # Stack the tensors
    augmented_images = tf.stack(augmented_images)
    augmented_labels = tf.stack(augmented_labels)

    return tf.data.Dataset.from_tensor_slices((augmented_images, augmented_labels))

augmented_train_data = generate_augmented_dataset(preprocessed_train_data)  # type: ignore


print("\n\n== Augmented training data ==")
utils.show_dataset_info(augmented_train_data)  # type: ignore