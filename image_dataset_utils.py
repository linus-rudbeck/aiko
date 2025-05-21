import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# Preprocess one image by converting it to a tensor and normalizing it
def preprocess_tensorflow(image, label=0):
    image = tf.convert_to_tensor(image)
    image_float32 = tf.cast(image, tf.float32)
    image_float32 = tf.math.divide(image_float32, 255.0)
    return image_float32, label


# Load one image from a file path and preprocess it
def load_one_image(image_path, label=0):
    image = Image.open(image_path).resize((128, 128))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image, label = preprocess_tensorflow(image, label)
    return image, label


# Show one image with its label
def show_image(preprocessed_image, label) -> None:
    label = str(label)
    plt.figure(figsize=(5, 5))
    plt.imshow(preprocessed_image[0])
    plt.title(label)
    plt.axis("off")
    plt.show()

# Load and preprocess image data from a directory
def load_and_preprocess_image_data(directory, batch_size):
    image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=(128, 128),
        batch_size=batch_size,
        labels="inferred",
        label_mode="int",
    )

    preprocessed_image_dataset = image_dataset.map(
        lambda x, y: (preprocess_tensorflow(x, y))
    )
    
    preprocessed_image_dataset.class_names = image_dataset.class_names
    
    return preprocessed_image_dataset

# Show a sample of images from a dataset
def show_images(dataset, class_names) -> None:
    plt.figure(figsize=(12, 8))
    for images, labels in dataset:
        for i in range(12):
            if i >= len(images):
                break
            ax = plt.subplot(3, 4, i + 1)
            plt.imshow(images[i])
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()


# Function to count total number of images in the dataset
def count_dataset(dataset):
    count = 0
    for images, _ in dataset:
        count += len(images)
    return count


# Function to show information like shape, dtype, mean of images in the dataset
def show_dataset_info(dataset):
    print(type(dataset))

    if hasattr(dataset, "class_names"):
        print("Class names:", dataset.class_names)

    dataset_count = count_dataset(dataset)
    print("Total number of images in the dataset:", dataset_count)

    for a, b in dataset:
        print("Images shape:", a.shape)
        print("Labels shape:", b.shape)
        print("Images dtype:", a.dtype)
        print("Labels dtype:", b.dtype)
        print("Images mean:", tf.reduce_mean(a))
        break

    print("")


# Function to augment an image dataset
def augment_image_dataset(preprocessed_image_dataset, augmentations, batch_size):
    data_augmentation = get_data_augmentation_model()

    augmented_images = []
    augmented_labels = []

    # Count the number of images in the dataset
    total = count_dataset(preprocessed_image_dataset) * augmentations
    tqdm_bar = tqdm(total=total, desc="Augmenting images")
    
    for image, label in preprocessed_image_dataset.unbatch():
        for _ in range(augmentations):
            augmented_image = data_augmentation(tf.expand_dims(image, axis=0), training=True)
            augmented_images.append(tf.squeeze(augmented_image, axis=0))
            augmented_labels.append(label)
            tqdm_bar.update(1)
            
    tqdm_bar.close()

    augmented_images = tf.stack(augmented_images)
    augmented_labels = tf.stack(augmented_labels)

    augmented_dataset = tf.data.Dataset.from_tensor_slices((augmented_images, augmented_labels))

    combined_image_dataset = preprocessed_image_dataset.unbatch().concatenate(augmented_dataset)
    combined_image_dataset = combined_image_dataset.shuffle(1000).batch(batch_size)
    combined_image_dataset.class_names = preprocessed_image_dataset.class_names
    
    return combined_image_dataset

# Create a data augmentation model
def get_data_augmentation_model():
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2)
        ]
    )
    
    return data_augmentation


# Save class names to a text file
def save_class_names_to_file(class_names, file_path):
    with open(file_path, "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

# Load class names from a text file
def load_class_names_from_file(file_path):
    with open(file_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Split a dataset into training and validation sets
def split_dataset(dataset, batch_size, validation_split=0.2):
    dataset_size = count_dataset(dataset)
    validation_size = int(dataset_size * validation_split)
    training_size = dataset_size - validation_size

    dataset = dataset.unbatch()
    train_dataset = dataset.take(training_size)
    val_dataset = dataset.skip(training_size)

    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    val_dataset = val_dataset.shuffle(1000).batch(batch_size)
    
    return train_dataset, val_dataset