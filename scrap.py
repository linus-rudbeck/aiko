# Function to reverse the preprocessing for TensorFlow so that the image can be displayed
def reverse_preprocess_tensorflow(image: np.ndarray, label=0) -> tuple:
    image = tf.convert_to_tensor(image)
    image_float32 = tf.cast(image, tf.float32)
    image_float32 = tf.math.multiply(image_float32, 255.0)
    image_int8 = tf.cast(image_float32, tf.uint8)
    return image_int8, label