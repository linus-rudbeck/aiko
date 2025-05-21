from flask import Flask, render_template, request
import uuid
import image_dataset_utils as utils
import tensorflow as tf

loaded_model = tf.keras.models.load_model("car_classifier_model.h5")
loaded_class_names = utils.load_class_names_from_file("augmented_cars/class_names.txt")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_path = save_image()
        
        loaded_image, _ = utils.load_one_image(image_path)
        predictions = loaded_model.predict(loaded_image)
        predicted_class = tf.argmax(predictions[0]).numpy()
        
        predicted_class_name = loaded_class_names[predicted_class]
        
        return render_template('index.html', predicted_class=predicted_class_name)
    return render_template('index.html')

def save_image():
    image = request.files['image']
    image_name = image.filename
    unique_name = f"{uuid.uuid4()}_{image_name}"
    image_path = 'images/' + unique_name
    image.save(image_path)
    return image_path

if __name__ == '__main__':
    app.run(debug=True)