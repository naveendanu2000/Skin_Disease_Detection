from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from flask import Flask, render_template, request
import numpy as np
import cv2

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_message = ""
    probability_message = ""
    predicted_label = ""
    if 'image' in request.files:
        model = load_model('skin_disease.h5')
        img = request.files['image']
        np_arr = np.frombuffer(img.read(), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = preprocess_input(img)
        img = cv2.resize(img, (224, 224))
        prediction = model.predict(np.expand_dims(img, axis=0))

        probability = np.max(prediction)
        predicted_class_index = np.argmax(prediction)
        class_labels = ['Bacterial Infection - Cellulitis', 'Bacterial Infection - Impetigo', 'Fungal Infection - Athlete Foot', 'Fungal Infection - Nail Fungus', 'Fungal Infection - Ringworm',
                        'PA - Cutaneous Larva Migrans', 'Viral Infection - Chickenpox', 'Viral Infection - Shingles']
        predicted_label = class_labels[predicted_class_index]

        if probability.astype('float') > 0.5000000:
            prediction_message = 'Predicted Infection: ' + predicted_label
            probability_message = 'Probability: ' + probability.astype('str')
        else:
            prediction_message = 'No skin disease detected'

    return render_template('prediction.html', prediction_message=prediction_message, probability_message=probability_message, predicted_label=predicted_label)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
