from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
import tensorflow as tf
from io import BytesIO
from tensorflow.keras.preprocessing.image import array_to_img,img_to_array,load_img 

# Load the preprocessing function from disk
filename = 'preprocessing.pkl'
with open(filename, 'rb') as f:
    preprocess_input = pickle.load(f)

# Load the Keras model from disk
model = tf.keras.models.load_model('model.h5')

# Load the SVM model from disk
filename = 'svc_model.pkl'
clf = pickle.load(open(filename, 'rb'))

# Initialize the Flask app
app = Flask(__name__)

# category
Categories=[ 'Peace',  "WhatsUp",  'Me',  'Bad',  'House',  'Good']

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the form
    file = request.files['image']
    
    # Load the image from the file object
    img_bytes = file.read()
    img = load_img(BytesIO(img_bytes), target_size=(224, 224))
    
    # Preprocess the image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Get the prediction from the model
    prediction = model.predict(img_array)
    flat_arr = prediction.flatten()
    
    # Get the final prediction from the SVM
    final_prediction = clf.predict(flat_arr.reshape(1,-1))
    final_prob = clf.predict_proba(flat_arr.reshape(1,-1))[0][final_prediction[0]]
    
    # Return the predicted class as JSON
    return jsonify({'class': Categories[final_prediction[0]],
                   'prob': final_prob})

if __name__ == '__main__':
    app.run(debug=True)