from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
import librosa
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'E:\front end\UPLOAD_FOLDER'
# Load the pre-trained model and preprocessing objects
loaded_model = load_model('best_model1_weights.h5')
scaler2 = pickle.load(open('scaler2.pickle', 'rb'))
encoder2 = pickle.load(open('encoder2.pickle', 'rb'))

# Function to extract features from audio
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def chroma(data, sr, frame_length=2048, hop_length=512, flatten=True):
    chroma_result = librosa.feature.chroma_stft(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(chroma_result.T) if not flatten else np.ravel(chroma_result.T)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_result = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(mfcc_result.T) if not flatten else np.ravel(mfcc_result.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        chroma(data, sr, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                       ))
    return result

# Function to preprocess audio data and make predictions
def predict_emotion(audio_path):
    data, sr = librosa.load(audio_path, duration=2.5, offset=0.6)
    features = extract_features(data)
    result = np.array(features)
    result = np.reshape(result, newshape=(1, 3564))
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    predictions = loaded_model.predict(final_result)
    predicted_emotion = encoder2.inverse_transform(predictions)
    return predicted_emotion[0][0]

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text='No selected file')
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        if filename.endswith('.wav'):
            prediction = predict_emotion(file_path)
        else:
            prediction = "Unsupported file format"
        return render_template('index.html', prediction_text=prediction)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
