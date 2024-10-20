# unveiling-speech-emotion-using-CNN
Speech Emotion Recognition (SER) system using the RAVDESS dataset and a Convolutional Neural Network (CNN) to classify emotions from speech. It extracts key audio features—MFCC, Zero Crossing Rate (ZCR), and Chroma—to capture spectral and temporal information for accurate emotion classification.

1. Dataset (RAVDESS):
The system begins by using the RAVDESS dataset, which consists of labeled emotional speech samples. This dataset provides raw audio input, necessary for training and evaluating the model.

2. Preprocessing:
   In this phase, raw audio signals are enhanced through several techniques:
* Pitch Shift: Alters the pitch of the audio to increase the diversity of training data.
* Noise Addition: Introduces background noise to make the model more robust to real-world noisy environments.
* Stretching: Slightly changes the duration of the audio without affecting the pitch, further augmenting the dataset.
These steps ensure that the data is suitable for feeding into the model, making it resilient to variations in real-world audio conditions.

3. Feature Extraction:

 * MFCC (Mel-Frequency Cepstral Coefficients): Extracts key spectral features of the audio, which helps in capturing the timbre and tone.
 * ZCR (Zero-Crossing Rate): Computes the rate at which the signal changes its sign (from positive to negative and vice versa), providing information about the signal’s temporal characteristics.
 * Chroma Features: Encodes pitch class information and helps in capturing tonal aspects of the speech.
These features are critical for capturing both the temporal and spectral dynamics of the speech, which are key indicators of emotion.

4. Model Training:

The extracted features are fed into a Convolutional Neural Network (CNN) model. The CNN consists of layers such as:
 * Convolutional Layers: For extracting local patterns from the input features.
 * Pooling Layers: For dimensionality reduction and selecting important features.
 * Fully Connected Layers: For final classification of emotions.
 * Dropout Regularization and Early Stopping techniques are applied during training to prevent overfitting and ensure optimal performance.
The training process leverages supervised learning to map audio features to specific emotion labels.

5.Testing and Prediction:

* After the model is trained, it is evaluated on new audio samples. During the testing phase, the model processes the new input audio in the same way (preprocessing and feature extraction) and predicts the emotion class.
* The predicted emotions are output as one of the predefined categories (e.g., happy, sad, angry, neutral).
