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
