# Speech_Emotion_Analysis

Analysing and detection emotions in an audio file using Deep Learning Methods 
This repository contains a deep learning model for **Speech Emotion Analysis** that classifies audio samples into one of six emotion classes: **angry, sad, fear, happy, neutral, and disgust**. The model leverages **MFCC (Mel-frequency cepstral coefficients)** for feature extraction and a **Bi-directional LSTM** network for sequence modeling.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)


## Introduction
This project aims to develop an efficient model that can recognize human emotions from audio signals. Emotion recognition plays a crucial role in many applications, such as virtual assistants, healthcare, and entertainment. The model can distinguish between six different emotions: **angry, sad, fear, happy, neutral, and disgust**.

## Dataset
The dataset used contains various audio samples labeled with their corresponding emotions. Each audio file represents a spoken word or phrase. Emotions present in the dataset include:
- **Angry**
- **Sad**
- **Fear**
- **Happy**
- **Neutral**
- **Disgust**

The dataset can be accessed using the below link:
https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess


## Preprocessing
The audio signals are preprocessed to extract **MFCC (Mel-frequency cepstral coefficients)**, which capture important features of the audio. This step converts the raw audio into a set of MFCCs that the model can interpret.

Steps in preprocessing:
1. **Audio Sampling:** Standardize the sampling rate across all audio files.
2. **MFCC Extraction:** Compute MFCCs to capture the essential features of the audio signals.
3. **Normalization:** Normalize the MFCCs to ensure all features are on the same scale.

## Model Architecture
The model consists of a **Bi-directional Long Short-Term Memory (LSTM)** network, which is highly effective in handling sequential data like audio signals. The bi-directional nature allows the model to capture temporal dependencies in both directions, improving the model's accuracy.

### Model Structure:
- **Input Layer:** MFCC features extracted from audio signals.
- **Bi-directional LSTM Layers:** Two stacked bi-directional LSTM layers to model the temporal dependencies in the audio.
- **Fully Connected Layer:** Dense layer for emotion classification.
- **Output Layer:** Softmax layer with six output neurons, each representing an emotion class.

## Training
The model was trained using:
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam optimizer
- **Batch Size:** 32
- **Epochs:** 50
- **Validation Split:** 20%

## Evaluation
The model was evaluated using accuracy, precision, recall, and F1-score on a validation dataset.

## Results
The model achieved the following performance on the validation set:
- Train Accuracy: 0.9902 
- Train Loss0: 0.0327
- Test Accuracy: 0.9866071343421936
- Test Loss: 0.03881167620420456
