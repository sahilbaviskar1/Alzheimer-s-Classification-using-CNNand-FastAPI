# Alzheimer's Classification using CNN and FastAPI

This project utilizes Convolutional Neural Networks (CNN) for the classification of Alzheimer's disease based on brain scan images (such as MRI scans). FastAPI is used to deploy the trained model as an API for real-time predictions.

## Overview

The goal of this project is to develop a machine learning model that can classify brain scans into two categories: Alzheimer's or non-Alzheimer's. A CNN model is trained on a dataset of brain scan images, and FastAPI is used to create a simple API to serve predictions from the trained model.

### Key Features:
- **CNN Model:** The model uses Convolutional Neural Networks to automatically learn features from brain scans and classify them as either Alzheimer's or non-Alzheimer's.
- **FastAPI Deployment:** FastAPI serves as the framework for exposing the trained CNN model through a REST API, allowing users to interact with it in real-time.
- **TensorFlow/Keras:** The model is built and trained using TensorFlow and Keras for deep learning.

## Installation

### Prerequisites
Make sure you have Python 3.x installed on your system. You also need `pip` for installing dependencies.

### 1. Create a Virtual Environment

It is highly recommended to use a virtual environment for this project to manage dependencies.

#### For Windows:
Open a terminal (Command Prompt or PowerShell) and run the following commands:

```bash
# Navigate to the project directory
cd path\to\your\project\folder

# Create a virtual environment (you can name it 'venv' or any other name)
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

## Install Dependencies
      pip install tensorflow fastapi uvicorn

## Running the FastApi Server
      uvicorn main:app --reload

## Training the model
      python train_model.py

## Prediction of an image
      python predict_image.py
```
#### For macOS/Linux:
```bash
# Navigate to the project directory
cd path/to/your/project/folder

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

## Running the FastApi Server
      uvicorn main:app --reload

## Training the model
      python train_model.py

## Prediction of an image
      python predict_image.py
```
The API will be available locally at http://127.0.0.1:8000. You can interact with it by sending requests.

### Project Structure
```plaintext
Alzheimers-Classification-using-CNN-and-FastAPI/
│
├── main.py                   # FastAPI server file to deploy the model
├── train_model.py            # Script to train the CNN model
├── model/                    # Directory where the trained model is saved
│   └── model.h5              # Trained model file
├── requirements.txt          # Python dependencies for the project
├── .gitignore                # Git ignore file to exclude unnecessary files
└── README.md                 # Project documentation


