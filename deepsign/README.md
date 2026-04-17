# Real-Time Sign Language to Text Converter

This application uses a webcam, OpenCV, MediaPipe for hand landmark extraction, and TensorFlow/Keras to predict American Sign Language (ASL) alphabets in real-time.

## Features
- Real-time hand landmark detection using MediaPipe.
- Gestures mapped to A-Z using a TensorFlow Keras model.
- GUI built with Tkinter.
- Text-to-speech functionality to speak formed sentences.
- Auto-generates a mock Keras model if one isn't found, handling out-of-the-box execution perfectly.

## Setup Instructions

1. **Install Dependencies**
   It is highly recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   Start the application by running the main entry point:
   ```bash
   python main.py
   ```
   
   *Note: If a pre-trained model is not found, the app automatically generates a "mock" model. The mock model is structure-complete but will produce random predictions. You can replace it with a fully trained model for accurate ASL predictions.*
