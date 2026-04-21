🧠 Real-Time Sign Language to Text Converter

A real-time application that translates American Sign Language (ASL) hand gestures into text using computer vision and deep learning. The system captures hand movements through a webcam, detects key landmarks, and predicts corresponding alphabets (A–Z) using a trained neural network.

🚀 Features
🎥 Real-time Hand Tracking
Uses MediaPipe to detect and track hand landmarks with high accuracy.
🤖 Deep Learning Prediction
Classifies hand gestures into ASL alphabets (A–Z) using a TensorFlow/Keras model.
🖥️ Interactive GUI
Built with Tkinter for a simple and user-friendly interface.
🔊 Text-to-Speech (TTS)
Converts recognized text into speech for better accessibility.
⚡ Plug-and-Play Execution
Automatically generates a mock model if no trained model is found — making the project runnable instantly.
🛠️ Tech Stack
Language: Python
Libraries & Frameworks:
OpenCV (Computer Vision)
MediaPipe (Hand Landmark Detection)
TensorFlow / Keras (Model Prediction)
Tkinter (GUI)
pyttsx3 / gTTS (Text-to-Speech)
