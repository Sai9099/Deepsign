import traceback
try:
    import ml_classifier
    import numpy as np
    print("ML Classifier imported successfully.")
    print(f"Model loaded: {ml_classifier.has_trained_model()}")
    if ml_classifier.has_trained_model():
        ml_classifier.load_model()
        print(f"Model classes: {ml_classifier._labels}")
except Exception:
    print(traceback.format_exc())
