import ml_classifier
import numpy as np

class MockLM:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

mock_lms = [MockLM() for _ in range(21)]
features = ml_classifier.extract_features(mock_lms)
print(f"Feature vector length: {len(features)}")
if len(features) == 87:
    print("✅ Feature length is CORRECT (87).")
else:
    print(f"❌ Feature length is INCORRECT ({len(features)}).")
