import os
import numpy as np

# A to Z letters
labels = [chr(i) for i in range(65, 91)]

def build_mock_model():
    """Generates a mock Keras model."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(42,)),
            Dense(32, activation='relu'),
            Dense(len(labels), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except ImportError:
        # Simple pure Python mock if tensorflow is completely missing
        class MockModel:
            def predict(self, x, verbose=0):
                batch_size = x.shape[0]
                result = np.zeros((batch_size, len(labels)), dtype=np.float32)
                for i in range(batch_size):
                    result[i, np.random.randint(0, len(labels))] = 1.0
                return result
        return MockModel()

def get_model(model_path="model.h5"):
    """Loads a genuine model if available, otherwise returns a mock."""
    if os.path.exists(model_path):
        try:
            from tensorflow.keras.models import load_model
            return load_model(model_path)
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    
    # Missing or failed to load, return mock
    return build_mock_model()
