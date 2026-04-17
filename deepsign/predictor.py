import cv2
import time
import numpy as np

from gesture_classifier import classify_gesture
import ml_classifier

class SignLanguagePredictor:
    def __init__(self, model=None):
        self.model = model  # kept for future use with a trained model
        self.use_ml = ml_classifier.has_trained_model()
        
        if self.use_ml:
            ml_classifier.load_model()
            print("ML model loaded - using trained classifier")
        else:
            print("No trained model found - using rule-based classifier")
            print("Use the 'Calibrate' button in the app to train a personalized model")

        # Initialize MediaPipe HandLandmarker Tasks API
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            running_mode=vision.RunningMode.IMAGE
        )
        try:
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self.has_mp = True
            self.mp = mp
        except Exception as e:
            print(f"Error initializing MediaPipe Task HandLandmarker: {e}")
            self.landmarker = None
            self.has_mp = False

        # Define connections for drawing skeleton
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def _draw_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Define modern color palette (BGR)
        PALETTE = {
            'wrist': (246, 92, 139),  # glow_purple
            'thumb': (246, 130, 59),  # glow_blue
            'index': (131, 52, 83),   # accent_bright
            'middle': (246, 92, 139), 
            'ring': (246, 130, 59),
            'pinky': (131, 52, 83),
            'joint_core': (255, 255, 255), # White core
            'joint_glow': (246, 92, 139)   # Glow color
        }

        # Subsets of connections for different colors
        segments = {
            'thumb': self.connections[0:4],
            'index': self.connections[4:8],
            'middle': self.connections[8:12],
            'ring': self.connections[12:16],
            'pinky': self.connections[16:20],
            'base': [self.connections[20], (0, 5), (5, 9), (9, 13), (13, 17)]
        }

        # Draw skeleton lines
        for seg_name, conn_list in segments.items():
            color = PALETTE.get(seg_name, PALETTE['wrist'])
            for p1, p2 in conn_list:
                cv2.line(frame, points[p1], points[p2], color, 3, cv2.LINE_AA)
        
        # Draw joints with glowing effect
        for point in points:
            # Outer glow
            cv2.circle(frame, point, 6, PALETTE['joint_glow'], -1, cv2.LINE_AA)
            # Inner core
            cv2.circle(frame, point, 3, PALETTE['joint_core'], -1, cv2.LINE_AA)

    def detect_landmarks(self, frame):
        """Detect hand landmarks without classification. Returns landmarks or None."""
        if not self.has_mp:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect(mp_image)
        
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            return result.hand_landmarks[0]
        return None

    def process_frame(self, frame):
        predicted_letter = None
        confidence = 0.0
        debug_info = {'ext': 0, 'orient': 'none'}

        if not self.has_mp:
            return frame, predicted_letter, confidence, debug_info

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Draw skeleton
                self._draw_landmarks(frame, hand_landmarks)

                # Use ML model if available, otherwise fall back to rules
                if self.use_ml:
                    predicted_letter, confidence = ml_classifier.predict(hand_landmarks)
                    # For ML, we still want to show finger counts if possible
                    _, _, debug_info = classify_gesture(hand_landmarks)
                else:
                    predicted_letter, confidence, debug_info = classify_gesture(hand_landmarks)

                # Process only one hand
                break

        return frame, predicted_letter, confidence, debug_info

    def reload_ml_model(self):
        """Reload ML model after training."""
        if ml_classifier.has_trained_model():
            ml_classifier.load_model()
            self.use_ml = True
            print("ML model reloaded")
        
    def release(self):
        if self.has_mp and self.landmarker:
            self.landmarker.close()
