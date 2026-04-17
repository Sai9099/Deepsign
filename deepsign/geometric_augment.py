import json
import os
import random
import math
import numpy as np
from ml_classifier import extract_features, DATA_PATH, MODEL_PATH, LETTERS, train_model

class Landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def rotate_landmarks(landmarks, ax, ay, az):
    """Rotate landmarks around the wrist (0) by angles in radians."""
    wrist = landmarks[0]
    new_lms = []
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    for lm in landmarks:
        # Relative to wrist
        v = np.array([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        v_rot = R @ v
        new_lms.append(Landmark(v_rot[0] + wrist.x, v_rot[1] + wrist.y, v_rot[2] + wrist.z))
    return new_lms

def augment():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found. Calibrate at least one letter first.")
        return

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    if 'raw_landmarks' not in data or not data['raw_landmarks']:
        print("❌ Error: No raw landmarks found in dataset. Please clear data and re-calibrate.")
        return

    raw_samples = data['raw_landmarks']
    labels = data['labels']
    
    new_samples = []
    new_labels = []
    
    TARGET_PER_LETTER = 1000
    print(f"🚀 Boosting {len(set(labels))} letters using geometric augmentation...")

    for letter in LETTERS:
        indices = [i for i, l in enumerate(labels) if l == letter]
        if not indices: continue
        
        print(f"  Boosting '{letter}'...")
        for _ in range(TARGET_PER_LETTER):
            idx = random.choice(indices)
            base_raw = raw_samples[idx]
            # Convert list of dicts to list of Landmark objects
            lms = [Landmark(item['x'], item['y'], item['z']) for item in base_raw]
            
            # Apply transformations
            # 1. 3D Rotation (±20 degrees)
            ax = random.uniform(-0.35, 0.35)
            ay = random.uniform(-0.35, 0.35)
            az = random.uniform(-0.35, 0.35)
            aug_lms = rotate_landmarks(lms, ax, ay, az)
            
            # 2. Scaling (±15%)
            scale = random.uniform(0.85, 1.15)
            wrist = aug_lms[0]
            for lm in aug_lms:
                lm.x = wrist.x + (lm.x - wrist.x) * scale
                lm.y = wrist.y + (lm.y - wrist.y) * scale
                lm.z = wrist.z + (lm.z - wrist.z) * scale
            
            # 3. Micro-jitter
            for lm in aug_lms:
                lm.x += random.uniform(-0.005, 0.005)
                lm.y += random.uniform(-0.005, 0.005)
                lm.z += random.uniform(-0.005, 0.005)
            
            # Extract features from augmented landmarks
            feat = extract_features(aug_lms)
            new_samples.append(feat)
            new_labels.append(letter)

    # Save augmented dataset
    AUG_PATH = DATA_PATH.replace(".json", "_boosted.json")
    with open(AUG_PATH, 'w') as f:
        json.dump({'samples': new_samples, 'labels': new_labels}, f)
    
    print(f"✅ Created augmented dataset with {len(new_samples)} samples.")
    
    # Train
    import ml_classifier
    original_path = ml_classifier.DATA_PATH
    ml_classifier.DATA_PATH = AUG_PATH
    success, msg = train_model()
    ml_classifier.DATA_PATH = original_path
    
    if success:
        print(f"✨ SUCCESS: {msg}")
    else:
        print(f"❌ FAILED: {msg}")

if __name__ == "__main__":
    augment()
