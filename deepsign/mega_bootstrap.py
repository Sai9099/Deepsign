import json
import os
import random
import math
import numpy as np
from ml_classifier import extract_features, DATA_PATH, MODEL_PATH, LETTERS, train_model
from gesture_classifier import classify_gesture

class Landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def generate_random_hand():
    """Generate a random but physically plausible hand pose."""
    # Start with wrist at origin
    lms = [Landmark(0.5, 0.5, 0)]
    
    # Simple finger generation: each joint is a random extension from the previous
    # This is very rough but good enough for the 'rule discriminator' to filter
    for _ in range(20):
        lms.append(Landmark(random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(-0.2, 0.2)))
        
    return lms

def generate_constrained_hand(letter):
    """Generate a hand pose specifically for a letter (A-Z)."""
    # Base wrist
    lms = [Landmark(0.5, 0.8, 0)]
    
    # We use our knowledge of signs to place tips roughly
    # This is faster than pure random + filter
    # A: Fist + thumb out side
    # B: All up
    # C: Claw
    # ... etc
    # For now, let's use the 'Filter' approach with high-speed random generation
    return generate_random_hand()

def mega_bootstrap():
    print(f"🌟 Starting Mega Bootstrap for 95%+ Accuracy...")
    
    samples = []
    labels = []
    
    TARGET_PER_LETTER = 5000
    counts = {l: 0 for l in LETTERS}
    
    print(f"🏗️ Generating {TARGET_PER_LETTER * 26:,} synthetic samples...")
    
    attempts = 0
    while any(c < TARGET_PER_LETTER for c in counts.values()):
        attempts += 1
        # 1. Generate many random hand poses
        batch_lms = []
        for _ in range(500): # Batch size
            lms = []
            # Start at wrist
            lms.append(Landmark(0.5, 0.7, 0))
            # Generate fingers with some physical constraints
            for f_idx in range(5):
                prev_x, prev_y, prev_z = 0.5, 0.7, 0
                for j_idx in range(4):
                    # Point generally upwards/outwards
                    angle = random.uniform(-math.pi, math.pi)
                    dist = random.uniform(0.05, 0.15)
                    new_x = prev_x + math.cos(angle)*dist
                    new_y = prev_y + math.sin(angle)*dist
                    new_z = prev_z + random.uniform(-0.05, 0.05)
                    lms.append(Landmark(new_x, new_y, new_z))
                    prev_x, prev_y, prev_z = new_x, new_y, new_z
            
            # 2. Discriminate using our high-end rules
            res_letter, conf, debug = classify_gesture(lms)
            if res_letter and counts[res_letter] < TARGET_PER_LETTER:
                # 3. Extract the new 157 features
                feat = extract_features(lms)
                samples.append(feat)
                labels.append(res_letter)
                counts[res_letter] += 1

        if attempts % 10 == 0:
            progress = sum(counts.values()) / (26 * TARGET_PER_LETTER)
            print(f"  Progress: {progress:.1%} ({sum(counts.values())} samples)...")
            
        if attempts > 2000: # Safety break
            break

    print(f"✅ Generation complete. Total samples: {len(samples)}")
    
    # Save to JSON
    MEGA_PATH = DATA_PATH.replace(".json", "_mega.json")
    with open(MEGA_PATH, 'w') as f:
        json.dump({'samples': samples, 'labels': labels}, f)
        
    print(f"💾 Saved mega dataset to {MEGA_PATH}")
    
    # Point classifier to new data and train
    import ml_classifier
    old_data = ml_classifier.DATA_PATH
    ml_classifier.DATA_PATH = MEGA_PATH
    
    print("🧠 Training Ultra-High Accuracy Model (500 estimators)...")
    import sklearn.ensemble
    import joblib
    
    # Customize training inside this script for maximum power
    clf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    X = np.array(samples)
    y = np.array(labels)
    
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    print(f"✨ Model deployed to {MODEL_PATH}")
    
    # Cleanup
    ml_classifier.DATA_PATH = old_data

if __name__ == "__main__":
    mega_bootstrap()
