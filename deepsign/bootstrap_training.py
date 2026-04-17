import json
import os
import random
import math
import numpy as np
from ml_classifier import extract_features, DATA_PATH, MODEL_PATH, LETTERS, train_model
import joblib

class SyntheticLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def augment_landmarks(landmarks, noise_scale=0.012, rotation_range=15, scale_range=0.1):
    """
    Apply random noise, rotation, and scaling to a set of features.
    landmarks is a list of 87 features (from extract_features).
    However, it's easier to augment the RAW landmarks then re-extract.
    But we only have the EXTRACTED features in the JSON.
    Let's check if we have the raw landmarks.
    Actually, the JSON has the 87 features.
    """
    features = list(landmarks)
    
    # Simple jitter on the 87 features
    for i in range(len(features)):
        jitter = random.normalvariate(0, noise_scale)
        features[i] += jitter
        
    return features

def bootstrap():
    print("🚀 Starting Bootstrap Training...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    original_samples = data['samples']
    original_labels = data['labels']
    
    num_original = len(original_samples)
    print(f"📊 Loaded {num_original} original samples.")
    
    boosted_samples = []
    boosted_labels = []
    
    # Goal: ~2000 samples per letter
    TARGET_PER_LETTER = 1500
    
    from collections import Counter
    counts = Counter(original_labels)
    
    for letter in LETTERS:
        letter_samples = [s for s, l in zip(original_samples, original_labels) if l == letter]
        
        if not letter_samples:
            print(f"⚠️ Warning: No samples for '{letter}'. Skipping bootstrap for this letter.")
            continue
            
        print(f"🏗️ Boosting '{letter}' ({len(letter_samples)} original)...")
        
        # 1. Keep originals
        boosted_samples.extend(letter_samples)
        boosted_labels.extend([letter] * len(letter_samples))
        
        # 2. Generate synthetic variations
        shortfall = TARGET_PER_LETTER - len(letter_samples)
        for _ in range(shortfall):
            base = random.choice(letter_samples)
            # Higher noise for bootstrap to explore decision space
            boosted_samples.append(augment_landmarks(base, noise_scale=0.015))
            boosted_labels.append(letter)
            
    print(f"✅ Created {len(boosted_samples)} total samples.")
    
    # Save to a temporary 'boosted' file for training
    BOOSTED_PATH = DATA_PATH.replace(".json", "_boosted.json")
    with open(BOOSTED_PATH, 'w') as f:
        json.dump({'samples': boosted_samples, 'labels': boosted_labels}, f)
    
    print(f"💾 Saved boosted dataset to {BOOSTED_PATH}")
    
    # Backup original model
    if os.path.exists(MODEL_PATH):
        backup = MODEL_PATH + ".bak"
        if os.path.exists(backup): os.remove(backup)
        os.rename(MODEL_PATH, backup)
        print(f"📦 Backed up old model to {backup}")

    # Temporarily point ml_classifier to the boosted data
    import ml_classifier
    old_data_path = ml_classifier.DATA_PATH
    ml_classifier.DATA_PATH = BOOSTED_PATH
    
    print("🧠 Training RandomForest model on boosted data...")
    success, message = train_model()
    
    # Restore path
    ml_classifier.DATA_PATH = old_data_path
    
    if success:
        print(f"✨ SUCCESS: {message}")
    else:
        print(f"❌ FAILED: {message}")

if __name__ == "__main__":
    bootstrap()
