#!/usr/bin/env python3
"""
CFP Dataset Verification for EMA Validation
==========================================

This script verifies that the CFP dataset is correctly structured
and suitable for the EMA validation framework.
"""

import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace

def verify_dataset_structure(cfp_path):
    """Verify the structure of the CFP dataset"""
    print("\n📁 Verifying CFP Dataset Structure...")
    
    # Check if the dataset exists
    cfp_path = Path(cfp_path)
    if not cfp_path.exists():
        print("❌ CFP dataset path does not exist!")
        return False
    
    # Check for required directories
    data_path = cfp_path / "Data"
    protocol_path = cfp_path / "Protocol"
    
    if not data_path.exists():
        print("❌ Missing Data directory!")
        return False
    
    if not protocol_path.exists():
        print("❌ Missing Protocol directory!")
        return False
    
    # Check for Images directory
    images_path = data_path / "Images"
    if not images_path.exists():
        print("❌ Missing Images directory!")
        return False
    
    # Count subject directories
    subject_dirs = [d for d in images_path.iterdir() if d.is_dir()]
    num_subjects = len(subject_dirs)
    
    if num_subjects == 0:
        print("❌ No subject directories found!")
        return False
    
    print(f"✅ Found {num_subjects} subject directories")
    
    # Check a random sample of subject directories
    sample_size = min(10, num_subjects)
    sample_dirs = random.sample(subject_dirs, sample_size)
    
    valid_structure = True
    frontal_count = 0
    profile_count = 0
    
    for subject_dir in sample_dirs:
        frontal_dir = subject_dir / "frontal"
        profile_dir = subject_dir / "profile"
        
        if not frontal_dir.exists():
            print(f"❌ Missing frontal directory in {subject_dir.name}")
            valid_structure = False
        else:
            frontal_images = list(frontal_dir.glob("*.jpg"))
            frontal_count += len(frontal_images)
        
        if not profile_dir.exists():
            print(f"❌ Missing profile directory in {subject_dir.name}")
            valid_structure = False
        else:
            profile_images = list(profile_dir.glob("*.jpg"))
            profile_count += len(profile_images)
    
    avg_frontal = frontal_count / sample_size
    avg_profile = profile_count / sample_size
    
    print(f"✅ Average frontal images per subject: {avg_frontal:.1f}")
    print(f"✅ Average profile images per subject: {avg_profile:.1f}")
    
    return valid_structure

def test_embedding_extraction(cfp_path):
    """Test embedding extraction on a few images"""
    print("\n🔍 Testing Face Embedding Extraction...")
    
    cfp_path = Path(cfp_path)
    images_path = cfp_path / "Data" / "Images"
    
    if not images_path.exists():
        print("❌ Images directory not found!")
        return False
    
    # Find a subject with both frontal and profile images
    for subject_dir in images_path.iterdir():
        if not subject_dir.is_dir():
            continue
            
        frontal_dir = subject_dir / "frontal"
        profile_dir = subject_dir / "profile"
        
        if not frontal_dir.exists() or not profile_dir.exists():
            continue
            
        frontal_images = list(frontal_dir.glob("*.jpg"))
        profile_images = list(profile_dir.glob("*.jpg"))
        
        if len(frontal_images) > 0 and len(profile_images) > 0:
            test_images = [frontal_images[0], profile_images[0]]
            break
    else:
        print("❌ Could not find subject with both frontal and profile images!")
        return False
    
    # Test embedding extraction
    try:
        print(f"   Testing on subject {subject_dir.name}...")
        embeddings = []
        
        for img_path in test_images:
            print(f"   Processing {img_path.name}...")
            embedding = DeepFace.represent(
                img_path=str(img_path),
                model_name="Facenet512",
                enforce_detection=False
            )
            
            if isinstance(embedding, list):
                embedding = embedding[0]["embedding"]
                
            embeddings.append(embedding)
            print(f"   ✅ Successfully extracted embedding: {len(embedding)} dimensions")
        
        # Calculate similarity
        embedding1 = np.array(embeddings[0])
        embedding2 = np.array(embeddings[1])
        
        similarity = np.dot(
            embedding1 / np.linalg.norm(embedding1),
            embedding2 / np.linalg.norm(embedding2)
        )
        
        print(f"   ✅ Frontal-Profile similarity: {similarity:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error extracting embeddings: {e}")
        return False

def verify_for_ema_validation(cfp_path):
    """Verify that the dataset is suitable for EMA validation"""
    print("\n🧪 Verifying Dataset for EMA Validation...")
    
    cfp_path = Path(cfp_path)
    images_path = cfp_path / "Data" / "Images"
    
    if not images_path.exists():
        print("❌ Images directory not found!")
        return False
    
    # Check for enough subjects with sufficient images
    valid_subjects = 0
    total_frontal = 0
    total_profile = 0
    
    for subject_dir in images_path.iterdir():
        if not subject_dir.is_dir():
            continue
            
        frontal_dir = subject_dir / "frontal"
        profile_dir = subject_dir / "profile"
        
        if not frontal_dir.exists() or not profile_dir.exists():
            continue
            
        frontal_images = list(frontal_dir.glob("*.jpg"))
        profile_images = list(profile_dir.glob("*.jpg"))
        
        if len(frontal_images) >= 5:  # Need at least 5 frontal images for validation
            valid_subjects += 1
            total_frontal += len(frontal_images)
            total_profile += len(profile_images)
    
    print(f"✅ Found {valid_subjects} subjects with sufficient images")
    
    if valid_subjects >= 8:
        print(f"✅ Dataset is SUITABLE for EMA validation")
        print(f"   Total frontal images: {total_frontal}")
        print(f"   Total profile images: {total_profile}")
        return True
    else:
        print(f"❌ Not enough subjects with sufficient images for validation")
        print(f"   Need at least 8 subjects, found {valid_subjects}")
        return False

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🔍 CFP DATASET VERIFICATION FOR EMA VALIDATION")
    print("=" * 60)
    
    cfp_path = "cfp-dataset"
    
    # Verify dataset structure
    structure_valid = verify_dataset_structure(cfp_path)
    
    # Test embedding extraction
    extraction_valid = test_embedding_extraction(cfp_path)
    
    # Verify for EMA validation
    validation_valid = verify_for_ema_validation(cfp_path)
    
    # Final verdict
    print("\n" + "=" * 60)
    if structure_valid and extraction_valid and validation_valid:
        print("✅ VERIFICATION PASSED: Dataset is ready for EMA validation")
    else:
        print("❌ VERIFICATION FAILED: Dataset needs attention")
    print("=" * 60)

if __name__ == "__main__":
    main() 