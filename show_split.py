"""
TNO Dataset Train/Test Split Listesi
"""
from backend.utils.llvip_dataset_loader import LLVIPDatasetLoader
import os
from pathlib import Path

# Dataset yükle
loader = LLVIPDatasetLoader(os.path.join(os.getcwd(), 'LLVIP'))

# Path'leri al
train_paths = loader.train_pairs
test_paths = loader.test_pairs

print("\n" + "="*60)
print("LLVIP DATASET - TRAIN/TEST SPLIT")
print("="*60)

print(f"\n📚 TRAIN SET ({len(train_paths)} pairs - modeller bunlarla eğitildi):")
print("-" * 60)
for i, (ir_path, vis_path) in enumerate(train_paths, 1):
    file_name = os.path.basename(ir_path)
    print(f"  {i:2d}. {file_name}")

print("\n" + "="*60)
print(f"\n🧪 TEST SET ({len(test_paths)} pairs - performans ölçümü için):")
print("-" * 60)
for i, (ir_path, vis_path) in enumerate(test_paths, 1):
    file_name = os.path.basename(ir_path)
    print(f"  {i:2d}. {file_name}")

print("\n" + "="*60)
print(f"\n📊 Toplam: {len(train_paths) + len(test_paths)} sample")
print("\n💡 Frontend'de TEST görsellerini kullanarak pre-trained")
print("   modellerin performansını görebilirsin!")
print("="*60 + "\n")
