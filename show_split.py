"""
TNO Dataset Train/Test Split Listesi
"""
from backend.utils.tno_dataset_loader import TNODatasetLoader
import os

# Dataset yükle
loader = TNODatasetLoader(os.path.join(os.getcwd(), 'TNO_Image_Fusion_Dataset', 'TNO_Image_Fusion_Dataset'))

# Path'leri al (numpy array değil)
train_paths = loader.train_pairs
test_paths = loader.test_pairs

print("\n" + "="*60)
print("TNO DATASET - TRAIN/TEST SPLIT (70-30)")
print("="*60)

print("\n📚 TRAIN SET (42 pairs - modeller bunlarla eğitildi):")
print("-" * 60)
for i, (ir_path, vis_path) in enumerate(train_paths, 1):
    folder_name = os.path.basename(os.path.dirname(ir_path))
    print(f"  {i:2d}. {folder_name}")

print("\n" + "="*60)
print("\n🧪 TEST SET (18 pairs - performans ölçümü için):")
print("-" * 60)
for i, (ir_path, vis_path) in enumerate(test_paths, 1):
    folder_name = os.path.basename(os.path.dirname(ir_path))
    print(f"  {i:2d}. {folder_name}")

print("\n" + "="*60)
print("\n💡 Frontend'de TEST görsellerini kullanarak pre-trained")
print("   modellerin performansını görebilirsin!")
print("="*60 + "\n")
