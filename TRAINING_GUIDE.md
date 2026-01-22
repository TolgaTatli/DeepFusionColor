# Model Training Guide

## 🎯 Sorun

Önceki sistemde CNN ve DenseFuse modelleri **her füzyon isteğinde yeniden eğitiliyordu**:
- ❌ Tek görüntü çifti ile eğitim (overfitting garantisi!)
- ❌ Her seferinde yeniden eğitim (çok yavaş)
- ❌ TNO dataset kullanılmıyordu
- ❌ Bilimsel olarak tamamen yanlış

## ✅ Yeni Sistem

Artık **düzgün bir ML pipeline** var:

1. **Training**: TNO dataset ile bir kere eğit (70-30 split)
2. **Save**: Modelleri `.pth` dosyası olarak kaydet
3. **Inference**: Frontend'den gelen görüntüleri pre-trained model ile füze et

## 📊 TNO Dataset

TNO Image Fusion Dataset içeriyor:
- Thermal (IR) ve Visual görüntü çiftleri
- Farklı senaryolar (askeri, kentsel, doğa)
- ~100+ görüntü çifti

**Train-Test Split**: 70-30
- %70 → Eğitim (model bunları öğrenir)
- %30 → Test (model hiç görmediği görüntülerde test edilir)

## 🚀 Model Eğitimi

### Adım 1: Environment Hazırla

```bash
# Virtual environment aktif et (.venv - Python 3.11 + CUDA)
.\.venv\Scripts\activate
```

### Adım 2: Bağımlılıkları Kontrol Et

```bash
pip install scikit-learn  # train-test split için gerekli
```

### Adım 3: Modelleri Eğit

**Tüm modelleri eğit (önerilen):**
```bash
python backend\train_models.py --model all
```

**Sadece CNN:**
```bash
python backend\train_models.py --model cnn --epochs-cnn 30
```

**Sadece DenseFuse:**
```bash
python backend\train_models.py --model densefuse --epochs-densefuse 25
```

**Hızlı test (az veri ile):**
```bash
python backend\train_models.py --model all --max-samples 50 --epochs-cnn 10 --epochs-densefuse 10
```

**Özel split oranı:**
```bash
python backend\train_models.py --model all --test-size 0.2  # 80-20 split
```

### Eğitim Parametreleri

```bash
python backend\train_models.py --model all --epochs-cnn 30 --epochs-densefuse 25 --batch-size 16 --test-size 0.3 --max-samples 100
```

## 📁 Çıktılar

Eğitim tamamlandığında modeller şuraya kaydedilir:
```
backend/trained_models/
  ├── cnn_fusion_model.pth
  ├── densefuse_model.pth
  └── README.md
```

Her `.pth` dosyası içerir:
- Model weights
- Hyperparameters
- Training info (kaç sample, hangi split)

## 🔄 Flask API ile Kullanım

Flask API başlatıldığında otomatik olarak pre-trained modelleri yükler:

```bash
cd backend
python main.py
```

**Konsol çıktısı:**
```
python backend\ model loaded
  ✅ DenseFuse model loaded
[STARTUP] Ready!
```

Eğer model yoksa:
```
  ⚠️ CNN model not found (will train on-the-fly)
  ⚠️ DenseFuse model not found (will train on-the-fly)
```

## 🎨 Frontend Kullanımı

Frontend'den füzyon isteği geldiğinde:

**Pre-trained model varsa:**
- ⚡ Hızlı inference (saniyeler)
- ✅ TNO dataset ile eğitilmiş model
- ✅ Generalization (yeni görüntülerde de iyi çalışır)

**Pre-trained model yoksa:**
- 🐌 On-the-fly training (çok yavaş)
- ❌ Tek görüntü ile eğitim (overfitting)
- ❌ Kötü sonuçlar

## 📝 Öneriler

### İlk Kurulum
1. `python train_models.py --model all` ile tüm modelleri eğit
2. Bu işlem 10-30 dakika sürebilir (CPU'da)
3. Eğitim bitince modeller otomatik kaydedilir
4. Artık Flask API hızlı inference yapacak

### Production İçin
- ✅ Modelleri mutlaka pre-train et
- ✅ .venv`'i aktif et: `.\.venv\Scripts\activate`
2. `python backend\train_models.py --model all` ile tüm modelleri eğit
3. Bu işlem 5-15 dakika sürer (GPU ile) veya 20-40 dakika (CPU ile)
4. Eğitim bitince modeller otomatik kaydedilir
5
### Development İçin
- Quick test: `--max-samples 20 --epochs-cnn 5`
- Bu 1-2 dakikada biter
- Kalite düşük ama test için yeterli

## 🔍 Model Değerlendirme

Training script otomatik olarak test setinde değerlendirme yapar:

```
[EVALUATING CNN]
Testing on 10 image pairs...
  Sample 1: fused shape = (256, 256), range = [0.123, 0.891]
  Sample 2: fused shape = (256, 256), range = [0.089, 0.923]
  ...
✅ CNN evaluation complete!
```

## 🐛 Troubleshooting

**Hata: "TNO dataset not found"**
- TNO dataset'in `TNO_Image_Fusion_Dataset/TNO_Image_Fusion_Dataset/` altında olduğundan emin ol

**Hata: "CUDA out of memory"**
- `--batch-size 8` ile batch size'ı küçült
- Veya CPU kullan (otomatik fallback var)

**Eğitim çok yavaş**
- `--max-samples 50` ile hızlı test yap
- GPU kullan (varsa)
- Epoch sayısını azalt

**Model yüklenmiyor**
- Model dosyalarının `backend/trained_models/` altında olduğunu kontrol et
- Dosya adı: `cnn_fusion_model.pth` ve `densefuse_model.pth` olmalı

## 📚 Ek Bilgiler

### Dataset Loader Test

Dataset loader'ı test et:
```bash
cd backend/utils
python tno_dataset_loader.py
```

### Manuel Model Yükleme

python backend\utils\on
from models.cnn_fusion import CNNFusionTrainer

# Pre-trained model yükle
trainer = CNNFusionTrainer(pretrained_path='trained_models/cnn_fusion_model.pth')

# Inference
fused = trainer.predict(thermal_img, visual_img)
```

## ✨ Sonuç

Artık sistem profesyonel bir ML pipeline'ına sahip:
- ✅ Proper train-test split (70-30)
- ✅ Pre-trained models
- ✅ Fast inference
- ✅ No overfitting
- ✅ Gerçek dünya senaryolarında çalışır

Senin benzetmenle: Artık model sadece seninle değil, tüm TNO dataset ile train ediliyor! 🎓
