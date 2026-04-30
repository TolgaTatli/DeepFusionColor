# Eğitim Rehberi - LLVIP Dataset ile

## 📊 Kısa Özet

Yeni sistem **LLVIP dataset** kullanıyor ve 3 aşamada eğitim yapılıyor:

### 1️⃣ ÖRNEK EĞİTİM (Şu an çalışıyor - ~5-10 dakika)
```bash
python backend\train_models.py --model all --max-samples 1000 \
    --epochs-dnn 2 --epochs-cnn 3 --epochs-densefuse 3
```
**Ne işe yarar:**
- ✅ Sistemin doğru çalıştığını test et
- ✅ Model mimarisini doğrula
- ✅ Hızlı sonuç

**Model kalitesi:** Düşük (az eğitim veri)

---

### 2️⃣ PRODUCTION EĞİTİM (Daha sonra - Full veri ile)
```bash
python train_full_production.py
```
**Ne yapar:**
- ✅ Tüm 12,025 training sample ile eğitim
- ✅ Production-ready modeller
- ✅ Yüksek kalite

**Süre:** 
- GPU ile: ~30-45 dakika
- CPU ile: ~4-6 saat

**Model kalitesi:** Çok iyi ✨

---

## 🎯 Adımlar

### Adım 1: Kısa Test Çalıştır (İLK İyiyse)
```powershell
cd c:\Users\korkm\OneDrive\Masaüstü\DeepFusionColor
& .\venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = 'utf-8'

# 1000 sample ile test
python backend\train_models.py --model all --max-samples 1000
```

**Beklenen sonuç:**
```
[LLVIP Dataset]
  Train pairs: 1000
  Test pairs: 286

================================================
DNN MODEL TRAINING
================================================
[1/4] Loading training data...
...
✓ 50 pair loaded
Shape: (50, 256, 256)

[4/4] Saving model...
✓ DNN model saved to: backend/trained_models/dnn_fusion_model.pth
...
✓ TRAINING COMPLETE!
```

---

### Adım 2: Modelleri Test Et
```bash
# Frontend'i başlat
python backend\main.py

# Veya
npm start  (frontend klasöründen)
```

Frontend'den upload et ve test et.

---

### Adım 3: Full Production Eğitim (İsteğe Bağlı)

#### Varsa GPU kullan:
```bash
# GPU PyTorch kurmak istersen (RTX/RTX 30XX/RTX 20XX)
py -3.11 -m pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

#### Full eğitim çalıştır:
```bash
python train_full_production.py
```

---

## 📋 Eğitim Parametreleri

| Seçenek | Kısa Test | Production |
|---------|-----------|-----------|
| Samples | 1,000 | 12,025 (full) |
| DNN epochs | 2 | 20 |
| CNN epochs | 3 | 40 |
| DenseFuse epochs | 3 | 30 |
| Batch size | 16 | 16 |
| Süre (CPU) | ~5 min | 4-6 hours |
| Süre (GPU) | ~30s | 30-45 min |
| Model kalitesi | %60 | %95+ |

---

## 🔍 Status Kontrol

### Process'i izle:
```powershell
Get-Process python | Where-Object {$_.CPU -gt 0} | Format-Table Name, Id, CPU, Memory
```

### Kaydedilen modeller:
```powershell
Get-ChildItem backend\trained_models\ -File | Select-Object Name, @{Name='Size(MB)';Expression={[math]::Round($_.Length/1MB, 2)}}
```

---

## 🚨 Sorunlar

### Model yüklenmiyor
```
ModuleNotFoundError: No module named 'torch'
```
**Çözüm:** Adım 3'te PyTorch kurmayı atladın
```bash
pip install torch
```

### Unicode hatası
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
```
**Çözüm:** 
```powershell
$env:PYTHONIOENCODING = 'utf-8'
```

### Eğitim çok yavaş
- GPU kurru (SETUP.md'de komutu var)
- Batch size'ı küçült: `--batch-size 8`
- Epoch sayısını azalt

---

## ✅ Checklist

- [ ] Kısa test eğitim başarılı mı?
- [ ] Modeller `backend/trained_models/` klasöründe kaydedildi mi?
  - [ ] dnn_fusion_model.pth
  - [ ] cnn_fusion_model.pth
  - [ ] densefuse_model.pth
- [ ] Frontend `main.py` ile yüklenir mi?
- [ ] Test görüntüsü upload edip füzyon işlemi yapılıyor mu?
- [ ] Sonuç gösteriliyor mu?

İçi tamamlandıysa: **3 aşında Full eğitime geç!**

---

## 💡 Tips

**Eğitim sırasında bilgisayarı kapat/uyut:**
- ❌ Eğitim durur, model kaybolur
- Devam ettirmek zorundasın

**Daha hızlı sonuç istiyorsan:**
- GPU kurra (10x hızlı)
- Epoch sayısını azalt (kalite düşer)
- Batch size'ı düşür (bellek az ama yavaş)

**Production deployment:**
- Full eğitim tamamla
- Modelleri yedekle (backup)
- `backend/trained_models/` klasörü git'e commit et
