# DeepFusionColor Kurulum Rehberi
## 🚀 NVIDIA GPU ile Hızlı Kurulum

Bu rehber, projeyi **NVIDIA GPU ile çalıştırmak** için adım adım kurulum talimatlarını içerir.

---

## ✅ Gereksinimler

- **Python 3.11** (ÖNEMLİ: 3.11 kullanın, 3.12+ ile PyTorch CUDA uyumsuzluk yapabilir)
- **NVIDIA GPU** (GTX 1660 Super, RTX 4080, vb.)
- **CUDA 11.8** veya 12.1 uyumlu sürücüler
- **Git**

---

## 📥 1. Repo'yu Klonlayın

```bash
git clone https://github.com/TolgaTatli/DeepFusionColor.git
cd DeepFusionColor
```

---

## 🐍 2. Python Versiyonunu Kontrol Edin

```bash
# Python 3.11 yüklü mü?
python --version
# veya
py -3.11 --version
```

**Python 3.11 yoksa indirin:** https://www.python.org/downloads/

---

## ⚡ 3. PyTorch CUDA Kurulumu (EN ÖNEMLİ ADIM!)

### RTX 4080 / RTX 30XX / RTX 20XX için (CUDA 11.8):

```bash
py -3.11 -m pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Alternatif: En güncel CUDA 12.1 ile:

```bash
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Test edin:**

```bash
py -3.11 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

✅ **Çıktı şöyle olmalı:**
```
CUDA: True
GPU: NVIDIA GeForce RTX 4080
```

---

## 📦 4. Diğer Paketleri Kurun

```bash
cd backend
py -3.11 -m pip install -r requirements.txt
```

---

## 🎯 5. Backend'i Başlatın

```bash
py -3.11 main.py
```

✅ **Başarılı ise şunu göreceksiniz:**
```
[DNN] 🚀 GPU aktif: NVIDIA GeForce RTX 4080
[CNN] 🚀 GPU aktif: NVIDIA GeForce RTX 4080
[DenseFuse] 🚀 GPU aktif: NVIDIA GeForce RTX 4080
 * Running on http://localhost:5000
```

---

## 🌐 6. Frontend'i Başlatın

**Yeni terminal açın:**

```bash
cd frontend
py -3.11 -m http.server 8000
```

---

## 🎉 7. Tarayıcıda Açın

http://localhost:8000

---

## 🔧 Sorun Giderme

### ❌ "CUDA: False" hatası

**Çözüm 1:** NVIDIA sürücüleri güncel mi?
```bash
nvidia-smi
```

**Çözüm 2:** PyTorch CPU versiyonu kurulu olabilir, kaldırıp CUDA versiyonunu kurun:
```bash
pip uninstall torch torchvision -y
py -3.11 -m pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### ❌ "ModuleNotFoundError: No module named 'numpy'"

```bash
py -3.11 -m pip install "numpy<2.0"
```

### ❌ "Failed to initialize NumPy: _ARRAY_API not found"

NumPy 2.x uyumsuzluğu, 1.x kurun:
```bash
pip uninstall numpy -y
py -3.11 -m pip install "numpy<2.0"
```

### ❌ Çok yavaş çalışıyor (GPU kullanmıyor)

Backend başlatırken konsolu kontrol edin:
- ✅ Görüyorsanız: `🚀 GPU aktif: NVIDIA GeForce RTX 4080`
- ❌ Görüyorsanız: `CPU kullanılıyor (GPU bulunamadı)`

CPU kullanıyorsa PyTorch CUDA kurulumunu tekrar yapın.

---

## 📊 Performans Beklentileri

| GPU | DNN | CNN | DenseFuse |
|-----|-----|-----|-----------|
| **CPU only** | ~30s | ~2min | ~5-10min |
| **GTX 1660 Super** | ~3s | ~10s | ~30s |
| **RTX 4080** | ~1s | ~5s | ~15s |

---

## 🎓 Dataset

TNO Image Fusion Dataset zaten proje klasöründe:
```
TNO_Image_Fusion_Dataset/
```

Test için örnek görüntüler:
- Thermal: `TNO_Image_Fusion_Dataset/TNO_Image_Fusion_Dataset/Athena_images/bunker/IR.bmp`
- Visible: `TNO_Image_Fusion_Dataset/TNO_Image_Fusion_Dataset/Athena_images/bunker/VIS.bmp`

---

## 💡 Ek Notlar

- **Python 3.11 kullanın!** 3.14 ile PyTorch CUDA uyumsuzluk var
- **NumPy 1.x kullanın!** 2.x ile PyTorch uyumsuz
- GPU kullanımını görmek için: Task Manager → Performance → GPU

---

## 📝 Hızlı Başlangıç (Tek Komut)

Tüm adımları otomatik yapmak için:

```bash
# 1. PyTorch CUDA kur
py -3.11 -m pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 2. Diğer paketleri kur
cd backend
py -3.11 -m pip install -r requirements.txt

# 3. Backend başlat
py -3.11 main.py
```

**Ayrı terminalde:**
```bash
cd frontend
py -3.11 -m http.server 8000
```

**Tarayıcı:** http://localhost:8000

---

## 🆘 Yardım

Sorun yaşıyorsanız:
1. `nvidia-smi` komutunu çalıştırın (GPU görünüyor mu?)
2. PyTorch CUDA testini yapın (yukarıdaki test komutu)
3. Backend başlatırken konsolu kontrol edin (GPU aktif mi?)

Başarılar! 🚀
