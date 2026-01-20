# DeepFusionColor Hızlı Başlangıç Kılavuzu

## 🚀 5 Dakikada Başla!

### Adım 1: Bağımlılıkları Yükle (2 dk)
```bash
cd backend
pip install -r requirements.txt
```

### Adım 2: Backend'i Başlat (1 dk)
```bash
python main.py
```

Backend başladı! `http://localhost:5000` adresinde çalışıyor.

### Adım 3: Frontend'i Aç (1 dk)
Yeni bir terminal/command prompt aç:
```bash
cd frontend
python -m http.server 8000
```

Tarayıcıda `http://localhost:8000` adresine git.

### Adım 4: İlk Füzyonunu Yap! (1 dk)
1. İki görüntü yükle (TNO_Image_Fusion_Dataset'ten)
2. Bir yöntem seç (başlangıç için "Wavelet" öneriyorum - en hızlı)
3. "Füzyon Yap" butonuna tıkla
4. Sonuçları incele!

---

## 🎯 Hızlı Test

Komut satırından hızlı test:

```python
# test_quick.py oluştur ve çalıştır
python
```

```python
from backend.models.wavelet_fusion import wavelet_fusion
from backend.utils.image_utils import load_image, save_image

# Görüntüleri yükle (kendi yollarını kullan)
thermal = load_image('TNO_Image_Fusion_Dataset/TNO_Image_Fusion_Dataset/Athena_images/bunker/EO_Bunker.bmp')
visible = load_image('TNO_Image_Fusion_Dataset/TNO_Image_Fusion_Dataset/Athena_images/bunker/IR_Bunker.bmp')

# Füzyon yap
fused = wavelet_fusion(thermal, visible)

# Kaydet
save_image(fused, 'results/my_first_fusion.png')

print("✅ İlk füzyonun hazır! results/my_first_fusion.png dosyasına bak!")
```

---

## 📝 Sık Karşılaşılan Hatalar

### Hata: "Module not found"
**Çözüm**: 
```bash
pip install -r backend/requirements.txt
```

### Hata: "CUDA out of memory"
**Çözüm**: 
- Görüntü boyutunu küçült
- Batch size'ı azalt
- CPU kullan (otomatik fallback olacak)

### Hata: "Port already in use"
**Çözüm**: 
- Backend için: `python main.py` yerine farklı port kullan
  ```python
  # main.py'de son satırı değiştir:
  app.run(debug=True, host='0.0.0.0', port=5001)
  ```
- Frontend için: `python -m http.server 8001`

---

## 💡 Pro İpuçları

1. **Hızlı test için**: Wavelet veya VIF kullan
2. **En iyi sonuç için**: DenseFuse kullan (ama yavaş)
3. **Batch test**: `python tests/test_all_methods.py` çalıştır
4. **GPU varsa**: PyTorch otomatik kullanır, endişelenme!

---

## 📞 Yardım Lazım?

README.md dosyasını oku - her şey detaylı anlatılmış!

İyi füzyonlar! 🎉
