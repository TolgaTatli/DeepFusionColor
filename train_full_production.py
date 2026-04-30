"""
Full LLVIP Dataset Training Script
===================================
Tüm LLVIP dataseti (12025 train sample) ile ultimate model eğitimi

Bu script:
- Tüm LLVIP training data ile eğitim yapar
- Production-ready modeller oluşturur
- ~4-6 saat alır (CPU'da) veya 30+ dakika (GPU'da)

Kullanım:
    python train_full_production.py
    
    # GPU'da çalıştırmak için:
    CUDA_VISIBLE_DEVICES=0 python train_full_production.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime, timedelta

# Backend modüllerini import et
sys.path.append(os.path.dirname(__file__))

def print_header(text):
    """Başlık yazdır"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_info(text):
    """Bilgi yazdır"""
    print(f"[INFO] {text}")

def print_warning(text):
    """Uyarı yazdır"""
    print(f"[WARNING] {text}")

def print_success(text):
    """Başarı mesajı yazdır"""
    print(f"[OK] {text}")

def estimate_training_time():
    """Eğitim süresi tahmini"""
    import torch
    
    has_gpu = torch.cuda.is_available()
    
    if has_gpu:
        device_name = torch.cuda.get_device_name(0)
        print_success(f"GPU detected: {device_name}")
        estimated = "30-45 minutes"
        multiplier = "CPU'dan ~10x daha HIZLI"
    else:
        print_warning("GPU bulunamadı - CPU ile çalışacak")
        estimated = "4-6 hours"
        multiplier = "1x (çok yavaş!)"
    
    print_info(f"Tahmini eğitim süresi: {estimated} ({multiplier})")
    return has_gpu

def run_full_training():
    """Full eğitim çalıştır"""
    
    print_header("LLVIP PRODUCTION MODEL TRAINING")
    
    print_info("Bu script tüm LLVIP dataseti ile eğitim yapacak")
    print_info("Tüm 12,025 training sample kullanılacak")
    
    # GPU check
    has_gpu = estimate_training_time()
    
    print_warning("Eğitim saati boyunca bilgisayarı açık tutun!")
    print_warning("Eğitim işlemi kapatılırsa yeniden başlanacak")
    
    # Onay al
    user_input = input("\nDevam etmek istiyor musun? (evet/no): ").strip().lower()
    if user_input != "evet":
        print("Eğitim iptal edildi.")
        return
    
    # Training komutunu hazırla
    cmd = [
        sys.executable,
        "backend\\train_models.py",
        "--model", "all",  # DNN, CNN, DenseFuse
        "--epochs-dnn", "20",        # DNN epochs
        "--epochs-cnn", "40",        # CNN epochs (daha yavaş)
        "--epochs-densefuse", "30",  # DenseFuse epochs
        "--batch-size", "16"         # Batch size
    ]
    
    # CPU/GPU'ya göre parametreler ayarla
    if not has_gpu:
        print_warning("CPU kullanılıyor - batch size küçültülüyor")
        cmd = cmd[:-2] + ["--batch-size", "8"]
    
    print_header("EĞITIM BAŞLANIYOR")
    print_info(f"Komut: {' '.join(cmd)}")
    print_info(f"Başlama zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Eğitim çalıştır
    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        
        elapsed_time = time.time() - start_time
        elapsed_hours = elapsed_time / 3600
        
        if result.returncode == 0:
            print_header("TRAINING COMPLETE!")
            print_success(f"Eğitim başarıyla tamamlandı!")
            print_success(f"Aldığı süre: {elapsed_hours:.1f} saat ({int(elapsed_time/60)} dakika)")
            print_success(f"Bitirme zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            models_dir = "backend\\trained_models"
            if os.path.exists(models_dir):
                models = os.listdir(models_dir)
                print_success(f"Kaydedilen modeller ({len(models)}):")
                for model in models:
                    size_mb = os.path.getsize(os.path.join(models_dir, model)) / (1024*1024)
                    print(f"  - {model} ({size_mb:.1f} MB)")
        else:
            print_warning(f"Eğitim hata ile sonlandı (code: {result.returncode})")
            
    except KeyboardInterrupt:
        print_warning("\n\nEğitim kullanıcı tarafından durduruldu!")
        elapsed_time = time.time() - start_time
        elapsed_hours = elapsed_time / 3600
        print_info(f"Çalışma süresi: {elapsed_hours:.1f} saat")
        print_info("Eğitimi tekrar başlatabilirsin - kaldığı yerden devam edecek")

def main():
    """Ana fonksiyon"""
    try:
        run_full_training()
        
    except Exception as e:
        print_warning(f"Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("\n")
    main()
    print("\n")
