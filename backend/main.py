"""
DeepFusionColor - Ana Backend API
==================================
Flask web server ile frontend'e servis sağlar.

Endpoints:
- /fusion: Görüntü füzyonu yapar
- /metrics: Metrik hesaplamaları
- /methods: Mevcut füzyon yöntemlerini listeler
"""

import os
import sys
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Kendi modüllerimizi import et
from models.wavelet_fusion import wavelet_fusion
from models.dnn_fusion import dnn_fusion
from models.cnn_fusion import cnn_fusion
from models.latentlrr_fusion import latentlrr_fusion
from models.vif_fusion import vif_fusion
from models.densefuse_fusion import densefuse_fusion
from metrics.evaluation_metrics import calculate_all_metrics
from utils.image_utils import load_image, save_image, preprocess_for_fusion, convert_to_uint8

# Flask app oluştur
app = Flask(__name__)
CORS(app)  # Cross-origin isteklere izin ver

# Sonuçları kaydetme dizini
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def image_to_base64(image_array):
    """
    NumPy array'i base64 string'e çevirir (frontend'e göndermek için)
    
    Parametreler:
    ------------
    image_array : numpy.ndarray
        Görüntü array'i
        
    Returns:
    -------
    str : Base64 encoded string
    """
    # uint8'e çevir
    img_uint8 = convert_to_uint8(image_array)
    
    # PIL Image'a çevir
    pil_img = Image.fromarray(img_uint8)
    
    # Base64'e encode et
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str


def base64_to_image(base64_str):
    """
    Base64 string'i NumPy array'e çevirir
    
    Parametreler:
    ------------
    base64_str : str
        Base64 encoded görüntü
        
    Returns:
    -------
    numpy.ndarray : Görüntü array'i
    """
    # Base64'ten decode et
    img_data = base64.b64decode(base64_str.split(',')[1] if ',' in base64_str else base64_str)
    
    # PIL Image'a çevir
    pil_img = Image.open(BytesIO(img_data))
    
    # NumPy array'e çevir
    img_array = np.array(pil_img)
    
    # Grayscale'e çevir (eğer renkli ise)
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)
    
    # [0, 1] aralığına normalize et
    img_array = img_array.astype(np.float32) / 255.0
    
    return img_array


@app.route('/methods', methods=['GET'])
def get_methods():
    """
    Mevcut füzyon yöntemlerini listeler
    """
    methods = [
        {
            'id': 'wavelet',
            'name': 'Wavelet Fusion',
            'type': 'Traditional',
            'description': 'Wavelet dönüşümü tabanlı klasik yöntem',
            'speed': 'Fast',
            'quality': 'Good'
        },
        {
            'id': 'dnn',
            'name': 'Deep Neural Network',
            'type': 'Deep Learning',
            'description': 'Fully connected neural network',
            'speed': 'Medium',
            'quality': 'Good'
        },
        {
            'id': 'cnn',
            'name': 'Convolutional Neural Network',
            'type': 'Deep Learning',
            'description': 'CNN ile spatial feature extraction',
            'speed': 'Medium',
            'quality': 'Very Good'
        },
        {
            'id': 'latentlrr',
            'name': 'Latent LRR',
            'type': 'Traditional',
            'description': 'Low-rank representation tabanlı',
            'speed': 'Slow',
            'quality': 'Very Good'
        },
        {
            'id': 'vif',
            'name': 'Visual Information Fidelity',
            'type': 'Traditional',
            'description': 'Multi-scale visual saliency tabanlı',
            'speed': 'Medium',
            'quality': 'Very Good'
        },
        {
            'id': 'densefuse',
            'name': 'DenseFuse (SOTA)',
            'type': 'Deep Learning',
            'description': 'Dense blocks ile state-of-the-art fusion',
            'speed': 'Slow',
            'quality': 'Excellent'
        }
    ]
    
    return jsonify({'methods': methods})


@app.route('/fusion', methods=['POST'])
def perform_fusion():
    """
    Görüntü füzyonu yapar
    
    Request Body:
    {
        'image1': 'base64_encoded_image',
        'image2': 'base64_encoded_image',
        'method': 'wavelet|dnn|cnn|latentlrr|vif|densefuse',
        'params': {...}  # Method-specific parameters (optional)
    }
    """
    try:
        data = request.json
        
        # Görüntüleri al
        img1 = base64_to_image(data['image1'])
        img2 = base64_to_image(data['image2'])
        method = data.get('method', 'wavelet')
        params = data.get('params', {})
        
        print(f"\n[API] Fusion request: method={method}, img1 shape={img1.shape}, img2 shape={img2.shape}")
        
        # Görüntüleri aynı boyuta getir
        img1, img2 = preprocess_for_fusion(img1, img2, target_size=(256, 256), normalize=True)
        
        # Füzyon yap
        if method == 'wavelet':
            fused = wavelet_fusion(img1, img2, **params)
        elif method == 'dnn':
            fused = dnn_fusion(img1, img2, **params)
        elif method == 'cnn':
            fused = cnn_fusion(img1, img2, **params)
        elif method == 'latentlrr':
            fused = latentlrr_fusion(img1, img2, **params)
        elif method == 'vif':
            fused = vif_fusion(img1, img2, **params)
        elif method == 'densefuse':
            fused = densefuse_fusion(img1, img2, **params)
        else:
            return jsonify({'error': f'Unknown method: {method}'}), 400
        
        # Metrikleri hesapla
        metrics = calculate_all_metrics(img1, fused, img2, verbose=True)
        
        # Metrikleri JSON serializable hale getir (numpy types -> Python types)
        metrics_json = {
            key: float(value) if isinstance(value, (np.floating, np.integer)) else value
            for key, value in metrics.items()
        }
        
        # Sonucu base64'e çevir
        fused_base64 = image_to_base64(fused)
        
        print(f"[API] Fusion completed successfully!")
        
        return jsonify({
            'success': True,
            'fused_image': fused_base64,
            'metrics': metrics_json,
            'method': method
        })
        
    except Exception as e:
        print(f"[API Error] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['POST'])
def calculate_metrics():
    """
    Metrik hesaplaması yapar
    
    Request Body:
    {
        'image1': 'base64_encoded_image',
        'image2': 'base64_encoded_image',
        'fused': 'base64_encoded_image'
    }
    """
    try:
        data = request.json
        
        img1 = base64_to_image(data['image1'])
        img2 = base64_to_image(data['image2'])
        fused = base64_to_image(data['fused'])
        
        # Metrikleri hesapla
        metrics = calculate_all_metrics(img1, fused, img2, verbose=True)
        
        # JSON serializable hale getir
        metrics_json = {
            key: float(value) if isinstance(value, (np.floating, np.integer)) else value
            for key, value in metrics.items()
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics_json
        })
        
    except Exception as e:
        print(f"[API Error] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    API sağlık kontrolü
    """
    return jsonify({
        'status': 'healthy',
        'message': 'DeepFusionColor API is running!'
    })


if __name__ == '__main__':
    print("="*60)
    print("DeepFusionColor Backend API")
    print("="*60)
    print("Available endpoints:")
    print("  GET  /health     - Health check")
    print("  GET  /methods    - List fusion methods")
    print("  POST /fusion     - Perform image fusion")
    print("  POST /metrics    - Calculate metrics")
    print("="*60)
    print("Starting server on http://localhost:5000")
    print("="*60 + "\n")
    
    # Development server (production için gunicorn kullan)
    app.run(debug=True, host='0.0.0.0', port=5000)
