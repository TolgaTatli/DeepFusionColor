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
import json
import time
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# Kendi modüllerimizi import et
from models.wavelet_fusion import wavelet_fusion
from models.dnn_fusion import dnn_fusion, DNNFusionTrainer
from models.cnn_fusion import cnn_fusion, CNNFusionTrainer
from models.latentlrr_fusion import latentlrr_fusion
from models.vif_fusion import vif_fusion
from models.densefuse_fusion import densefuse_fusion, DenseFuseTrainer
from metrics.evaluation_metrics import calculate_all_metrics
from utils.image_utils import load_image, save_image, preprocess_for_fusion, convert_to_uint8
from utils.session_manager import (
    generate_session_id,
    store_session_data,
    load_session_data,
    cleanup_session,
    cleanup_old_sessions
)

# Flask app oluştur
app = Flask(__name__)
CORS(app)  # Cross-origin isteklere izin ver

# Sonuçları kaydetme dizini
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pre-trained modeller dizini
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')

# Global model cache (her istek için yeniden yüklemek yerine)
PRETRAINED_MODELS = {
    'dnn': None,
    'cnn': None,
    'densefuse': None
}

# Global Vision-LLM analyzer (lazy loading)
VISION_ANALYZER = None

def get_vision_analyzer():
    """Get or create Vision-LLM analyzer instance."""
    global VISION_ANALYZER
    if VISION_ANALYZER is None:
        try:
            from llm.analyzer import VisionFusionAnalyzer
            VISION_ANALYZER = VisionFusionAnalyzer()
            print("  ✅ Vision-LLM analyzer initialized (will load on first use)")
        except Exception as e:
            print(f"  ⚠️ Vision-LLM analyzer initialization failed: {e}")
            print("  ℹ️ Streaming analysis will not be available")
            return None
    return VISION_ANALYZER


def load_pretrained_models():
    """
    Pre-trained modelleri yükler (varsa)
    """
    print("\n[STARTUP] Checking for pre-trained models...")
    # DNN model
    dnn_path = os.path.join(MODELS_DIR, 'dnn_fusion_model.pth')
    if os.path.exists(dnn_path):
        try:
            PRETRAINED_MODELS['dnn'] = DNNFusionTrainer(
                hidden_sizes=[256, 128, 64],
                pretrained_path=dnn_path
            )
            print("  ✅ DNN model loaded")
        except Exception as e:
            print(f"  ⚠️ DNN model load failed: {e}")
    else:
        print("  ⚠️ DNN model not found (will train on-the-fly)")
    
    # 
    # CNN model
    cnn_path = os.path.join(MODELS_DIR, 'cnn_fusion_model.pth')
    if os.path.exists(cnn_path):
        try:
            PRETRAINED_MODELS['cnn'] = CNNFusionTrainer(
                num_filters=[16, 32, 64],
                kernel_size=3,
                pretrained_path=cnn_path
            )
            print("  ✅ CNN model loaded")
        except Exception as e:
            print(f"  ⚠️ CNN model load failed: {e}")
    else:
        print("  ⚠️ CNN model not found (will train on-the-fly)")
    
    # DenseFuse model
    densefuse_path = os.path.join(MODELS_DIR, 'densefuse_model.pth')
    if os.path.exists(densefuse_path):
        try:
            PRETRAINED_MODELS['densefuse'] = DenseFuseTrainer(
                growth_rate=16,
                num_blocks=3,
                num_layers_per_block=4,
                pretrained_path=densefuse_path
            )
            print("  ✅ DenseFuse model loaded")
        except Exception as e:
            print(f"  ⚠️ DenseFuse model load failed: {e}")
    else:
        print("  ⚠️ DenseFuse model not found (will train on-the-fly)")
    
    print("[STARTUP] Ready!\n")


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
            'name': 'DenseFuse',
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
            # Pre-trained model varsa kullan, yoksa on-the-fly training
            if PRETRAINED_MODELS['dnn']:
                print("[API] Using pre-trained DNN model")
                fused = PRETRAINED_MODELS['dnn'].predict(img1, img2)
            else:
                print("[API] No pre-trained DNN, training on-the-fly (slow!)")
                fused = dnn_fusion(img1, img2, **params)
        elif method == 'cnn':
            # Pre-trained model varsa kullan, yoksa on-the-fly training
            if PRETRAINED_MODELS['cnn']:
                print("[API] Using pre-trained CNN model")
                fused = PRETRAINED_MODELS['cnn'].predict(img1, img2)
            else:
                print("[API] No pre-trained CNN, training on-the-fly (slow!)")
                fused = cnn_fusion(img1, img2, **params)
        elif method == 'latentlrr':
            fused = latentlrr_fusion(img1, img2, **params)
        elif method == 'vif':
            fused = vif_fusion(img1, img2, **params)
        elif method == 'densefuse':
            # Pre-trained model varsa kullan, yoksa on-the-fly training
            if PRETRAINED_MODELS['densefuse']:
                print("[API] Using pre-trained DenseFuse model")
                fused = PRETRAINED_MODELS['densefuse'].predict(img1, img2)
            else:
                print("[API] No pre-trained DenseFuse, training on-the-fly (slow!)")
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
        
        # Store session data for streaming analysis
        session_id = generate_session_id()
        try:
            # Convert to PIL images for storage
            img1_pil = Image.fromarray(convert_to_uint8(img1))
            img2_pil = Image.fromarray(convert_to_uint8(img2))
            fused_pil = Image.fromarray(convert_to_uint8(fused))
            
            # Store for streaming analysis
            store_session_data(session_id, img1_pil, img2_pil, fused_pil, metrics_json, method)
            print(f"[API] Session {session_id} stored for streaming analysis")
        except Exception as e:
            print(f"[API Warning] Failed to store session: {e}")
            session_id = None
        
        return jsonify({
            'success': True,
            'fused_image': fused_base64,
            'metrics': metrics_json,
            'method': method,
            'session_id': session_id,
            'analysis_url': f'/stream-analysis/{session_id}' if session_id else None
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


@app.route('/stream-analysis/<session_id>')
def stream_analysis(session_id):
    """
    Server-Sent Events endpoint for streaming AI analysis.
    
    Args:
        session_id: Session identifier from /fusion response
    
    Returns:
        SSE stream with analysis chunks
    """
    def generate():
        try:
            # Load session data
            session_data = load_session_data(session_id)
            if session_data is None:
                error_data = json.dumps({
                    'type': 'error',
                    'message': 'Session not found or expired'
                })
                yield f'data: {error_data}\n\n'
                return
            
            ir_img, vis_img, fused_img, metrics, method = session_data
            
            # Get analyzer
            analyzer = get_vision_analyzer()
            if analyzer is None:
                error_data = json.dumps({
                    'type': 'error',
                    'message': 'Vision-LLM analyzer not available'
                })
                yield f'data: {error_data}\n\n'
                return
            
            # Send start signal
            yield 'data: {"type": "start"}\n\n'
            
            # Stream analysis
            print(f"[API] Starting streaming analysis for session {session_id}")
            for text_chunk in analyzer.analyze_fusion_streaming(
                ir_img, vis_img, fused_img, metrics, method
            ):
                if text_chunk:
                    data = json.dumps({
                        'type': 'chunk',
                        'text': text_chunk
                    })
                    yield f'data: {data}\n\n'
                    time.sleep(0.01)  # Small delay for smooth rendering
            
            # Send done signal
            yield 'data: {"type": "done"}\n\n'
            print(f"[API] Streaming analysis completed for session {session_id}")
            
            # Cleanup session
            cleanup_session(session_id)
            
        except Exception as e:
            print(f"[API Error] Streaming analysis failed: {e}")
            import traceback
            traceback.print_exc()
            error_data = json.dumps({
                'type': 'error',
                'message': f'Analysis failed: {str(e)}'
            })
            yield f'data: {error_data}\n\n'
            
            # Cleanup on error
            try:
                cleanup_session(session_id)
            except:
                pass
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@app.route('/health', methods=['GET'])
def health_check():
    """
    API sağlık kontrolü
    """
    # Cleanup old sessions
    cleanup_old_sessions()
    
    analyzer = get_vision_analyzer()
    analyzer_status = 'available' if analyzer else 'unavailable'
    
    return jsonify({
        'status': 'healthy',
        'message': 'DeepFusionColor API is running!',
        'vision_llm': analyzer_status
    })


if __name__ == '__main__':
    print("="*60)
    print("DeepFusionColor Backend API")
    print("="*60)
    
    # Pre-trained modelleri yükle
    load_pretrained_models()
    
    # Initialize Vision-LLM analyzer
    get_vision_analyzer()
    
    # Flask uygulamasını başlat
    print("Available endpoints:")
    print("  GET  /health              - Health check")
    print("  GET  /methods             - List fusion methods")
    print("  POST /fusion              - Perform image fusion")
    print("  POST /metrics             - Calculate metrics")
    print("  GET  /stream-analysis/:id - Stream AI analysis (SSE)")
    print("="*60)
    print("Starting server on http://localhost:5000")
    print("="*60 + "\n")
    
    # Development server (production için gunicorn kullan)
    app.run(debug=False, host='0.0.0.0', port=5000)
