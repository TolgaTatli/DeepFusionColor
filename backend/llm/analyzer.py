"""
Vision-LLM Streaming Analyzer
==============================
Streaming analysis of fusion results using Vision-Language Model.
"""

import torch
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from PIL import Image
import logging

from .model_config import (
    MODEL_NAME, 
    get_model_kwargs, 
    get_generation_config,
    check_vram_available,
    clear_vram_cache
)
from .prompts import SYSTEM_PROMPT, create_user_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionFusionAnalyzer:
    """
    Vision-Language Model analyzer for image fusion results.
    Provides streaming academic analysis with content understanding.
    """
    
    def __init__(self):
        """Initialize analyzer (model loaded lazily on first use)."""
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_loaded = False
        logger.info("VisionFusionAnalyzer initialized (lazy loading)")
    
    def _load_model(self):
        """
        Load Vision-LLM model with 4-bit quantization.
        Only called on first analysis request.
        """
        if self.is_loaded:
            return
        
        logger.info("Loading Vision-LLM model (this may take 30-60 seconds)...")
        
        # Check VRAM availability
        cuda_available, total_vram, free_vram = check_vram_available()
        if not cuda_available:
            raise RuntimeError("CUDA not available. Vision-LLM requires GPU.")
        
        logger.info(f"VRAM: {free_vram:.2f}GB free / {total_vram:.2f}GB total")
        
        if free_vram < 4.5:
            logger.warning(f"Low VRAM detected ({free_vram:.2f}GB). Using conservative settings.")
        
        if total_vram <= 6.0:
            logger.info("6GB VRAM card detected - using optimized settings (320px images, 768 tokens)")
        
        try:
            # Load model with quantization
            model_kwargs = get_model_kwargs()
            
            logger.info(f"Loading model: {MODEL_NAME}")
            self.model = AutoModel.from_pretrained(
                MODEL_NAME,
                **model_kwargs
            )
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            
            # Set model to eval mode
            self.model.eval()
            
            self.is_loaded = True
            
            # Check VRAM after loading
            _, _, free_vram_after = check_vram_available()
            vram_used = free_vram - free_vram_after
            logger.info(f"Model loaded successfully! VRAM used: {vram_used:.2f}GB")
            logger.info(f"VRAM remaining: {free_vram_after:.2f}GB")
            
        except Exception as e:
            logger.error(f"Failed to load Vision-LLM model: {e}")
            raise
    
    def analyze_fusion_streaming(self, ir_image_pil, vis_image_pil, 
                                fused_image_pil, metrics_dict, method_name):
        """
        Generate streaming analysis of fusion result.
        
        Args:
            ir_image_pil: PIL.Image - Infrared source image
            vis_image_pil: PIL.Image - Visible source image
            fused_image_pil: PIL.Image - Fusion result
            metrics_dict: dict - Metrics (PSNR, SSIM, SF, MI, Entropy, MSE)
            method_name: str - Name of fusion method
        
        Yields:
            str: Text chunks as they are generated
        """
        # Load model if not already loaded
        if not self.is_loaded:
            self._load_model()
        
        try:
            # Prepare prompt
            user_prompt = create_user_prompt(metrics_dict, method_name)
            
            # Prepare images (resize for VRAM efficiency)
            images = self._prepare_images([ir_image_pil, vis_image_pil, fused_image_pil])
            
            # Create conversation
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            
            # Prepare inputs
            logger.info("Preparing model inputs...")
            
            # For MiniCPM-V, we need to format the input properly
            # This is a simplified version - actual implementation may vary
            prompt_text = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Get generation config
            generation_config = get_generation_config()
            generation_config["streamer"] = streamer
            
            # Prepare model inputs
            # Note: This is a simplified version. Actual MiniCPM-V implementation
            # may require different input formatting with images
            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            logger.info("Starting generation thread...")
            
            # Start generation in separate thread
            generation_kwargs = {**inputs, **generation_config}
            thread = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            thread.start()
            
            # Yield text chunks as they arrive
            logger.info("Streaming analysis started")
            for text_chunk in streamer:
                if text_chunk:
                    yield text_chunk
            
            # Wait for thread to complete
            thread.join()
            logger.info("Streaming analysis completed")
            
            # Clean up VRAM
            clear_vram_cache()
            
        except Exception as e:
            logger.error(f"Error during streaming analysis: {e}")
            # Yield error message
            yield f"\n\n[Error: Analysis failed - {str(e)}]"
            clear_vram_cache()
    
    def _prepare_images(self, images_pil, target_size=320):
        """
        Prepare images for model input (resize for VRAM efficiency).
        Optimized for 6GB VRAM cards - using 320px instead of 448px.
        
        Args:
            images_pil: list of PIL.Image
            target_size: int - Target dimension (default 320 for 6GB VRAM)
        
        Returns:
            list: Processed PIL images
        """
        processed = []
        for img in images_pil:
            # Resize if needed
            if max(img.size) > target_size:
                img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            processed.append(img)
        
        return processed
    
    def analyze_fusion_simple(self, ir_image_pil, vis_image_pil,
                             fused_image_pil, metrics_dict, method_name):
        """
        Non-streaming version (collects all text then returns).
        
        Args:
            Same as analyze_fusion_streaming
        
        Returns:
            str: Complete analysis text
        """
        full_text = ""
        for chunk in self.analyze_fusion_streaming(
            ir_image_pil, vis_image_pil, fused_image_pil, 
            metrics_dict, method_name
        ):
            full_text += chunk
        
        return full_text
    
    def get_model_info(self):
        """
        Get model status and VRAM info.
        
        Returns:
            dict: Model information
        """
        cuda_available, total_vram, free_vram = check_vram_available()
        
        return {
            "model_name": MODEL_NAME,
            "is_loaded": self.is_loaded,
            "cuda_available": cuda_available,
            "total_vram_gb": total_vram,
            "free_vram_gb": free_vram,
            "vram_usage_gb": total_vram - free_vram if cuda_available else 0
        }
