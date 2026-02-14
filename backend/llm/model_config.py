"""
Vision-LLM Model Configuration
================================
Configuration for MiniCPM-V 2.6 with 6GB VRAM optimization.
"""

import torch
from transformers import BitsAndBytesConfig

# Model identifier
# Option 1: MiniCPM-V-2_6 (gated - requires HuggingFace login)
# MODEL_NAME = "openbmb/MiniCPM-V-2_6"

# Option 2: Qwen2-VL-7B (no gate, works immediately)
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# VRAM limit (in GB) - Conservative for 6GB cards
MAX_VRAM_GB = 4.8  # Leave buffer for system and PyTorch overhead


def get_quantization_config():
    """
    Get 4-bit quantization config for VRAM optimization.
    
    Returns:
        BitsAndBytesConfig: Quantization configuration
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True
        
    )


def get_model_kwargs():
    """
    Get model loading kwargs with VRAM optimization.
    
    Returns:
        dict: Model loading arguments
    """
    return {
        "quantization_config": get_quantization_config(),
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "max_memory": {0: f"{MAX_VRAM_GB}GB"}
    }


def get_generation_config():
    """
    Get generation configuration for streaming.
    Optimized for 6GB VRAM cards.
    
    Returns:
        dict: Generation parameters
    """
    return {
        "max_new_tokens": 768,  # Reduced from 1024 for VRAM safety
        "temperature": 0.4,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.15,
    }


def check_vram_available():
    """
    Check if CUDA is available and get VRAM info.
    
    Returns:
        tuple: (is_available, total_vram_gb, free_vram_gb)
    """
    if not torch.cuda.is_available():
        return False, 0, 0
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_vram = (torch.cuda.get_device_properties(0).total_memory - 
                 torch.cuda.memory_allocated(0)) / (1024**3)
    
    return True, total_vram, free_vram


def clear_vram_cache():
    """Clean up VRAM cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
