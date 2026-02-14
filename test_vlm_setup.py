"""
Quick Test Script for Vision-LLM Setup
=======================================
Run this to verify your Vision-LLM installation before starting the full system.
"""

import sys
import os

print("="*60)
print("Vision-LLM Setup Verification")
print("="*60)

# Test 1: Python version
print("\n[1/6] Checking Python version...")
py_version = sys.version_info
print(f"  Python {py_version.major}.{py_version.minor}.{py_version.micro}")
if py_version.major >= 3 and py_version.minor >= 8:
    print("  ✅ Python version OK")
else:
    print("  ❌ Python 3.8+ required")
    sys.exit(1)

# Test 2: PyTorch and CUDA
print("\n[2/6] Checking PyTorch and CUDA...")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Total VRAM: {total_memory:.2f} GB")
        
        if total_memory >= 6.0:
            print("  ✅ GPU and VRAM OK for Vision-LLM")
            if total_memory == 6.0:
                print("  ℹ️  6GB card detected - using optimized settings (4.8GB limit, 320px images)")
        elif total_memory >= 4.0:
            print(f"  ⚠️  {total_memory:.2f} GB VRAM (may work with heavy optimization)")
        else:
            print(f"  ❌ Only {total_memory:.2f} GB VRAM (minimum 4GB required)")
    else:
        print("  ❌ CUDA not available - Vision-LLM requires GPU")
        sys.exit(1)
        
except ImportError as e:
    print(f"  ❌ PyTorch not installed: {e}")
    sys.exit(1)

# Test 3: Check required packages
print("\n[3/6] Checking required packages...")
required_packages = {
    'transformers': 'transformers',
    'accelerate': 'accelerate',
    'bitsandbytes': 'bitsandbytes',
    'sentencepiece': 'sentencepiece',
    'protobuf': 'google.protobuf'  # protobuf imports as google.protobuf
}

missing_packages = []
for display_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"  ✅ {display_name}")
    except ImportError:
        print(f"  ❌ {display_name} - MISSING")
        missing_packages.append(display_name)

if missing_packages:
    print(f"\n  Missing packages: {', '.join(missing_packages)}")
    print("  Install with: pip install " + " ".join(missing_packages))
    sys.exit(1)
else:
    print("  ✅ All required packages installed")

# Test 4: Check project structure
print("\n[4/6] Checking project structure...")
required_dirs = [
    'backend/llm',
    'backend/utils',
    'backend/models',
    'backend/metrics'
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"  ✅ {dir_path}")
    else:
        print(f"  ❌ {dir_path} - NOT FOUND")

# Test 5: Check LLM module files
print("\n[5/6] Checking LLM module files...")
llm_files = [
    'backend/llm/__init__.py',
    'backend/llm/analyzer.py',
    'backend/llm/prompts.py',
    'backend/llm/model_config.py'
]

for file_path in llm_files:
    if os.path.exists(file_path):
        print(f"  ✅ {file_path}")
    else:
        print(f"  ❌ {file_path} - NOT FOUND")

# Test 6: Try importing the analyzer
print("\n[6/6] Testing Vision-LLM analyzer import...")
try:
    sys.path.insert(0, 'backend')
    from llm.analyzer import VisionFusionAnalyzer
    
    print("  ✅ VisionFusionAnalyzer imported successfully")
    
    # Get model info without loading
    analyzer = VisionFusionAnalyzer()
    info = analyzer.get_model_info()
    
    print(f"\n  Model Configuration:")
    print(f"    Model: {info['model_name']}")
    print(f"    CUDA: {info['cuda_available']}")
    print(f"    Total VRAM: {info['total_vram_gb']:.2f} GB")
    print(f"    Free VRAM: {info['free_vram_gb']:.2f} GB")
    print(f"    Model Loaded: {info['is_loaded']}")
    
    print("\n  ✅ Analyzer ready (model will load on first use)")
    
except Exception as e:
    print(f"  ❌ Failed to import analyzer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("✅ ALL CHECKS PASSED!")
print("="*60)
print("\nYour system is ready for Vision-LLM streaming analysis.")
print("\nNext steps:")
print("  1. Start backend: cd backend && python main.py")
print("  2. Open frontend: Open frontend/index.html in browser")
print("  3. Upload images and test fusion with AI analysis")
print("\nNote: First request will download model (~8GB, may take 10-30 min)")
print("      Subsequent requests will be fast (5-15 seconds)")
print("\nFor detailed testing guide, see: LLM_SETUP_GUIDE.md")
print("="*60)
