# Vision-LLM Streaming Analysis Setup Guide

## 🎯 Overview

This guide will help you set up and test the new Vision-Language Model (VLM) streaming analysis feature. The system uses MiniCPM-V 2.6 to provide real-time, academic-style English analysis of fusion results.

## 📋 Prerequisites

- **GPU Required**: NVIDIA GPU with 6GB+ VRAM
- **CUDA**: Installed and configured
- **Python**: 3.8+
- **PyTorch**: Already installed with CUDA support

## 🔧 Installation Steps

### Step 1: Install New Dependencies

```powershell
# Navigate to backend directory
cd backend

# Install new Vision-LLM dependencies
pip install transformers>=4.40.0
pip install accelerate>=0.27.0
pip install bitsandbytes>=0.43.0
pip install sentencepiece>=0.2.0
pip install protobuf>=4.25.0
```

Or install all dependencies:

```powershell
pip install -r requirements.txt
```

### Step 2: Verify GPU Availability

```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output:
```
CUDA Available: True
GPU: [Your GPU Name]
```

### Step 3: Test Model Loading (Optional - First Run Test)

Create a test file `test_vlm.py`:

```python
from llm.analyzer import VisionFusionAnalyzer

analyzer = VisionFusionAnalyzer()
print("Analyzer initialized successfully")

# Check model info
info = analyzer.get_model_info()
print(f"Model: {info['model_name']}")
print(f"CUDA Available: {info['cuda_available']}")
print(f"Total VRAM: {info['total_vram_gb']:.2f} GB")
print(f"Free VRAM: {info['free_vram_gb']:.2f} GB")
```

Run:
```powershell
python test_vlm.py
```

**Note**: First time will download the model (~8GB). This may take 10-30 minutes depending on internet speed.

## 🚀 Running the System

### Start Backend Server

```powershell
cd backend
python main.py
```

Expected output should include:
```
✅ DNN model loaded
✅ CNN model loaded
✅ DenseFuse model loaded
✅ Vision-LLM analyzer initialized (will load on first use)

Available endpoints:
  GET  /health              - Health check
  GET  /methods             - List fusion methods
  POST /fusion              - Perform image fusion
  POST /metrics             - Calculate metrics
  GET  /stream-analysis/:id - Stream AI analysis (SSE)

Starting server on http://localhost:5000
```

### Open Frontend

```powershell
# In a new terminal, navigate to frontend
cd frontend

# Open index.html in browser
# Or use a simple HTTP server:
python -m http.server 8000
```

Open browser: `http://localhost:8000`

## 🧪 Testing the Streaming Analysis

### Test 1: Basic Fusion with Auto-Analysis

1. **Upload Images**:
   - Load an IR (thermal) image as Image 1
   - Load a Visible image as Image 2
   - Example images available in: `TNO_Image_Fusion_Dataset/`

2. **Select Method**:
   - Choose any fusion method (e.g., "Wavelet Fusion" for quick test)

3. **Click "Füzyon Yap"**:
   - Fusion will complete (~2-5 seconds)
   - Metrics will appear immediately
   - **AI Analysis section will automatically appear below**
   - Watch text stream word-by-word like ChatGPT (~10-15 seconds)

4. **Verify Streaming**:
   - Look for the "🤖 AI Analysis" section
   - Should see typing indicator (▋) blinking
   - Status: "Analyzing with Vision AI..."
   - Text appears progressively
   - Final formatted analysis with proper markdown

5. **Check Analysis Content**:
   - Should have these sections:
     - IMAGE CONTENT ANALYSIS (detects what's in images: human, vehicle, building, etc.)
     - METRIC EVALUATION (detailed analysis of all 6 metrics)
     - IMPROVEMENT RECOMMENDATIONS (specific suggestions)
     - OVERALL ASSESSMENT (summary)
   - All in English, academic style
   - 300-500 words total

### Test 2: Different Image Content

Try different image types to test content detection:
- Soldier images → Should mention "soldier", "human figure"
- Vehicle images → Should mention "vehicle", "military equipment"
- Building images → Should mention "structure", "building"
- Nature images → Should mention "vegetation", "outdoor scene"

### Test 3: Check Browser Developer Tools

1. Open DevTools (F12)
2. Go to **Network** tab
3. Filter by **EventStream**
4. Perform fusion
5. Look for `stream-analysis/[session-id]` request
6. Should see:
   - Type: `text/event-stream`
   - Multiple `data:` events streaming in
   - Status: 200 OK

### Test 4: Metric Highlighting

After analysis completes:
- Metric values should be color-coded:
  - 🟢 Green: Good values
  - 🟡 Yellow: Acceptable values
  - 🔴 Red: Poor values
- Check if PSNR, SSIM, SF, etc. are highlighted

## 🐛 Troubleshooting

### Issue: "Vision-LLM analyzer not available"

**Cause**: Model loading failed or dependencies missing

**Solution**:
```powershell
# Reinstall dependencies
pip install --upgrade transformers accelerate bitsandbytes

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "CUDA out of memory"

**Cause**: VRAM insufficient (need 6GB+)

**Solution**:
1. Close other GPU applications
2. Restart backend
3. Try with smaller images
4. Check VRAM: `nvidia-smi`

### Issue: "Analysis timeout"

**Cause**: Model taking too long to load on first request

**Solution**:
- First request after server start may take 30-60 seconds
- Subsequent requests should be faster (5-15 seconds)
- Check backend console for loading progress

### Issue: "Connection failed" or "SSE Error"

**Cause**: Backend not reachable or session expired

**Solution**:
1. Check backend is running on port 5000
2. Check CORS is enabled
3. Verify session ID in browser console
4. Try refreshing page and retry

### Issue: Analysis is in wrong language or too short

**Cause**: Prompt not properly applied

**Solution**:
- Check `backend/llm/prompts.py` - should have English prompts
- Verify `SYSTEM_PROMPT` is properly formatted
- Check backend console logs for generation details

## 📊 Performance Expectations

### First Request (Cold Start):
- Model loading: 30-60 seconds
- VRAM usage: 5-5.5 GB
- Analysis generation: 10-15 seconds
- **Total: ~45-75 seconds**

### Subsequent Requests (Warm):
- Model already loaded
- VRAM usage: ~5.5 GB (persistent)
- Analysis generation: 5-15 seconds
- **Total: 5-15 seconds**

### Analysis Quality Metrics:
- Length: 300-500 words
- Sections: 4-5 (Content, Metrics, Recommendations, Assessment)
- Metric coverage: All 6 metrics analyzed
- Language: English, academic tone
- Content detection: Objects/scenes identified

## 🔍 Monitoring

### Backend Logs

Watch for these key messages:

```
[API] Fusion completed successfully!
[API] Session {id} stored for streaming analysis
[API] Starting streaming analysis for session {id}
Loading Vision-LLM model (this may take 30-60 seconds)...
VRAM: X.XX GB free / Y.YY GB total
Model loaded successfully! VRAM used: Z.ZZ GB
[API] Streaming analysis completed for session {id}
```

### VRAM Monitoring

```powershell
# Watch VRAM usage in real-time
nvidia-smi --query-gpu=memory.used,memory.free --format=csv --loop=1
```

### Session Cleanup

Sessions are automatically cleaned up:
- After analysis completes
- After 1 hour if unused
- On `/health` endpoint call

Check session stats:
```python
from utils.session_manager import get_session_stats
stats = get_session_stats()
print(stats)
```

## 🎓 Example Analysis Output

Here's what a good analysis should look like:

```
## IMAGE CONTENT ANALYSIS

This fusion image depicts a military scene with a soldier figure positioned 
in an outdoor environment. The thermal source clearly highlights the human 
subject through temperature contrast, while the visible image provides 
contextual background details including vegetation and terrain features. The 
fusion successfully combines both modalities.

## METRIC EVALUATION

Based on the metrics:

### PSNR (Peak Signal-to-Noise Ratio): 28.35 dB

The PSNR value of 28.35 dB falls within the acceptable range (20-30 dB) for 
fusion quality. While this indicates reasonable signal preservation, it falls 
short of the optimal threshold of 30+ dB. This suggests some noise artifacts 
may be present, particularly in homogeneous regions. The value indicates 
moderate reconstruction quality from the source images...

[continues with detailed analysis of all metrics]

## IMPROVEMENT RECOMMENDATIONS

The spatial frequency value of 18.2 is below the optimal threshold of 20+, 
indicating edge preservation could be enhanced. For scenes containing human 
subjects, edge clarity is critical for detection and recognition tasks. 
Consider using DenseFuse or CNN Fusion methods which employ edge-preserving 
architectures...

[continues with specific recommendations]

## OVERALL ASSESSMENT AND RECOMMENDATIONS

Overall, the fusion achieves acceptable quality with good structural 
preservation (SSIM: 0.87) and information transfer (MI: 2.14). For military 
surveillance applications involving human detection, switching to DenseFuse 
method may yield sharper results. Additionally, pre-processing with edge 
enhancement could improve the SF metric...
```

## ✅ Success Criteria

Your implementation is working correctly if:

- ✅ Backend starts without errors
- ✅ Vision-LLM analyzer initializes
- ✅ Fusion completes and returns session_id
- ✅ AI Analysis section appears automatically
- ✅ Text streams word-by-word (visible in UI)
- ✅ Analysis completes in 5-15 seconds (after first load)
- ✅ Final analysis is formatted with markdown
- ✅ Content detection works (mentions objects in images)
- ✅ All 6 metrics are analyzed
- ✅ Improvement recommendations are specific
- ✅ Language is English and academic
- ✅ 300-500 words total length
- ✅ VRAM usage stays under 6GB
- ✅ Sessions cleanup properly
- ✅ Multiple requests work consecutively

## 📚 Next Steps

After successful testing:

1. **Fine-tuning (Optional)**:
   - Generate training dataset with `llm_data_generator.py`
   - Fine-tune model for better domain-specific analysis
   - See `TRAINING_GUIDE.md` for details

2. **Optimization**:
   - Adjust `max_new_tokens` in `model_config.py`
   - Tune `temperature` for consistency vs creativity
   - Profile VRAM usage for your specific GPU

3. **Customization**:
   - Modify prompts in `llm/prompts.py`
   - Adjust metric thresholds
   - Add custom analysis sections

4. **Production**:
   - Set up proper error handling
   - Add request rate limiting
   - Configure persistent session storage
   - Set up model caching strategy

## 🆘 Support

If you encounter issues:

1. Check backend console logs
2. Check browser console (F12)
3. Verify CUDA and GPU availability
4. Review `nvidia-smi` output
5. Check `temp_sessions/` directory exists
6. Verify all dependencies installed

For GPU/CUDA issues, refer to PyTorch documentation:
https://pytorch.org/get-started/locally/

---

**Note**: The Vision-LLM feature is GPU-intensive. Performance may vary based on your hardware. The 6GB VRAM requirement is a minimum; 8GB+ recommended for optimal performance.
