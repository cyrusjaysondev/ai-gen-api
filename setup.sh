#!/bin/bash
# =============================================================
# AI Video & Image Generation API - Full Setup Script
# Supports: LTX 2.3, Wan 2.2, FLUX.2 Klein 9B, JuggernautXL,
#           Face swap (art-venture), BFS head swap, inpainting
#
# Runs BEFORE ComfyUI starts (called by template's /start.sh)
# Set in RunPod template:
#   download_ltx_23_22b_dev_fp8_29gb = true
#   SETUP_SCRIPT_URL = https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api/main/setup.sh
# =============================================================

# NOTE: no set -e — we want setup to continue even if a non-critical step fails

LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

log "=========================================="
log "AI Generation API Setup Started"
log "Pod ID: $RUNPOD_POD_ID"
log "=========================================="

MODELS="/workspace/ComfyUI/models"
NODES="/workspace/ComfyUI/custom_nodes"

# ─────────────────────────────────────────────
# 1. System dependencies
# ─────────────────────────────────────────────
log "[1/15] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq zstd > /dev/null 2>&1
log "  zstd installed"

# ─────────────────────────────────────────────
# 2. Install pip dependencies
# ─────────────────────────────────────────────
log "[2/15] Installing pip dependencies..."
pip install -q fastapi uvicorn httpx websockets python-multipart
pip install -q insightface onnx onnxruntime-gpu opencv-python gguf
pip install -q segment-anything
pip install -q "numpy>=2.0.0,<3"
log "  Core API + face swap + SAM dependencies installed"

# ─────────────────────────────────────────────
# 3. Delete unused 23GB Gemma file (auto-downloaded by template)
#    We use the fp4_mixed version instead
# ─────────────────────────────────────────────
GEMMA_UNUSED="$MODELS/text_encoders/gemma_3_12B_it.safetensors"
if [ -f "$GEMMA_UNUSED" ]; then
  log "[3/15] Deleting unused gemma_3_12B_it.safetensors (23GB)..."
  rm "$GEMMA_UNUSED"
  log "  Freed 23GB"
else
  log "[3/15] Unused Gemma already removed or not present"
fi

# ─────────────────────────────────────────────
# 4. LTX text encoder: Gemma fp4_mixed (~8.8GB)
# ─────────────────────────────────────────────
GEMMA_FP4="$MODELS/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
if [ ! -f "$GEMMA_FP4" ]; then
  log "[4/15] Downloading Gemma fp4_mixed (8.8GB)..."
  mkdir -p "$MODELS/text_encoders"
  wget -q --show-progress \
    -O "$GEMMA_FP4" \
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
  log "  Gemma fp4_mixed downloaded"
else
  log "[4/15] Gemma fp4_mixed already exists, skipping"
fi

# ─────────────────────────────────────────────
# 5. Wan 2.2 TI2V-5B diffusion model (~9.3GB)
# ─────────────────────────────────────────────
WAN_MODEL="$MODELS/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
if [ ! -f "$WAN_MODEL" ]; then
  log "[5/15] Downloading Wan 2.2 TI2V-5B (9.3GB)..."
  mkdir -p "$MODELS/diffusion_models"
  wget -q --show-progress \
    -O "$WAN_MODEL" \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
  log "  Wan 2.2 model downloaded"
else
  log "[5/15] Wan 2.2 model already exists, skipping"
fi

# ─────────────────────────────────────────────
# 6. Wan 2.2 VAE (~1.3GB) + UMT5 text encoder (~10.6GB)
# ─────────────────────────────────────────────
WAN_VAE="$MODELS/vae/wan2.2_vae.safetensors"
if [ ! -f "$WAN_VAE" ]; then
  log "[6/15] Downloading Wan 2.2 VAE (1.3GB)..."
  mkdir -p "$MODELS/vae"
  wget -q --show-progress \
    -O "$WAN_VAE" \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors"
  log "  Wan 2.2 VAE downloaded"
else
  log "[6/15] Wan 2.2 VAE already exists, skipping"
fi

UMT5="$MODELS/text_encoders/umt5-xxl-enc-bf16.safetensors"
if [ ! -f "$UMT5" ]; then
  log "[6/15] Downloading UMT5 text encoder (10.6GB)..."
  wget -q --show-progress \
    -O "$UMT5" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"
  log "  UMT5 downloaded"
else
  log "[6/15] UMT5 already exists, skipping"
fi

# ─────────────────────────────────────────────
# 7. FLUX.2 Klein 9B UNET (~18GB)
# ─────────────────────────────────────────────
FLUX_UNET="$MODELS/diffusion_models/flux2-klein-9b.safetensors"
if [ ! -f "$FLUX_UNET" ]; then
  log "[7/15] Downloading FLUX.2 Klein 9B UNET (18GB)..."
  mkdir -p "$MODELS/diffusion_models"
  wget -q --show-progress \
    -O "$FLUX_UNET" \
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/flux2-klein-9b.safetensors"
  log "  FLUX Klein 9B downloaded"
else
  log "[7/15] FLUX Klein 9B already exists, skipping"
fi
# Symlink with dashed name (workflow references flux-2-klein-9b)
ln -sf "$FLUX_UNET" "$MODELS/diffusion_models/flux-2-klein-9b.safetensors"

# ─────────────────────────────────────────────
# 8. FLUX.2 VAE (~336MB) + Qwen 3 8B text encoder (~8.7GB)
# ─────────────────────────────────────────────
FLUX_VAE="$MODELS/vae/flux2-vae.safetensors"
if [ ! -f "$FLUX_VAE" ]; then
  log "[8/15] Downloading FLUX.2 VAE (336MB)..."
  wget -q --show-progress \
    -O "$FLUX_VAE" \
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors"
  log "  FLUX VAE downloaded"
else
  log "[8/15] FLUX VAE already exists, skipping"
fi

QWEN="$MODELS/text_encoders/qwen_3_8b_fp8mixed.safetensors"
if [ ! -f "$QWEN" ]; then
  log "[8/15] Downloading Qwen 3 8B text encoder (8.7GB)..."
  wget -q --show-progress \
    -O "$QWEN" \
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"
  log "  Qwen 3 8B downloaded"
else
  log "[8/15] Qwen 3 8B already exists, skipping"
fi

# ─────────────────────────────────────────────
# 9. BFS Head Swap LoRAs for FLUX Klein
# ─────────────────────────────────────────────
mkdir -p "$MODELS/loras"

BFS_HEAD="$MODELS/loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors"
if [ ! -f "$BFS_HEAD" ]; then
  log "[9/15] Downloading BFS head swap LoRA (663MB)..."
  wget -q --show-progress \
    -O "$BFS_HEAD" \
    "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors"
  log "  BFS head LoRA downloaded"
else
  log "[9/15] BFS head LoRA already exists, skipping"
fi

BFS_FACE="$MODELS/loras/bfs-face-swap-flux2-klein-9b.safetensors"
if [ ! -f "$BFS_FACE" ]; then
  log "[9/15] Downloading BFS face swap LoRA (331MB)..."
  wget -q --show-progress \
    -O "$BFS_FACE" \
    "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs-face-swap-flux2-klein-9b.safetensors"
  log "  BFS face LoRA downloaded"
else
  log "[9/15] BFS face LoRA already exists, skipping"
fi

# ─────────────────────────────────────────────
# 10. JuggernautXL SDXL checkpoint (~7.1GB)
#     Used for t2i, i2i, inpainting, head-swap
# ─────────────────────────────────────────────
JUGGER="$MODELS/checkpoints/juggernautXL_v9Rdphoto2Lightning.safetensors"
if [ ! -f "$JUGGER" ]; then
  log "[10/15] Downloading JuggernautXL (7.1GB)..."
  mkdir -p "$MODELS/checkpoints"
  wget -q --show-progress \
    -O "$JUGGER" \
    "https://huggingface.co/AiWise/Juggernaut-XL-V9-GE-RDPhoto2-Lightning_4S/resolve/main/juggernautXL_v9Rdphoto2Lightning.safetensors"
  log "  JuggernautXL downloaded"
else
  log "[10/15] JuggernautXL already exists, skipping"
fi

# ─────────────────────────────────────────────
# 11. Face swap + detection models
# ─────────────────────────────────────────────
log "[11/15] Checking face swap models..."

# inswapper_128 (default swap model)
mkdir -p "$MODELS/insightface"
INSWAPPER="$MODELS/insightface/inswapper_128.onnx"
if [ ! -f "$INSWAPPER" ]; then
  log "  Downloading inswapper_128.onnx (500MB)..."
  wget -q --show-progress \
    -O "$INSWAPPER" \
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"
  log "  inswapper_128.onnx downloaded"
fi

# Hyperswap models (alternative swap models exposed in API)
mkdir -p "$MODELS/hyperswap"
for hs_model in hyperswap_1a_256.onnx hyperswap_1b_256.onnx hyperswap_1c_256.onnx; do
  if [ ! -f "$MODELS/hyperswap/$hs_model" ]; then
    log "  Downloading $hs_model..."
    wget -q --show-progress \
      -O "$MODELS/hyperswap/$hs_model" \
      "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/hyperswap/$hs_model"
    log "  $hs_model downloaded"
  fi
done

# GFPGANv1.4 face restoration
mkdir -p "$MODELS/facerestore_models"
GFPGAN="$MODELS/facerestore_models/GFPGANv1.4.pth"
if [ ! -f "$GFPGAN" ]; then
  log "  Downloading GFPGANv1.4.pth (300MB)..."
  wget -q --show-progress \
    -O "$GFPGAN" \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
  log "  GFPGANv1.4.pth downloaded"
fi

# Face detection models (used by CodeFormer/GFPGAN)
mkdir -p "$MODELS/facedetection"
if [ ! -f "$MODELS/facedetection/detection_Resnet50_Final.pth" ]; then
  log "  Downloading face detection models..."
  wget -q -O "$MODELS/facedetection/detection_Resnet50_Final.pth" \
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
  wget -q -O "$MODELS/facedetection/parsing_parsenet.pth" \
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
  log "  Face detection models downloaded"
fi

# SAM model (for inpainting auto-segmentation)
mkdir -p "$MODELS/sams"
SAM="$MODELS/sams/sam_vit_b_01ec64.pth"
if [ ! -f "$SAM" ]; then
  log "  Downloading SAM vit_b (375MB)..."
  wget -q --show-progress \
    -O "$SAM" \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
  log "  SAM downloaded"
fi

log "  Face swap models complete"

# ─────────────────────────────────────────────
# 12. Install custom nodes
# ─────────────────────────────────────────────
log "[12/15] Checking custom nodes..."

# WanVideoWrapper (Wan 2.2 video generation)
if [ ! -d "$NODES/ComfyUI-WanVideoWrapper" ]; then
  log "  Installing ComfyUI-WanVideoWrapper..."
  cd "$NODES"
  git clone -q https://github.com/kijai/ComfyUI-WanVideoWrapper
  cd ComfyUI-WanVideoWrapper && pip install -q -r requirements.txt
  log "  WanVideoWrapper installed"
else
  log "  WanVideoWrapper already exists"
fi

# comfyui-art-venture (provides ReActorFaceSwap node)
if [ ! -d "$NODES/comfyui-art-venture" ]; then
  log "  Installing comfyui-art-venture..."
  cd "$NODES"
  git clone -q https://github.com/sipherxyz/comfyui-art-venture
  cd comfyui-art-venture && pip install -q -r requirements.txt
  log "  comfyui-art-venture installed"
else
  log "  comfyui-art-venture already exists"
fi

# ComfyUI-Impact-Pack (SAMLoader, CLIPSegDetectorProvider, ImpactSimpleDetectorSEGS — used by /inpaint)
if [ ! -d "$NODES/ComfyUI-Impact-Pack" ]; then
  log "  Installing ComfyUI-Impact-Pack..."
  cd "$NODES"
  git clone -q https://github.com/ltdrdata/ComfyUI-Impact-Pack
  cd ComfyUI-Impact-Pack && pip install -q -r requirements.txt
  # Impact-Pack has an install script for submodules
  if [ -f "install.py" ]; then
    python install.py || log "  Warning: Impact-Pack install.py had errors (may still work)"
  fi
  log "  Impact-Pack installed"
else
  log "  Impact-Pack already exists"
fi

# ComfyUI-CLIPSeg (CLIPSeg detector for auto-mask inpainting)
# NOTE: Do NOT install its requirements.txt — it pins torch==2.0.0+cu118 which breaks everything.
# Its actual deps (transformers, etc.) are already installed by other packages.
if [ ! -d "$NODES/ComfyUI-CLIPSeg" ]; then
  log "  Installing ComfyUI-CLIPSeg..."
  cd "$NODES"
  git clone -q https://github.com/biegert/ComfyUI-CLIPSeg
  log "  CLIPSeg installed (deps already satisfied by other packages)"
else
  log "  CLIPSeg already exists"
fi

# ComfyUI-VideoHelperSuite (VHS_VideoCombine — used by Wan video output)
if [ ! -d "$NODES/ComfyUI-VideoHelperSuite" ]; then
  log "  Installing ComfyUI-VideoHelperSuite..."
  cd "$NODES"
  git clone -q https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
  cd ComfyUI-VideoHelperSuite && pip install -q -r requirements.txt
  log "  VideoHelperSuite installed"
else
  log "  VideoHelperSuite already exists"
fi

# ComfyUI-LTXVideo (LTX audio/video nodes)
if [ ! -d "$NODES/ComfyUI-LTXVideo" ]; then
  log "  Installing ComfyUI-LTXVideo..."
  cd "$NODES"
  git clone -q https://github.com/Lightricks/ComfyUI-LTXVideo
  if [ -f "$NODES/ComfyUI-LTXVideo/requirements.txt" ]; then
    cd ComfyUI-LTXVideo && pip install -q -r requirements.txt
  fi
  log "  LTXVideo installed"
else
  log "  LTXVideo already exists"
fi

# LanPaint (FLUX head swap — LanPaint_KSampler, ReferenceLatent, EmptyFlux2LatentImage)
if [ ! -d "$NODES/LanPaint" ]; then
  log "  Installing LanPaint custom node..."
  cd "$NODES"
  git clone -q https://github.com/scraed/LanPaint
  if [ -f "$NODES/LanPaint/requirements.txt" ]; then
    cd LanPaint && pip install -q -r requirements.txt
  fi
  log "  LanPaint installed"
else
  log "  LanPaint already exists"
fi

# Final numpy override (some nodes pin old versions)
pip install -q "numpy>=2.0.0,<3"

log "  Custom nodes complete"

# ─────────────────────────────────────────────
# 13. Install Ollama (prompt enhancement)
# ─────────────────────────────────────────────
log "[13/15] Setting up Ollama..."
if ! command -v ollama &> /dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh 2>&1 | tail -1
  log "  Ollama binary installed"
else
  log "  Ollama already installed"
fi

# Store models on volume (persists across restarts)
export OLLAMA_MODELS="/workspace/ollama_models"
mkdir -p "$OLLAMA_MODELS"

# Pull model in background (takes ~2min, don't block setup)
(
  export OLLAMA_MODELS="/workspace/ollama_models"
  ollama serve &
  OLLAMA_PID=$!
  sleep 5
  ollama pull llama3.2:3b >> $LOG 2>&1
  log "  Ollama llama3.2:3b model ready"
  kill $OLLAMA_PID 2>/dev/null
) &

# ─────────────────────────────────────────────
# 14. Download main.py + start API watcher
# ─────────────────────────────────────────────
log "[14/15] Downloading API main.py from GitHub..."
mkdir -p /workspace/api
wget -q \
  -O /workspace/api/main.py \
  "https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api/main/api/main.py"

if [ ! -f "/workspace/api/main.py" ] || [ ! -s "/workspace/api/main.py" ]; then
  log "WARNING: Failed to download main.py from GitHub. Will retry on next restart."
  log "You can manually download it or paste it into /workspace/api/main.py"
fi
log "  main.py downloaded"

# Update pod ID in main.py — use a placeholder in the repo version
# The repo main.py should contain RUNPOD_POD_ID_PLACEHOLDER or the old pod ID
if [ ! -z "$RUNPOD_POD_ID" ]; then
  # Replace any known pod ID pattern (old IDs or placeholder)
  sed -i "s|RUNPOD_POD_ID_PLACEHOLDER|$RUNPOD_POD_ID|g" /workspace/api/main.py
  sed -i "s|t6pgge1y1kl2qt|$RUNPOD_POD_ID|g" /workspace/api/main.py
  sed -i "s|8mj50saxmbkhdz|$RUNPOD_POD_ID|g" /workspace/api/main.py
  log "  Pod ID updated to: $RUNPOD_POD_ID"
else
  log "  WARNING: RUNPOD_POD_ID not set — URLs will be wrong!"
fi

# ─────────────────────────────────────────────
# 15. Create start_api.sh (runs on every pod restart)
# ─────────────────────────────────────────────
log "[15/15] Creating start_api.sh..."

cat > /workspace/start_api.sh << 'STARTEOF'
#!/bin/bash
LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

# ── System deps (lost on restart) ──
apt-get update -qq && apt-get install -y -qq zstd > /dev/null 2>&1

# ── Reinstall pip packages (lost on restart) ──
log "Reinstalling pip packages..."

# Core API
pip install -q fastapi uvicorn httpx websockets python-multipart

# Face swap deps
pip install -q insightface onnx onnxruntime-gpu opencv-python gguf
pip install -q segment-anything

# Custom node requirements
# NOTE: ComfyUI-CLIPSeg excluded — its requirements.txt pins torch==2.0.0+cu118
for node_dir in ComfyUI-WanVideoWrapper comfyui-art-venture ComfyUI-Impact-Pack ComfyUI-VideoHelperSuite ComfyUI-LTXVideo LanPaint; do
  REQ="/workspace/ComfyUI/custom_nodes/$node_dir/requirements.txt"
  if [ -f "$REQ" ]; then
    pip install -q -r "$REQ"
  fi
done

# Override numpy LAST (some nodes pin old versions)
pip install -q "numpy>=2.0.0,<3"

log "Pip packages reinstalled"

# ── Install Ollama (lost on restart) ──
if ! command -v ollama &> /dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh 2>&1 | tail -1
fi
export OLLAMA_MODELS="/workspace/ollama_models"
ollama serve &
OLLAMA_PID=$!
sleep 5
ollama pull llama3.2:3b >> $LOG 2>&1
log "Ollama ready"

# ── Update pod ID if needed ──
if [ ! -z "$RUNPOD_POD_ID" ]; then
  sed -i "s|RUNPOD_POD_ID_PLACEHOLDER|$RUNPOD_POD_ID|g" /workspace/api/main.py
  sed -i "s|t6pgge1y1kl2qt|$RUNPOD_POD_ID|g" /workspace/api/main.py
  sed -i "s|8mj50saxmbkhdz|$RUNPOD_POD_ID|g" /workspace/api/main.py
fi

# ── Wait for ComfyUI ──
log "API watcher: waiting for ComfyUI..."
MAX_WAIT=300
WAITED=0
until curl -s http://localhost:8188/system_stats > /dev/null 2>&1; do
  sleep 3
  WAITED=$((WAITED + 3))
  if [ $WAITED -ge $MAX_WAIT ]; then
    log "ComfyUI did not start within ${MAX_WAIT}s"
    exit 1
  fi
done
log "ComfyUI ready after ${WAITED}s! Starting API..."

cd /workspace/api || exit 1
OLLAMA_MODELS="/workspace/ollama_models" /opt/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 7860 >> /workspace/api.log 2>&1
STARTEOF

chmod +x /workspace/start_api.sh

# Register for auto-start on pod restart
(crontab -l 2>/dev/null | grep -v "start_api.sh"; echo "@reboot bash /workspace/start_api.sh") | crontab -
log "  start_api.sh registered in crontab"

nohup bash /workspace/start_api.sh > /dev/null 2>&1 &

log "=========================================="
log "Setup Complete!"
log ""
log "Endpoints will be live after ComfyUI loads (~2 min):"
log "  Health: https://${RUNPOD_POD_ID}-7860.proxy.runpod.net/health"
log ""
log "Logs:"
log "  Setup:   tail -f /workspace/api_setup.log"
log "  API:     tail -f /workspace/api.log"
log "=========================================="
