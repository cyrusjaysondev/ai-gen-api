#!/bin/bash
# =============================================================
# AI Video & Image Generation API - Full Setup Script
# Verified against live server: t6pgge1y1kl2qt (RTX 5090 32GB)
# Last updated: March 2026
#
# Runs BEFORE ComfyUI starts (called by template's /start.sh)
# Set in RunPod template:
#   download_ltx_23_22b_dev_fp8_29gb = true
#   SETUP_SCRIPT_URL = https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api/main/setup.sh
# =============================================================

LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

log "=========================================="
log "AI Generation API Setup Started"
log "Pod ID: $RUNPOD_POD_ID"
log "=========================================="

MODELS="/workspace/ComfyUI/models"
NODES="/workspace/ComfyUI/custom_nodes"

# ─────────────────────────────────────────────
# 1. Install pip dependencies
# ─────────────────────────────────────────────
log "[1/9] Installing pip dependencies..."
pip install -q fastapi uvicorn httpx websockets python-multipart
log "✅ Core API dependencies installed"

# ─────────────────────────────────────────────
# 2. Delete unused 23GB Gemma file (auto-downloaded by template)
#    We use the fp4_mixed version instead — saves 23GB disk space
# ─────────────────────────────────────────────
GEMMA_UNUSED="$MODELS/text_encoders/gemma_3_12B_it.safetensors"
if [ -f "$GEMMA_UNUSED" ]; then
    log "[2/9] Deleting unused gemma_3_12B_it.safetensors (23GB, not used by our API)..."
    rm "$GEMMA_UNUSED"
    log "✅ Freed 23GB"
else
    log "[2/9] ✅ Unused Gemma already removed or not present"
fi

# ─────────────────────────────────────────────
# 3. Download Gemma fp4_mixed text encoder (~8.8GB)
#    Used by LTX 2.3 workflow
# ─────────────────────────────────────────────
GEMMA_FP4="$MODELS/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
if [ ! -f "$GEMMA_FP4" ]; then
    log "[3/9] Downloading Gemma fp4_mixed (8.8GB)..."
    mkdir -p "$MODELS/text_encoders"
    wget -q --show-progress \
        -O "$GEMMA_FP4" \
        "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
    log "✅ Gemma fp4_mixed downloaded"
else
    log "[3/9] ✅ Gemma fp4_mixed already exists, skipping"
fi

# ─────────────────────────────────────────────
# 4. Download Wan 2.2 TI2V-5B diffusion model (~9.3GB)
#    IMPORTANT: goes in diffusion_models/ NOT checkpoints/
# ─────────────────────────────────────────────
WAN_MODEL="$MODELS/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
if [ ! -f "$WAN_MODEL" ]; then
    log "[4/9] Downloading Wan 2.2 TI2V-5B model (9.3GB)..."
    mkdir -p "$MODELS/diffusion_models"
    wget -q --show-progress \
        -O "$WAN_MODEL" \
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
    log "✅ Wan 2.2 model downloaded"
else
    log "[4/9] ✅ Wan 2.2 model already exists, skipping"
fi

# ─────────────────────────────────────────────
# 5. Download Wan 2.2 VAE (~1.3GB)
# ─────────────────────────────────────────────
WAN_VAE="$MODELS/vae/wan2.2_vae.safetensors"
if [ ! -f "$WAN_VAE" ]; then
    log "[5/9] Downloading Wan 2.2 VAE (1.3GB)..."
    mkdir -p "$MODELS/vae"
    wget -q --show-progress \
        -O "$WAN_VAE" \
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors"
    log "✅ Wan 2.2 VAE downloaded"
else
    log "[5/9] ✅ Wan 2.2 VAE already exists, skipping"
fi

# ─────────────────────────────────────────────
# 6. Download UMT5 text encoder for Wan 2.2 (~10.6GB)
# ─────────────────────────────────────────────
UMT5="$MODELS/text_encoders/umt5-xxl-enc-bf16.safetensors"
if [ ! -f "$UMT5" ]; then
    log "[6/9] Downloading UMT5 text encoder (10.6GB)..."
    wget -q --show-progress \
        -O "$UMT5" \
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"
    log "✅ UMT5 downloaded"
else
    log "[6/9] ✅ UMT5 already exists, skipping"
fi

# ─────────────────────────────────────────────
# 7. Install ComfyUI-WanVideoWrapper
# ─────────────────────────────────────────────
WAN_WRAPPER="$NODES/ComfyUI-WanVideoWrapper"
if [ ! -d "$WAN_WRAPPER" ]; then
    log "[7/9] Installing ComfyUI-WanVideoWrapper..."
    cd "$NODES"
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper
    cd ComfyUI-WanVideoWrapper && pip install -q -r requirements.txt
    log "✅ WanVideoWrapper installed"
else
    log "[7/9] ✅ WanVideoWrapper already exists, skipping"
fi

# ─────────────────────────────────────────────
# 8. Install ReActor face swap + models
# ─────────────────────────────────────────────

# 8a. Clone and install comfyui-reactor-node
REACTOR_DIR="$NODES/comfyui-reactor-node"
if [ ! -d "$REACTOR_DIR" ]; then
    log "[8/9] Installing ReActor face swap node..."
    cd "$NODES"
    git clone https://github.com/edwios/comfyui-reactor.git comfyui-reactor-node
    cd comfyui-reactor-node
    pip install -q -r requirements.txt
    pip install -q "numpy>=2.0.0,<3"
    log "✅ ReActor node installed"
else
    log "[8/9] ✅ ReActor already exists, skipping"
fi

# 8b. Disable NSFW check — we handle content moderation at app level
REACTOR_NODES="$NODES/comfyui-reactor-node/nodes.py"
if grep -q "nsfw_image" "$REACTOR_NODES" 2>/dev/null; then
    sed -i 's/if not sfw.nsfw_image(tmp_img, NSFWDET_MODEL_PATH):/if True: # NSFW check disabled/' "$REACTOR_NODES"
    log "✅ NSFW check disabled in ReActor"
fi

# 8c. Download inswapper face swap model (~500MB)
#     IMPORTANT: must go in insightface/ folder — ReActor looks there
mkdir -p "$MODELS/insightface"
INSWAPPER="$MODELS/insightface/inswapper_128.onnx"
if [ ! -f "$INSWAPPER" ]; then
    log "    Downloading inswapper_128.onnx (500MB)..."
    wget -q --show-progress \
        -O "$INSWAPPER" \
        "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"
    log "✅ inswapper_128.onnx downloaded"
else
    log "    ✅ inswapper_128.onnx already exists"
fi

# 8d. Download GFPGANv1.4 face restoration model (~300MB)
mkdir -p "$MODELS/facerestore_models"
GFPGAN="$MODELS/facerestore_models/GFPGANv1.4.pth"
if [ ! -f "$GFPGAN" ]; then
    log "    Downloading GFPGANv1.4.pth (300MB)..."
    wget -q --show-progress \
        -O "$GFPGAN" \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    log "✅ GFPGANv1.4.pth downloaded"
else
    log "    ✅ GFPGANv1.4.pth already exists"
fi

# ─────────────────────────────────────────────
# 9. Download main.py + start API watcher
# ─────────────────────────────────────────────
log "[9/9] Downloading API main.py from GitHub..."
mkdir -p /workspace/api
wget -q \
    -O /workspace/api/main.py \
    "https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api/main/api/main.py"

if [ ! -f "/workspace/api/main.py" ] || [ ! -s "/workspace/api/main.py" ]; then
    log "❌ ERROR: Failed to download main.py!"
    exit 1
fi
log "✅ main.py downloaded"

# Update pod ID in main.py
if [ ! -z "$RUNPOD_POD_ID" ]; then
    sed -i "s|t6pgge1y1kl2qt|$RUNPOD_POD_ID|g" /workspace/api/main.py
    log "✅ Pod ID updated to: $RUNPOD_POD_ID"
else
    log "⚠️  WARNING: RUNPOD_POD_ID not set"
fi

# Create watcher that starts API after ComfyUI is ready
# Also reinstalls pip packages on every start — they are lost on pod restart
cat > /workspace/start_api.sh << 'EOF'
#!/bin/bash
LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

# ── Reinstall pip packages (lost on every pod restart) ──
log "Reinstalling pip packages..."

# Core API packages
pip install -q fastapi uvicorn httpx websockets python-multipart

# InsightFace + ReActor deps — onnxruntime-gpu required for GPU face swap
pip install -q insightface onnx onnxruntime-gpu opencv-python gguf

# WanVideoWrapper requirements first
if [ -f "/workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt" ]; then
    pip install -q -r /workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt
fi

# ReActor requirements — installs numpy 1.26.3
if [ -f "/workspace/ComfyUI/custom_nodes/comfyui-reactor-node/requirements.txt" ]; then
    pip install -q -r /workspace/ComfyUI/custom_nodes/comfyui-reactor-node/requirements.txt
fi

# Override numpy LAST — must come after reactor to fix conflict
pip install -q "numpy>=2.0.0,<3"

log "✅ Pip packages reinstalled"

# ── Wait for ComfyUI ──
log "API watcher: waiting for ComfyUI..."
MAX_WAIT=300
WAITED=0
until curl -s http://localhost:8188/system_stats > /dev/null 2>&1; do
    sleep 3
    WAITED=$((WAITED + 3))
    if [ $WAITED -ge $MAX_WAIT ]; then
        log "❌ ComfyUI did not start within ${MAX_WAIT}s"
        exit 1
    fi
done
log "✅ ComfyUI ready after ${WAITED}s! Starting API..."
cd /workspace/api
python3 -m uvicorn main:app --host 0.0.0.0 --port 7860 >> /workspace/api.log 2>&1
EOF

chmod +x /workspace/start_api.sh

# Register start_api.sh to run on every pod restart via crontab
# This ensures it works even on templates that don't call SETUP_SCRIPT_URL again
(crontab -l 2>/dev/null | grep -v "start_api.sh"; echo "@reboot bash /workspace/start_api.sh") | crontab -
log "✅ start_api.sh registered in crontab for auto-restart"

nohup bash /workspace/start_api.sh > /dev/null 2>&1 &

log "=========================================="
log "✅ Setup Complete!"
log ""
log "Endpoints will be live after ComfyUI loads (~2 min):"
log "  Health: https://${RUNPOD_POD_ID}-7860.proxy.runpod.net/health"
log ""
log "Logs:"
log "  Setup:   tail -f /workspace/api_setup.log"
log "  ComfyUI: tail -f /workspace/comfyui_${RUNPOD_POD_ID}_nohup.log"
log "  API:     tail -f /workspace/api.log"
log "=========================================="
