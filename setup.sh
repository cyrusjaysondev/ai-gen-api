#!/bin/bash
LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }
log "=========================================="
log "AI Generation API Setup Started"
log "Pod ID: $RUNPOD_POD_ID"
log "=========================================="
MODELS="/workspace/ComfyUI/models"
NODES="/workspace/ComfyUI/custom_nodes"
log "[1/9] Installing pip dependencies..."
pip install -q fastapi uvicorn httpx websockets python-multipart
log "[2/9] Deleting unused gemma_3_12B_it.safetensors (23GB)..."
rm -f "$MODELS/text_encoders/gemma_3_12B_it.safetensors"
GEMMA_FP4="$MODELS/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
if [ ! -f "$GEMMA_FP4" ]; then
    log "[3/9] Downloading Gemma fp4_mixed (8.8GB)..."
    wget -q --show-progress -O "$GEMMA_FP4" "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
else
    log "[3/9] Gemma fp4_mixed already exists"
fi
WAN_MODEL="$MODELS/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
if [ ! -f "$WAN_MODEL" ]; then
    log "[4/9] Downloading Wan 2.2 model (9.3GB)..."
    mkdir -p "$MODELS/diffusion_models"
    wget -q --show-progress -O "$WAN_MODEL" "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
else
    log "[4/9] Wan 2.2 model already exists"
fi
WAN_VAE="$MODELS/vae/wan2.2_vae.safetensors"
if [ ! -f "$WAN_VAE" ]; then
    log "[5/9] Downloading Wan 2.2 VAE (1.3GB)..."
    mkdir -p "$MODELS/vae"
    wget -q --show-progress -O "$WAN_VAE" "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors"
else
    log "[5/9] Wan 2.2 VAE already exists"
fi
UMT5="$MODELS/text_encoders/umt5-xxl-enc-bf16.safetensors"
if [ ! -f "$UMT5" ]; then
    log "[6/9] Downloading UMT5 (10.6GB)..."
    wget -q --show-progress -O "$UMT5" "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"
else
    log "[6/9] UMT5 already exists"
fi
WAN_WRAPPER="$NODES/ComfyUI-WanVideoWrapper"
if [ ! -d "$WAN_WRAPPER" ]; then
    log "[7/9] Installing ComfyUI-WanVideoWrapper..."
    cd "$NODES" && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper
    cd ComfyUI-WanVideoWrapper && pip install -q -r requirements.txt
else
    log "[7/9] WanVideoWrapper already exists"
fi
REACTOR_DIR="$NODES/comfyui-reactor-node"
if [ ! -d "$REACTOR_DIR" ]; then
    log "[8/9] Installing ReActor..."
    cd "$NODES" && git clone https://github.com/edwios/comfyui-reactor.git comfyui-reactor-node
    cd comfyui-reactor-node && pip install -q -r requirements.txt
    pip install -q "numpy>=2.0.0,<3"
fi
REACTOR_NODES="$NODES/comfyui-reactor-node/nodes.py"
if grep -q "nsfw_image" "$REACTOR_NODES" 2>/dev/null; then
    sed -i 's/if not sfw.nsfw_image(tmp_img, NSFWDET_MODEL_PATH):/if True: # NSFW check disabled/' "$REACTOR_NODES"
fi
mkdir -p "$MODELS/insightface"
INSWAPPER="$MODELS/insightface/inswapper_128.onnx"
if [ ! -f "$INSWAPPER" ]; then
    log "    Downloading inswapper_128.onnx (500MB)..."
    wget -q --show-progress -O "$INSWAPPER" "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"
fi
mkdir -p "$MODELS/facerestore_models"
GFPGAN="$MODELS/facerestore_models/GFPGANv1.4.pth"
if [ ! -f "$GFPGAN" ]; then
    log "    Downloading GFPGANv1.4.pth (300MB)..."
    wget -q --show-progress -O "$GFPGAN" "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
fi
log "[9/9] Downloading main.py..."
mkdir -p /workspace/api
wget -q -O /workspace/api/main.py "https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api/main/api/main.py"
[ ! -z "$RUNPOD_POD_ID" ] && sed -i "s|t6pgge1y1kl2qt|$RUNPOD_POD_ID|g" /workspace/api/main.py
cat > /workspace/start_api.sh << 'EOF'
#!/bin/bash
LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }
log "Waiting for ComfyUI..."
MAX_WAIT=300; WAITED=0
until curl -s http://localhost:8188/system_stats > /dev/null 2>&1; do
    sleep 3; WAITED=$((WAITED + 3))
    [ $WAITED -ge $MAX_WAIT ] && log "ERROR: ComfyUI timeout" && exit 1
done
log "ComfyUI ready! Starting API..."
cd /workspace/api && python3 -m uvicorn main:app --host 0.0.0.0 --port 7860 >> /workspace/api.log 2>&1
EOF
chmod +x /workspace/start_api.sh
nohup bash /workspace/start_api.sh > /dev/null 2>&1 &
log "Setup complete! API starting after ComfyUI loads."
