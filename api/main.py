import uuid, json, httpx, os
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import websockets, asyncio

async def enhance_prompt_with_ollama(prompt: str, media_type: str = "image") -> str:
    """Use local Llama 3.2 3B via Ollama to enhance prompts. Free, no API cost."""
    try:
        import httpx
        if media_type == "video":
            instruction = "Enhance this video generation prompt into a detailed, cinematic description with camera movement, lighting, and motion details. Return only the enhanced prompt, nothing else."
        else:
            instruction = "Enhance this image generation prompt into a detailed, cinematic, photorealistic description with lighting, camera, and composition details. Return only the enhanced prompt, nothing else."
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": f"{instruction} Original: {prompt}",
                    "stream": False
                }
            )
            data = response.json()
            enhanced = data.get("response", "").strip()
            return enhanced if enhanced else prompt
    except Exception as e:
        print(f"Ollama enhancement failed: {e}")
        return prompt


app = FastAPI(title="LTX 2.3 Video API")
COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/workspace/ComfyUI/output")
PUBLIC_BASE_URL = "https://t6pgge1y1kl2qt-8888.proxy.runpod.net"

# In-memory job store
jobs = {}

def seconds_to_frames(seconds: int) -> int:
    frames = seconds * 25
    frames = ((frames - 1) // 8) * 8 + 1
    return max(9, frames)

class T2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = "worst quality, low quality, lowres, blurry, pixelated, jpeg artifacts, compression artifacts, noisy, grainy, unclear details, low contrast, low resolution, bad art, cartoon, anime, illustration, painting, sketch, drawing, cgi, render, 3d, comic, manga, watercolor, oil painting, digital art, concept art, artstation, octane render, cinema 4d, unreal engine, 2d, flat art, watermark, signature, text, logo, username, artist name, copyright, bad anatomy, bad proportions, deformed, disfigured, malformed, mutated, extra limbs, extra arms, extra legs, missing arms, missing legs, floating limbs, disconnected limbs, amputated, bad hands, poorly drawn hands, mutated hands, extra hands, missing hands, fused fingers, extra fingers, missing fingers, too many fingers, extra digits, fewer digits, long fingers, short fingers, malformed hands, bad face, poorly drawn face, cloned face, fused face, extra eyes, bad eyes, ugly eyes, deformed eyes, deformed pupils, deformed iris, cross-eyed, wall eye, asymmetrical face, uneven eyes, misaligned eyes, oversized eyes, tiny eyes, long neck, short neck, extra heads, multiple heads, multiple faces, bad feet, poorly drawn feet, extra feet, missing feet, unnatural pose, stiff pose, rigid pose, awkward pose, plastic skin, waxy skin, rubber skin, shiny skin, oily skin, unnatural skin tone, orange skin, gray skin, green skin, mannequin, doll, puppet, fake, artificial, fabric artifacts, wrinkled texture, unrealistic texture, bad cloth, distorted cloth, melting cloth, wrong material, unrealistic material, bad texture, bad background, distorted background, background inconsistency, bad architecture, distorted buildings, broken perspective, floating objects, impossible physics, unrealistic environment, wrong scale, disproportionate objects, overexposed, underexposed, washed out, oversaturated, desaturated, harsh lighting, flat lighting, bad lighting, unnatural lighting, color bleeding, chromatic aberration, color banding, monochrome when not intended, wrong colors, inconsistent motion, jittery, stuttering, flickering, frame drops, temporal inconsistency, ghosting, video compression artifacts, low framerate, choppy, freezing, looping artifacts, morphing artifacts, identity change, face distortion between frames, motion blur, out of focus, duplicate, clone, tiling, collage, split screen, vhs, old film, film grain, vintage, retro, lens flare, static, glitch, corrupted, broken, ugly, gross, creepy, disturbing"
    width: int = 768
    height: int = 512
    seconds: int = 5
    seed: int = -1
    steps: int = 30
    cfg: float = 1.2
    enhance_prompt: bool = True
    audio: bool = True
    quality: str = "high"

def get_workflow(prompt, negative_prompt, width, height, length, seed, image_filename=None, cfg=1.0, steps=20, audio=True, use_lora=True):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}},
        "2": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}},
        "3": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors", "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors", "device": "default"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "ltx-2.3-22b-distilled-lora-384.safetensors", "strength_model": 1.0 if use_lora else 0.0, "model": ["1", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["3", 0]}},
        "7": {"class_type": "LTXVConditioning", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 25.0}},
        "8": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height, "length": length, "batch_size": 1}},
        "9": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": length, "frame_rate": 25, "batch_size": 1, "audio_vae": ["2", 0]}},
        "10": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["8", 0], "audio_latent": ["9", 0]}},
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_cfg_pp"}},
        "14": {"class_type": "LTXVScheduler", "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1, "latent": ["10", 0]}},
        "15": {"class_type": "CFGGuider", "inputs": {"cfg": cfg, "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1]}},
        "16": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["12", 0], "guider": ["15", 0], "sampler": ["13", 0], "sigmas": ["14", 0], "latent_image": ["10", 0]}},
        "17": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["16", 0]}},
        "18": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["17", 0], "vae": ["1", 2], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}},
        "19": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["17", 1], "audio_vae": ["2", 0]}},
        "20": {"class_type": "CreateVideo", "inputs": {"images": ["18", 0], **({"audio": ["19", 0]} if audio else {}), "fps": 24.0}},
        "21": {"class_type": "SaveVideo", "inputs": {"video": ["20", 0], "filename_prefix": f"video/output_{seed}", "format": "auto", "codec": "auto"}}
    }
    if image_filename:
        wf["22"] = {"class_type": "LoadImage", "inputs": {"image": image_filename}}
        wf["8"] = {"class_type": "LTXVImgToVideo", "inputs": {
            "positive": ["5", 0], "negative": ["6", 0],
            "vae": ["1", 2], "image": ["22", 0],
            "width": width, "height": height, "length": length,
            "batch_size": 1, "strength": 1.0
        }}
        wf["10"]["inputs"]["video_latent"] = ["8", 2]
        wf["15"]["inputs"]["positive"] = ["8", 0]
        wf["15"]["inputs"]["negative"] = ["8", 1]
    return wf

async def run_job(job_id: str, workflow: dict, image_path: str = None):
    jobs[job_id] = {**jobs.get(job_id, {}), "status": "processing", "workflow": workflow, "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        client_id = str(uuid.uuid4())
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id})
            if resp.status_code != 200:
                jobs[job_id] = {"status": "failed", "error": resp.text}
                return
            prompt_id = resp.json()["prompt_id"]

        ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
        async with websockets.connect(ws_url) as ws:
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") == "executing":
                    data = msg.get("data", {})
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        break

        async with httpx.AsyncClient() as client:
            history = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
            job_data = history.json().get(prompt_id, {})
            status = job_data.get("status", {}).get("status_str", "")
            if status == "error":
                messages = job_data.get("status", {}).get("messages", [])
                for m in messages:
                    if m[0] == "execution_error":
                        jobs[job_id] = {"status": "failed", "error": m[1].get("exception_message")}
                        return
            outputs = job_data.get("outputs", {})

        for node_output in outputs.values():
            for key in ["videos", "gifs", "images"]:
                if key in node_output:
                    item = node_output[key][0]
                    filename = item["filename"]
                    subfolder = item.get("subfolder", "")
                    path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                    if path.exists():
                        # Determine if image or video based on extension
                        ext = Path(filename).suffix.lower()
                        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
                            url = f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/image/{filename}"
                        else:
                            url = f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/video/{filename}"
                        completed_at = datetime.now(timezone.utc)
                        created_at_str = jobs[job_id].get("created_at")
                        duration_seconds = None
                        if created_at_str:
                            started = datetime.fromisoformat(created_at_str)
                            duration_seconds = round((completed_at - started).total_seconds(), 1)
                        jobs[job_id] = {"status": "completed", "url": url, "filename": filename, "completed_at": completed_at.isoformat(), "duration_seconds": duration_seconds}
                        if image_path:
                            Path(image_path).unlink(missing_ok=True)
                        return

        jobs[job_id] = {"status": "failed", "error": "No output found"}
    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e), "failed_at": datetime.now(timezone.utc).isoformat()}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.post("/t2v")
async def text_to_video(req: T2VRequest, background_tasks: BackgroundTasks):
    seed = req.seed if req.seed != -1 else uuid.uuid4().int % 2**32
    length = seconds_to_frames(req.seconds)

    # Quality presets
    if req.quality == "fast":
        steps, cfg, use_lora = 8, 1.0, True
    elif req.quality == "balanced":
        steps, cfg, use_lora = 20, 1.0, True
    elif req.quality == "high":
        steps, cfg, use_lora = 30, 1.0, False  # no distilled LoRA
    else:
        steps, cfg, use_lora = req.steps, req.cfg, True

    if req.enhance_prompt:
        prompt = await enhance_prompt_with_ollama(req.prompt, "video")
    else:
        prompt = req.prompt
    workflow = get_workflow(prompt, req.negative_prompt, req.width, req.height, length, seed, cfg=cfg, steps=steps, audio=req.audio, use_lora=use_lora)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow)
    return {"job_id": job_id, "status": "queued", "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"}

@app.post("/i2v/upload")
async def image_to_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form("worst quality, low quality, lowres, blurry, pixelated, jpeg artifacts, compression artifacts, noisy, grainy, unclear details, low contrast, low resolution, bad art, cartoon, anime, illustration, painting, sketch, drawing, cgi, render, 3d, comic, manga, watercolor, oil painting, digital art, concept art, artstation, octane render, cinema 4d, unreal engine, 2d, flat art, watermark, signature, text, logo, username, artist name, copyright, bad anatomy, bad proportions, deformed, disfigured, malformed, mutated, extra limbs, extra arms, extra legs, missing arms, missing legs, floating limbs, disconnected limbs, amputated, bad hands, poorly drawn hands, mutated hands, extra hands, missing hands, fused fingers, extra fingers, missing fingers, too many fingers, extra digits, fewer digits, long fingers, short fingers, malformed hands, bad face, poorly drawn face, cloned face, fused face, extra eyes, bad eyes, ugly eyes, deformed eyes, deformed pupils, deformed iris, cross-eyed, wall eye, asymmetrical face, uneven eyes, misaligned eyes, oversized eyes, tiny eyes, long neck, short neck, extra heads, multiple heads, multiple faces, bad feet, poorly drawn feet, extra feet, missing feet, unnatural pose, stiff pose, rigid pose, awkward pose, plastic skin, waxy skin, rubber skin, shiny skin, oily skin, unnatural skin tone, orange skin, gray skin, green skin, mannequin, doll, puppet, fake, artificial, fabric artifacts, wrinkled texture, unrealistic texture, bad cloth, distorted cloth, melting cloth, wrong material, unrealistic material, bad texture, bad background, distorted background, background inconsistency, bad architecture, distorted buildings, broken perspective, floating objects, impossible physics, unrealistic environment, wrong scale, disproportionate objects, overexposed, underexposed, washed out, oversaturated, desaturated, harsh lighting, flat lighting, bad lighting, unnatural lighting, color bleeding, chromatic aberration, color banding, monochrome when not intended, wrong colors, inconsistent motion, jittery, stuttering, flickering, frame drops, temporal inconsistency, ghosting, video compression artifacts, low framerate, choppy, freezing, looping artifacts, morphing artifacts, identity change, face distortion between frames, motion blur, out of focus, duplicate, clone, tiling, collage, split screen, vhs, old film, film grain, vintage, retro, lens flare, static, glitch, corrupted, broken, ugly, gross, creepy, disturbing"),
    width: int = Form(544), height: int = Form(960),
    seconds: int = Form(5), seed: int = Form(-1),
    cfg: float = Form(1.5), steps: int = Form(8), audio: bool = Form(True)
):
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    length = seconds_to_frames(seconds)
    image_filename = f"input_{uuid.uuid4().hex}.png"
    image_path = str(Path("/workspace/ComfyUI/input") / image_filename)
    Path(image_path).write_bytes(await file.read())
    workflow = get_workflow(prompt, negative_prompt, width, height, length, seed, image_filename, cfg=cfg, steps=steps, audio=audio)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, image_path)
    return {"job_id": job_id, "status": "queued", "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"}

@app.get("/queue")
async def get_queue():
    """Show only queued and processing jobs with count."""
    active = {jid: info for jid, info in jobs.items() 
              if info.get("status") in ["queued", "processing"]}
    return {
        "count": len(active),
        "jobs": [{"job_id": jid, "status": info["status"]} 
                 for jid, info in active.items()]
    }

@app.get("/jobs")
async def get_all_jobs():
    """Show all jobs and their statuses."""
    return {
        "total": len(jobs),
        "summary": {
            "queued": sum(1 for j in jobs.values() if j.get("status") == "queued"),
            "processing": sum(1 for j in jobs.values() if j.get("status") == "processing"),
            "completed": sum(1 for j in jobs.values() if j.get("status") == "completed"),
            "failed": sum(1 for j in jobs.values() if j.get("status") == "failed"),
        },
        "jobs": [{"job_id": jid, **info} for jid, info in jobs.items()]
    }

@app.get("/video/{filename}")
async def serve_video(filename: str):
    for path in [OUTPUT_DIR / "video" / filename, OUTPUT_DIR / filename]:
        if path.exists():
            return FileResponse(str(path), media_type="video/mp4", filename=filename)
    raise HTTPException(404, f"Not found: {filename}")

@app.delete("/video/{filename}")
async def delete_video(filename: str):
    """Delete a generated video file."""
    deleted = []
    not_found = []
    for path in [OUTPUT_DIR / "video" / filename, OUTPUT_DIR / filename]:
        if path.exists():
            path.unlink()
            deleted.append(str(path))
        else:
            not_found.append(str(path))
    if deleted:
        # Also remove from jobs store
        for job_id, info in list(jobs.items()):
            if info.get("filename") == filename:
                del jobs[job_id]
        return {"status": "deleted", "filename": filename}
    raise HTTPException(404, f"File not found: {filename}")

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated video file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    filename = job.get("filename")
    result = {"job_id": job_id, "deleted": True}
    if filename:
        for path in [OUTPUT_DIR / "video" / filename, OUTPUT_DIR / filename]:
            if path.exists():
                path.unlink()
                result["file_deleted"] = filename
    del jobs[job_id]
    return result

@app.delete("/jobs")
async def delete_all_jobs(completed_only: bool = True):
    """Delete all jobs. Pass completed_only=false to delete everything."""
    deleted_jobs = 0
    deleted_files = 0
    for job_id in list(jobs.keys()):
        job = jobs[job_id]
        if completed_only and job.get("status") != "completed":
            continue
        filename = job.get("filename")
        if filename:
            for path in [OUTPUT_DIR / "video" / filename, OUTPUT_DIR / filename]:
                if path.exists():
                    path.unlink()
                    deleted_files += 1
        del jobs[job_id]
        deleted_jobs += 1
    return {"deleted_jobs": deleted_jobs, "deleted_files": deleted_files}

@app.delete("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued or processing job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.get("status") == "completed":
        raise HTTPException(400, "Job already completed, use DELETE /jobs/{job_id} to remove it")
    if job.get("status") == "failed":
        raise HTTPException(400, "Job already failed")
    
    # Cancel in ComfyUI queue too
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{COMFYUI_URL}/queue", json={"delete": [job_id]})
    except:
        pass
    
    jobs[job_id] = {"status": "cancelled"}
    return {"job_id": job_id, "status": "cancelled"}

@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    """Retry a failed or cancelled job using the same workflow."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.get("status") not in ["failed", "cancelled"]:
        raise HTTPException(400, f"Can only retry failed or cancelled jobs. Current status: {job.get('status')}")
    if "workflow" not in job:
        raise HTTPException(400, "No workflow stored for this job, please submit a new request")
    
    new_job_id = str(uuid.uuid4())
    jobs[new_job_id] = {"status": "queued"}
    background_tasks.add_task(run_job, new_job_id, job["workflow"])
    return {
        "new_job_id": new_job_id,
        "original_job_id": job_id,
        "status": "queued",
        "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{new_job_id}"
    }

@app.get("/videos")
async def list_videos():
    """List all video files stored on disk."""
    video_dir = OUTPUT_DIR / "video"
    if not video_dir.exists():
        return {"total": 0, "videos": []}
    
    videos = []
    for f in sorted(video_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        videos.append({
            "filename": f.name,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/video/{f.name}",
            "created_at": stat.st_mtime
        })
    return {"total": len(videos), "videos": videos}


# ─────────────────────────────────────────────
# Wan 2.2 TI2V-5B Workflow
# ─────────────────────────────────────────────

class WanT2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = "worst quality, low quality, lowres, blurry, pixelated, jpeg artifacts, compression artifacts, noisy, grainy, unclear details, low contrast, low resolution, bad art, cartoon, anime, illustration, painting, sketch, drawing, cgi, render, 3d, comic, manga, watercolor, oil painting, digital art, concept art, artstation, octane render, cinema 4d, unreal engine, 2d, flat art, watermark, signature, text, logo, username, artist name, copyright, bad anatomy, bad proportions, deformed, disfigured, malformed, mutated, extra limbs, extra arms, extra legs, missing arms, missing legs, floating limbs, disconnected limbs, amputated, bad hands, poorly drawn hands, mutated hands, extra hands, missing hands, fused fingers, extra fingers, missing fingers, too many fingers, extra digits, fewer digits, long fingers, short fingers, malformed hands, bad face, poorly drawn face, cloned face, fused face, extra eyes, bad eyes, ugly eyes, deformed eyes, deformed pupils, deformed iris, cross-eyed, wall eye, asymmetrical face, uneven eyes, misaligned eyes, oversized eyes, tiny eyes, long neck, short neck, extra heads, multiple heads, multiple faces, bad feet, poorly drawn feet, extra feet, missing feet, unnatural pose, stiff pose, rigid pose, awkward pose, plastic skin, waxy skin, rubber skin, shiny skin, oily skin, unnatural skin tone, orange skin, gray skin, green skin, mannequin, doll, puppet, fake, artificial, fabric artifacts, wrinkled texture, unrealistic texture, bad cloth, distorted cloth, melting cloth, wrong material, unrealistic material, bad texture, bad background, distorted background, background inconsistency, bad architecture, distorted buildings, broken perspective, floating objects, impossible physics, unrealistic environment, wrong scale, disproportionate objects, overexposed, underexposed, washed out, oversaturated, desaturated, harsh lighting, flat lighting, bad lighting, unnatural lighting, color bleeding, chromatic aberration, color banding, monochrome when not intended, wrong colors, inconsistent motion, jittery, stuttering, flickering, frame drops, temporal inconsistency, ghosting, video compression artifacts, low framerate, choppy, freezing, looping artifacts, morphing artifacts, identity change, face distortion between frames, motion blur, out of focus, duplicate, clone, tiling, collage, split screen, vhs, old film, film grain, vintage, retro, lens flare, static, glitch, corrupted, broken, ugly, gross, creepy, disturbing"
    width: int = 832
    height: int = 480
    seconds: int = 5
    seed: int = -1
    steps: int = 30
    cfg: float = 6.0

def get_wan_t2v_workflow(prompt, negative_prompt, width, height, num_frames, seed, steps, cfg):
    return {
        "1": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": "wan2.2_ti2v_5B_fp16.safetensors",
                "base_precision": "bf16",
                "quantization": "disabled",
                "load_device": "offload_device"
            }
        },
        "2": {
            "class_type": "WanVideoVAELoader",
            "inputs": {
                "model_name": "wan2.2_vae.safetensors",
                "precision": "bf16"
            }
        },
        "3": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "inputs": {
                "model_name": "umt5-xxl-enc-bf16.safetensors",
                "precision": "bf16",
                "load_device": "offload_device",
                "quantization": "disabled"
            }
        },
        "4": {
            "class_type": "WanVideoTextEncode",
            "inputs": {
                "positive_prompt": prompt,
                "negative_prompt": negative_prompt,
                "t5": ["3", 0],
                "force_offload": True
            }
        },
        "5": {
            "class_type": "WanVideoEmptyEmbeds",
            "inputs": {
                "width": width,
                "height": height,
                "num_frames": num_frames
            }
        },
        "6": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "model": ["1", 0],
                "image_embeds": ["5", 0],
                "text_embeds": ["4", 0],
                "steps": steps,
                "cfg": cfg,
                "shift": 5.0,
                "seed": seed,
                "force_offload": True,
                "scheduler": "unipc",
                "riflex_freq_index": 0
            }
        },
        "7": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "vae": ["2", 0],
                "samples": ["6", 0],
                "enable_vae_tiling": False,
                "tile_x": 272,
                "tile_y": 272,
                "tile_stride_x": 144,
                "tile_stride_y": 128
            }
        },
        "8": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["7", 0],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": f"video/wan_output_{seed}",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True
            }
        }
    }

@app.post("/wan/t2v")
async def wan_text_to_video(req: WanT2VRequest, background_tasks: BackgroundTasks):
    seed = req.seed if req.seed != -1 else uuid.uuid4().int % 2**32
    # Wan uses 4n+1 frames
    num_frames = ((req.seconds * 24 - 1) // 4) * 4 + 1
    workflow = get_wan_t2v_workflow(
        req.prompt, req.negative_prompt,
        req.width, req.height, num_frames,
        seed, req.steps, req.cfg
    )
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow)
    return {
        "job_id": job_id,
        "status": "queued",
        "model": "wan2.2-ti2v-5b",
        "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"
    }

class WanT2IRequest(BaseModel):
    prompt: str
    negative_prompt: str = "cartoon, anime, illustration, painting, drawing, cgi, render, 3d, digital art, watermark, deformed, bad anatomy, disfigured, mutated, extra limbs, bad hands, bad face, ugly, blurry, low quality, worst quality, overexposed, underexposed"
    width: int = 1024
    height: int = 1024
    seed: int = -1
    steps: int = 20
    cfg: float = 6.0
    enhance_prompt: bool = True

def get_wan_t2i_workflow(prompt, negative_prompt, width, height, seed, steps, cfg):
    # Using JuggernautXL for photorealistic image generation
    # Wan 5B is a video model — JuggernautXL gives much better face/portrait quality
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "juggernautXL_v9Rdphoto2Lightning.safetensors"
        }},
        "2": {"class_type": "CLIPTextEncode", "inputs": {
            "text": prompt,
            "clip": ["1", 1]
        }},
        "3": {"class_type": "CLIPTextEncode", "inputs": {
            "text": negative_prompt,
            "clip": ["1", 1]
        }},
        "4": {"class_type": "EmptyLatentImage", "inputs": {
            "width": width,
            "height": height,
            "batch_size": 1
        }},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1.0
        }},
        "6": {"class_type": "VAEDecode", "inputs": {
            "samples": ["5", 0],
            "vae": ["1", 2]
        }},
        "7": {"class_type": "SaveImage", "inputs": {
            "images": ["6", 0],
            "filename_prefix": f"images/t2i_{seed}"
        }},
    }

@app.post("/wan/t2i")
async def wan_text_to_image(req: WanT2IRequest, background_tasks: BackgroundTasks):
    seed = req.seed if req.seed != -1 else uuid.uuid4().int % 2**32
    if req.enhance_prompt:
        prompt = await enhance_prompt_with_ollama(req.prompt, "image")
    else:
        prompt = req.prompt
    workflow = get_wan_t2i_workflow(
        prompt, req.negative_prompt,
        req.width, req.height,
        seed, req.steps, req.cfg
    )
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow)
    return {
        "job_id": job_id,
        "status": "queued",
        "model": "wan2.2-ti2v-5b",
        "type": "image",
        "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"
    }

@app.get("/image/{filename}")
async def serve_image(filename: str):
    """Serve generated image files."""
    for path in [
        OUTPUT_DIR / "images" / filename,
        OUTPUT_DIR / filename
    ]:
        if path.exists():
            return FileResponse(str(path), media_type="image/png", filename=filename)
    raise HTTPException(404, f"Image not found: {filename}")

@app.post("/face-swap")
async def face_swap(
    background_tasks: BackgroundTasks,
    source_image: UploadFile = File(..., description="Image containing the face to use as source (user's face)"),
    target_image: UploadFile = File(..., description="Template image where the face will be swapped into"),
    swap_model: str = Form("inswapper_128.onnx", description="Swap model: inswapper_128.onnx, hyperswap_1a_256.onnx, hyperswap_1b_256.onnx, hyperswap_1c_256.onnx"),
    face_restore_visibility: float = Form(1.0, description="How visible the face restoration is (0.1-1.0)"),
    codeformer_weight: float = Form(0.5, description="CodeFormer fidelity weight (0.0-1.0). Lower = more restoration, Higher = more original"),
    detect_gender_source: str = Form("no", description="Gender filter for source face: no, female, male"),
    detect_gender_target: str = Form("no", description="Gender filter for target face: no, female, male"),
    source_face_index: str = Form("0", description="Which face to use from source image (0 = first/largest face)"),
    target_face_index: str = Form("0", description="Which face in target to replace (0 = first/largest face)")
):
    """
    Swap a face from source_image into target_image.
    - source_image: the user face photo
    - target_image: the template image to swap into
    Returns a completed image with the swapped face.
    """
    seed = uuid.uuid4().int % 2**32

    # Save both uploaded images
    source_filename = f"reactor_source_{uuid.uuid4().hex}.png"
    target_filename = f"reactor_target_{uuid.uuid4().hex}.png"
    source_path = str(Path("/workspace/ComfyUI/input") / source_filename)
    target_path = str(Path("/workspace/ComfyUI/input") / target_filename)

    Path(source_path).write_bytes(await source_image.read())
    Path(target_path).write_bytes(await target_image.read())

    workflow = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": target_filename}
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": source_filename}
        },
        "3": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["1", 0],
                "source_image": ["2", 0],
                "swap_model": swap_model,
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "GFPGANv1.4.pth",
                "face_restore_visibility": face_restore_visibility,
                "codeformer_weight": codeformer_weight,
                "detect_gender_input": detect_gender_target,
                "detect_gender_source": detect_gender_source,
                "input_faces_index": target_face_index,
                "source_faces_index": source_face_index,
                "console_log_level": 1
            }
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["3", 0],
                "filename_prefix": f"images/faceswap_{seed}"
            }
        }
    }

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, source_path)
    # Also cleanup target after job
    background_tasks.add_task(lambda: Path(target_path).unlink(missing_ok=True))

    return {
        "job_id": job_id,
        "status": "queued",
        "type": "image",
        "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"
    }


# ─────────────────────────────────────────────
# Face Swap + Animate (Chained Workflow)
# ─────────────────────────────────────────────

@app.post("/face-swap/animate")
async def face_swap_animate(
    background_tasks: BackgroundTasks,
    source_image: UploadFile = File(..., description="User face photo"),
    target_image: UploadFile = File(..., description="Template image to swap face into"),
    prompt: str = Form("the person moves naturally, smooth cinematic motion", description="Motion description for the animation"),
    model: str = Form("ltx", description="Animation model: ltx or wan"),
    negative_prompt: str = Form("worst quality, low quality, blurry, distorted, inconsistent motion, jittery, flickering, ghosting, deformed, bad anatomy, watermark"),
    width: int = Form(544, description="Output video width in pixels. Portrait: 544, Landscape: 960"),
    height: int = Form(960, description="Output video height in pixels. Portrait: 960, Landscape: 544"),
    seconds: int = Form(5, description="Video duration in seconds (1-10)"),
    seed: int = Form(-1, description="Random seed. -1 = random each time. Use same seed to reproduce results"),
    steps: int = Form(20, description="Inference steps. LTX: 8-30 recommended. Wan: 20-50 recommended. More = better quality but slower"),
    cfg: float = Form(1.5, description="CFG scale. LTX: keep 1.0-2.0. Wan: use 4.0-7.0. Controls how closely model follows the prompt"),
    audio: bool = Form(True, description="Generate audio alongside video. LTX only, ignored for Wan"),
    quality: str = Form("balanced", description="LTX quality preset: fast (8 steps), balanced (20 steps), high (30 steps). Overrides steps if set. Wan ignores this"),
    face_restore_visibility: float = Form(1.0, description="Face restoration strength after swap. 1.0 = full restoration, 0.1 = minimal"),
    codeformer_weight: float = Form(0.5, description="Face fidelity vs restoration balance. 0.0 = max restoration, 1.0 = max fidelity to original face"),
    detect_gender_source: str = Form("no", description="Filter source face by gender: no, female, male. Useful when source image has multiple people"),
    detect_gender_target: str = Form("no", description="Filter target face by gender: no, female, male"),
    source_face_index: str = Form("0", description="Which face to use from source image. 0 = first/largest face"),
    target_face_index: str = Form("0", description="Which face in template to replace. 0 = first/largest face"),
    swap_model: str = Form("inswapper_128.onnx", description="Swap model: inswapper_128.onnx, hyperswap_1a_256.onnx, hyperswap_1b_256.onnx, hyperswap_1c_256.onnx"),
):
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32

    # Apply quality preset for LTX
    # If quality is set AND steps is still default (20), use preset
    # If user explicitly sets steps, use their value (custom mode)
    if model == "ltx":
        if quality == "fast" and steps == 20:
            steps, cfg_val, use_lora = 8, 1.0, True
        elif quality == "balanced" and steps == 20:
            steps, cfg_val, use_lora = 20, 1.0, True
        elif quality == "high" and steps == 20:
            steps, cfg_val, use_lora = 30, 1.0, False
        else:
            # User set steps manually — use their value
            cfg_val = cfg
            use_lora = False if steps > 25 else True
    else:
        # Wan — use steps as-is, default cfg to 6.0 if user didn't change from LTX default
        cfg_val = cfg if cfg != 1.5 else 6.0
        use_lora = True

    # Save both images
    source_filename = f"reactor_source_{uuid.uuid4().hex}.png"
    target_filename = f"reactor_target_{uuid.uuid4().hex}.png"
    source_path = str(Path("/workspace/ComfyUI/input") / source_filename)
    target_path = str(Path("/workspace/ComfyUI/input") / target_filename)
    Path(source_path).write_bytes(await source_image.read())
    Path(target_path).write_bytes(await target_image.read())

    if model == "wan":
        num_frames = ((seconds * 24 - 1) // 4) * 4 + 1
        workflow = {
            # Load images
            "1": {"class_type": "LoadImage", "inputs": {"image": target_filename}},
            "2": {"class_type": "LoadImage", "inputs": {"image": source_filename}},
            # Face swap
            "3": {
                "class_type": "ReActorFaceSwap",
                "inputs": {
                    "enabled": True,
                    "input_image": ["1", 0],
                    "source_image": ["2", 0],
                    "swap_model": swap_model,
                    "facedetection": "retinaface_resnet50",
                    "face_restore_model": "GFPGANv1.4.pth",
                    "face_restore_visibility": face_restore_visibility,
                    "codeformer_weight": codeformer_weight,
                    "detect_gender_input": detect_gender_target,
                    "detect_gender_source": detect_gender_source,
                    "input_faces_index": target_face_index,
                    "source_faces_index": source_face_index,
                    "console_log_level": 1
                }
            },
            # Wan 2.2 I2V pipeline
            "4": {"class_type": "WanVideoModelLoader", "inputs": {"model": "wan2.2_ti2v_5B_fp16.safetensors", "base_precision": "bf16", "quantization": "disabled", "load_device": "offload_device"}},
            "5": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": "wan2.2_vae.safetensors", "precision": "bf16"}},
            "6": {"class_type": "LoadWanVideoT5TextEncoder", "inputs": {"model_name": "umt5-xxl-enc-bf16.safetensors", "precision": "bf16", "load_device": "offload_device", "quantization": "disabled"}},
            "7": {"class_type": "WanVideoTextEncode", "inputs": {"positive_prompt": prompt, "negative_prompt": negative_prompt, "t5": ["6", 0], "force_offload": True}},
            "8": {
                "class_type": "WanVideoImageToVideoEncode",
                "inputs": {
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "noise_aug_strength": 0.0,
                    "start_latent_strength": 1.0,
                    "end_latent_strength": 1.0,
                    "force_offload": True,
                    "vae": ["5", 0],
                    "start_image": ["3", 0]
                }
            },
            "9": {
                "class_type": "WanVideoSampler",
                "inputs": {
                    "model": ["4", 0],
                    "image_embeds": ["8", 0],
                    "text_embeds": ["7", 0],
                    "steps": steps,
                    "cfg": cfg_val,
                    "shift": 5.0,
                    "seed": seed,
                    "force_offload": True,
                    "scheduler": "unipc",
                    "riflex_freq_index": 0
                }
            },
            "10": {"class_type": "WanVideoDecode", "inputs": {"vae": ["5", 0], "samples": ["9", 0], "enable_vae_tiling": False, "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
            "11": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["10", 0], "frame_rate": 24, "loop_count": 0, "filename_prefix": f"video/faceswap_wan_{seed}", "format": "video/h264-mp4", "pingpong": False, "save_output": True}}
        }
    else:
        # LTX I2V (default)
        length = seconds_to_frames(seconds)
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}},
            "2": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}},
            "3": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors", "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors", "device": "default"}},
            "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "ltx-2.3-22b-distilled-lora-384.safetensors", "strength_model": 1.0 if use_lora else 0.0, "model": ["1", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["3", 0]}},
            "7": {"class_type": "LTXVConditioning", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 25.0}},
            # Load target image for face swap
            "20": {"class_type": "LoadImage", "inputs": {"image": target_filename}},
            "21": {"class_type": "LoadImage", "inputs": {"image": source_filename}},
            # Face swap
            "22": {
                "class_type": "ReActorFaceSwap",
                "inputs": {
                    "enabled": True,
                    "input_image": ["20", 0],
                    "source_image": ["21", 0],
                    "swap_model": swap_model,
                    "facedetection": "retinaface_resnet50",
                    "face_restore_model": "GFPGANv1.4.pth",
                    "face_restore_visibility": face_restore_visibility,
                    "codeformer_weight": codeformer_weight,
                    "detect_gender_input": detect_gender_target,
                    "detect_gender_source": detect_gender_source,
                    "input_faces_index": target_face_index,
                    "source_faces_index": source_face_index,
                    "console_log_level": 1
                }
            },
            # LTX I2V using swapped image
            "8": {"class_type": "LTXVImgToVideo", "inputs": {
                "positive": ["5", 0], "negative": ["6", 0],
                "vae": ["1", 2], "image": ["22", 0],
                "width": width, "height": height, "length": length,
                "batch_size": 1, "strength": 1.0
            }},
            "9": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": length, "frame_rate": 25, "batch_size": 1, "audio_vae": ["2", 0]}},
            "10": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["8", 2], "audio_latent": ["9", 0]}},
            "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_cfg_pp"}},
            "14": {"class_type": "LTXVScheduler", "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1, "latent": ["10", 0]}},
            "15": {"class_type": "CFGGuider", "inputs": {"cfg": cfg_val, "model": ["4", 0], "positive": ["8", 0], "negative": ["8", 1]}},
            "16": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["12", 0], "guider": ["15", 0], "sampler": ["13", 0], "sigmas": ["14", 0], "latent_image": ["10", 0]}},
            "17": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["16", 0]}},
            "18": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["17", 0], "vae": ["1", 2], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}},
            "19": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["17", 1], "audio_vae": ["2", 0]}},
            "23": {"class_type": "CreateVideo", "inputs": {"images": ["18", 0], **({"audio": ["19", 0]} if audio else {}), "fps": 24.0}},
            "24": {"class_type": "SaveVideo", "inputs": {"video": ["23", 0], "filename_prefix": f"video/faceswap_ltx_{seed}", "format": "auto", "codec": "auto"}}
        }

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, source_path)
    background_tasks.add_task(lambda: Path(target_path).unlink(missing_ok=True))

    return {
        "job_id": job_id,
        "status": "queued",
        "model": f"faceswap+{model}",
        "type": "video",
        "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"
    }



# ─────────────────────────────────────────────
# Head Swap: ReActor + SDXL img2img refinement
# ─────────────────────────────────────────────

@app.post("/head-swap")
async def head_swap(
    background_tasks: BackgroundTasks,
    source_image: UploadFile = File(..., description="User face/headshot — provides face identity"),
    target_image: UploadFile = File(..., description="Template body image — head gets replaced"),
    prompt: str = Form(
        default="photorealistic portrait, natural skin texture, seamless hair blending, professional photography, sharp focus, high quality",
        description="Describe the desired output style."
    ),
    negative_prompt: str = Form(
        default="deformed, bad anatomy, disfigured, watermark, blurry, low quality, ugly, distorted face, seam, pasted look, artificial, fake skin",
        description="What to avoid."
    ),
    steps: int = Form(20, description="Inference steps. 15-25 recommended."),
    cfg: float = Form(5.0, description="CFG scale. 4.0-7.0 recommended."),
    denoise: float = Form(0.45, description="Denoising strength. 0.3-0.5 = subtle refinement preserving body. 0.6+ = more changes."),
    seed: int = Form(-1, description="Random seed. -1 = random."),
    face_restore_visibility: float = Form(1.0, description="ReActor face restore strength 0.1-1.0."),
    codeformer_weight: float = Form(0.5, description="0.0 = max restoration, 1.0 = max fidelity to source face."),
    detect_gender_source: str = Form("no", description="Filter source face by gender: no, female, male."),
    detect_gender_target: str = Form("no", description="Filter target face by gender: no, female, male."),
):
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32

    source_filename = f"headswap_source_{uuid.uuid4().hex}.png"
    target_filename = f"headswap_target_{uuid.uuid4().hex}.png"
    source_path = str(Path("/workspace/ComfyUI/input") / source_filename)
    target_path = str(Path("/workspace/ComfyUI/input") / target_filename)
    Path(source_path).write_bytes(await source_image.read())
    Path(target_path).write_bytes(await target_image.read())

    from PIL import Image as PILImage
    with PILImage.open(target_path) as img:
        img_w, img_h = img.size
    width = (img_w // 8) * 8
    height = (img_h // 8) * 8

    workflow = {
        # Load images
        "1": {"class_type": "LoadImage", "inputs": {"image": target_filename}},
        "2": {"class_type": "LoadImage", "inputs": {"image": source_filename}},

        # Step 1: ReActor — swap face from source onto target
        "3": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["1", 0],
                "source_image": ["2", 0],
                "swap_model": swap_model,
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "GFPGANv1.4.pth",
                "face_restore_visibility": face_restore_visibility,
                "codeformer_weight": codeformer_weight,
                "detect_gender_input": detect_gender_target,
                "detect_gender_source": detect_gender_source,
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1
            }
        },

        # Load JuggernautXL
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "juggernautXL_v9Rdphoto2Lightning.safetensors"
        }},

        # Prompts
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": negative_prompt}},

        # Step 2: VAE encode the face-swapped result
        "7": {"class_type": "VAEEncode", "inputs": {
            "pixels": ["3", 0],
            "vae": ["4", 2]
        }},

        # Step 3: SDXL img2img with low denoise to blend and refine
        # This preserves body while cleaning up the face swap seams
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["4", 0],
            "positive": ["5", 0],
            "negative": ["6", 0],
            "latent_image": ["7", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": denoise
        }},

        # Decode
        "9": {"class_type": "VAEDecode", "inputs": {
            "samples": ["8", 0],
            "vae": ["4", 2]
        }},

        # Save
        "10": {"class_type": "SaveImage", "inputs": {
            "images": ["9", 0],
            "filename_prefix": f"images/headswap_{seed}"
        }}
    }

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, source_path)
    background_tasks.add_task(lambda: Path(target_path).unlink(missing_ok=True))

    return {
        "job_id": job_id,
        "status": "queued",
        "type": "image",
        "created_at": jobs[job_id]["created_at"],
        "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"
    }


# ─────────────────────────────────────────────
# Image to Image using JuggernautXL
# ─────────────────────────────────────────────

@app.post("/i2i")
async def image_to_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Input image to transform"),
    prompt: str = Form(..., description="What you want the output to look like"),
    negative_prompt: str = Form("cartoon, anime, illustration, painting, drawing, cgi, render, 3d, digital art, watermark, deformed, bad anatomy, disfigured, mutated, extra limbs, bad hands, bad face, ugly, blurry, low quality, worst quality"),
    denoise: float = Form(0.75, description="How much to change the image. 0.3=subtle, 0.75=moderate, 1.0=complete change"),
    steps: int = Form(20, description="Inference steps. 15-30 recommended."),
    cfg: float = Form(6.0, description="CFG scale. 5.0-8.0 recommended."),
    seed: int = Form(-1, description="Random seed. -1 = random."),
    enhance_prompt: bool = Form(True, description="Auto-enhance prompt using Ollama LLM"),
):
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32

    img_filename = f"i2i_input_{uuid.uuid4().hex}.png"
    img_path = str(Path("/workspace/ComfyUI/input") / img_filename)
    Path(img_path).write_bytes(await image.read())

    if enhance_prompt:
        final_prompt = await enhance_prompt_with_ollama(prompt, "image")
    else:
        final_prompt = prompt

    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "juggernautXL_v9Rdphoto2Lightning.safetensors"
        }},
        "2": {"class_type": "CLIPTextEncode", "inputs": {
            "text": final_prompt,
            "clip": ["1", 1]
        }},
        "3": {"class_type": "CLIPTextEncode", "inputs": {
            "text": negative_prompt,
            "clip": ["1", 1]
        }},
        "4": {"class_type": "LoadImage", "inputs": {"image": img_filename}},
        "5": {"class_type": "VAEEncode", "inputs": {
            "pixels": ["4", 0],
            "vae": ["1", 2]
        }},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["5", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": denoise
        }},
        "7": {"class_type": "VAEDecode", "inputs": {
            "samples": ["6", 0],
            "vae": ["1", 2]
        }},
        "8": {"class_type": "SaveImage", "inputs": {
            "images": ["7", 0],
            "filename_prefix": f"images/i2i_{seed}"
        }}
    }

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, img_path)

    return {
        "job_id": job_id,
        "status": "queued",
        "type": "image",
        "created_at": jobs[job_id]["created_at"],
        "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"
    }


# ─────────────────────────────────────────────
# Inpainting — change specific areas only
# ─────────────────────────────────────────────

@app.post("/inpaint")
async def inpaint(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Input image"),
    mask: UploadFile = File(None, description="Optional mask image (white=change, black=keep). If not provided, auto-detects hair."),
    prompt: str = Form(..., description="What to put in the masked area"),
    negative_prompt: str = Form("deformed, bad anatomy, blurry, low quality, watermark, cartoon"),
    mask_area: str = Form("hair", description="Auto-mask area if no mask provided: hair, face, background, clothes"),
    denoise: float = Form(0.95, description="Inpainting strength 0.7-1.0"),
    steps: int = Form(25),
    cfg: float = Form(7.0),
    seed: int = Form(-1),
    enhance_prompt: bool = Form(False),
):
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32

    img_filename = f"inpaint_input_{uuid.uuid4().hex}.png"
    img_path = str(Path("/workspace/ComfyUI/input") / img_filename)
    Path(img_path).write_bytes(await image.read())

    if enhance_prompt:
        final_prompt = await enhance_prompt_with_ollama(prompt, "image")
    else:
        final_prompt = prompt

    # If mask provided, use it. Otherwise use SAM auto-segmentation
    if mask and mask.filename:
        mask_filename = f"inpaint_mask_{uuid.uuid4().hex}.png"
        mask_path = str(Path("/workspace/ComfyUI/input") / mask_filename)
        Path(mask_path).write_bytes(await mask.read())

        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
                "ckpt_name": "juggernautXL_v9Rdphoto2Lightning.safetensors"
            }},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": final_prompt, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
            "4": {"class_type": "LoadImage", "inputs": {"image": img_filename}},
            "5": {"class_type": "LoadImage", "inputs": {"image": mask_filename}},
            "6": {"class_type": "ImageToMask", "inputs": {"image": ["5", 0], "channel": "red"}},
            "7": {"class_type": "VAEEncodeForInpaint", "inputs": {
                "pixels": ["4", 0],
                "vae": ["1", 2],
                "mask": ["6", 0],
                "grow_mask_by": 6
            }},
            "8": {"class_type": "KSampler", "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise
            }},
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
            "10": {"class_type": "SaveImage", "inputs": {
                "images": ["9", 0],
                "filename_prefix": f"images/inpaint_{seed}"
            }}
        }
    else:
        # Use SAMLoader + SAMDetectorCombined from Impact Pack for auto-masking
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
                "ckpt_name": "juggernautXL_v9Rdphoto2Lightning.safetensors"
            }},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": final_prompt, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
            "4": {"class_type": "LoadImage", "inputs": {"image": img_filename}},
            # SAM auto-segment using text prompt
            "5": {"class_type": "SAMLoader", "inputs": {
                "model_name": "sam_vit_b_01ec64.pth",
                "device_mode": "AUTO"
            }},
            "6": {"class_type": "CLIPSegDetectorProvider", "inputs": {
                "text": mask_area,
                "blur": 7.0,
                "threshold": 0.4,
                "dilation_factor": 4
            }},
            "7": {"class_type": "ImpactSimpleDetectorSEGS", "inputs": {
                "bbox_detector": ["6", 0],
                "image": ["4", 0],
                "bbox_threshold": 0.4,
                "bbox_dilation": 10,
                "crop_factor": 3.0,
                "drop_size": 10,
                "sub_threshold": 0.5,
                "sub_dilation": 0,
                "sub_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_model_opt": ["5", 0]
            }},
            "8": {"class_type": "SegsToCombinedMask", "inputs": {"segs": ["7", 0]}},
            "9": {"class_type": "VAEEncodeForInpaint", "inputs": {
                "pixels": ["4", 0],
                "vae": ["1", 2],
                "mask": ["8", 0],
                "grow_mask_by": 6
            }},
            "10": {"class_type": "KSampler", "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["9", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise
            }},
            "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
            "12": {"class_type": "SaveImage", "inputs": {
                "images": ["11", 0],
                "filename_prefix": f"images/inpaint_{seed}"
            }}
        }

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, img_path)

    return {
        "job_id": job_id,
        "status": "queued",
        "type": "image",
        "created_at": jobs[job_id]["created_at"],
        "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"
    }



# ─────────────────────────────────────────────
# FLUX.2 Klein 9B Head/Face Swap
# ─────────────────────────────────────────────

FLUX_ASPECT_RATIOS = {
    "1:1":   (1024, 1024),
    "9:16":  (768, 1360),
    "3:4":   (880, 1168),
    "4:3":   (1168, 880),
    "16:9":  (1360, 768),
    "2:3":   (832, 1248),
    "21:9":  (1680, 720),
}

DEFAULT_FLUX_PROMPT = """head_swap: Use image 1 as the base image, preserving its environment, background, camera perspective, framing, exposure, contrast, and lighting. Remove the head and hair from image 1 and seamlessly replace it with the head from image 2.
Match the original head size, face-to-body ratio, neck thickness, shoulder alignment, and camera distance so proportions remain natural and unchanged.
Adapt the inserted head to the lighting of image 1 by matching light direction, intensity, softness, color temperature, shadows, and highlights, with no independent relighting.
Preserve the identity of image 2, including hair texture, eye color, nose structure, facial proportions, and skin details.
Match the pose and expression from image 1, including head tilt, rotation, eye direction, gaze, micro-expressions, and lip position.
Ensure seamless neck and jaw blending, consistent skin tone, realistic shadow contact, natural skin texture, and uniform sharpness.
Photorealistic, high quality, sharp details, 4K."""

@app.post("/flux/face-swap")
async def flux_face_swap(
    background_tasks: BackgroundTasks,
    target_image: UploadFile = File(..., description="Base/template image — body stays, head gets replaced"),
    face_image: UploadFile = File(..., description="Source face — identity to transfer"),
    seed: int = Form(-1, description="Random seed. -1 = random."),
):
    import copy
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32

    target_filename = f"flux_target_{uuid.uuid4().hex}.png"
    face_filename = f"flux_face_{uuid.uuid4().hex}.png"
    target_path = str(Path("/workspace/ComfyUI/input") / target_filename)
    face_path = str(Path("/workspace/ComfyUI/input") / face_filename)
    Path(target_path).write_bytes(await target_image.read())
    Path(face_path).write_bytes(await face_image.read())

    # Load raw workflow and convert to API format
    with open("/workspace/flux2_klein_face_swap.json") as f:
        wf_data = json.load(f)

    # Build links lookup: link_id -> [src_node_id, src_slot]
    links = {}
    for link in wf_data.get("links", []):
        # link format: [link_id, src_node, src_slot, dst_node, dst_slot, type]
        links[link[0]] = [str(link[1]), link[2]]

    # Build prompt
    prompt = {}
    for node in wf_data["nodes"]:
        nid = str(node["id"])
        ntype = node["type"]
        wvals = node.get("widgets_values", [])
        inputs = {}

        # Add linked inputs
        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is not None and link_id in links:
                inputs[inp["name"]] = links[link_id]

        # Add widget values by node type
        if ntype == "LoadImage":
            if node["id"] == 151:  # target body image
                inputs["image"] = target_filename
                inputs["upload"] = "image"
            elif node["id"] == 121:  # face reference
                inputs["image"] = face_filename
                inputs["upload"] = "image"
            else:
                inputs["image"] = face_filename  # use face as fallback for extra slots
                inputs["upload"] = "image"

        elif ntype == "SaveImage":
            inputs["filename_prefix"] = f"images/flux_swap_{seed}"

        elif ntype == "UNETLoader":
            inputs["unet_name"] = wvals[0] if len(wvals) > 0 else ""
            inputs["weight_dtype"] = wvals[1] if len(wvals) > 1 else "default"

        elif ntype == "CLIPLoader":
            inputs["clip_name"] = wvals[0] if len(wvals) > 0 else ""
            inputs["type"] = wvals[1] if len(wvals) > 1 else "flux2"
            inputs["device"] = wvals[2] if len(wvals) > 2 else "default"

        elif ntype == "VAELoader":
            inputs["vae_name"] = wvals[0] if wvals else ""

        elif ntype == "LoraLoader" or ntype == "LoraLoaderModelOnly":
            inputs["lora_name"] = wvals[0] if len(wvals) > 0 else ""
            inputs["strength_model"] = wvals[1] if len(wvals) > 1 else 1.0
            if ntype == "LoraLoader":
                inputs["strength_clip"] = wvals[2] if len(wvals) > 2 else 1.0

        elif ntype == "CLIPTextEncode":
            # Use custom prompt if provided, otherwise use default head swap prompt
            final_flux_prompt = prompt if prompt else DEFAULT_FLUX_PROMPT
            inputs["text"] = final_flux_prompt

        elif ntype == "FluxGuidance":
            inputs["guidance"] = wvals[0] if wvals else 4.0

        elif ntype == "LanPaint_KSampler":
            inputs["seed"] = seed
            inputs["control_after_generate"] = wvals[1] if len(wvals) > 1 else "randomize"
            inputs["steps"] = steps
            inputs["cfg"] = cfg
            inputs["sampler_name"] = wvals[4] if len(wvals) > 4 else "euler"
            inputs["scheduler"] = wvals[5] if len(wvals) > 5 else "simple"
            inputs["denoise"] = wvals[6] if len(wvals) > 6 else 1.0
            inputs["LanPaint_NumSteps"] = wvals[7] if len(wvals) > 7 else 2
            inputs["LanPaint_PromptMode"] = wvals[8] if len(wvals) > 8 else "Image First"
            inputs["Inpainting_mode"] = "🖼️ Image Inpainting"
            inputs["LanPaint_Info"] = wvals[9] if len(wvals) > 9 else "LanPaint KSampler"

        elif ntype == "ImageScaleToTotalPixels":
            inputs["upscale_method"] = wvals[0] if len(wvals) > 0 else "lanczos"
            inputs["megapixels"] = wvals[1] if len(wvals) > 1 else 1.0
            inputs["resolution_steps"] = wvals[2] if len(wvals) > 2 else 1

        elif ntype == "ImageScale":
            inputs["upscale_method"] = wvals[0] if len(wvals) > 0 else "lanczos"
            inputs["width"] = wvals[1] if len(wvals) > 1 else 1024
            inputs["height"] = wvals[2] if len(wvals) > 2 else 1024
            inputs["crop"] = wvals[3] if len(wvals) > 3 else "center"

        elif ntype == "ResizeImageMaskNode":
            inputs["resize_type"] = wvals[0] if len(wvals) > 0 else "scale dimensions"
            inputs["width"] = wvals[1] if len(wvals) > 1 else 512
            inputs["height"] = wvals[2] if len(wvals) > 2 else 512
            inputs["crop"] = wvals[3] if len(wvals) > 3 else "center"
            inputs["scale_method"] = wvals[4] if len(wvals) > 4 else "bicubic"
            inputs["resize_type.width"] = wvals[1] if len(wvals) > 1 else 512
            inputs["resize_type.height"] = wvals[2] if len(wvals) > 2 else 512
            inputs["resize_type.crop"] = wvals[3] if len(wvals) > 3 else "center"
            inputs["resize_type.scale_method"] = wvals[4] if len(wvals) > 4 else "bicubic"

        elif ntype == "FloatConstant":
            inputs["value"] = wvals[0] if wvals else 2.0

        elif ntype == "ConditioningZeroOut":
            pass  # only linked inputs

        elif "EmptyFlux2Latent" in ntype or "EmptyLatent" in ntype:
            inputs["width"] = wvals[0] if len(wvals) > 0 else 1008
            inputs["height"] = wvals[1] if len(wvals) > 1 else 1024
            inputs["batch_size"] = wvals[2] if len(wvals) > 2 else 1

        elif ntype == "SetLatentNoiseMask":
            pass  # only linked inputs

        elif ntype == "VAEEncode":
            pass  # only linked inputs

        elif ntype == "VAEDecode":
            pass  # only linked inputs

        elif ntype == "ReferenceLatent":
            pass  # only linked inputs

        elif ntype == "ComfySwitchNode":
            inputs["switch"] = wvals[0] if wvals else True

        elif ntype == "PreviewMask":
            pass  # preview only

        else:
            # Generic: map widget values positionally
            for i, wv in enumerate(wvals):
                inputs[f"widget_{i}"] = wv

        prompt[nid] = {"class_type": ntype, "inputs": inputs}

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, prompt, target_path)
    background_tasks.add_task(lambda: Path(face_path).unlink(missing_ok=True))

    return {
        "job_id": job_id,
        "status": "queued",
        "type": "image",
        "model": "flux2-klein-9b",
        "created_at": jobs[job_id]["created_at"],
        "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"
    }

# ─────────────────────────────────────────────
# FLUX.2 Klein Face Swap + Animate (Chained)
# ─────────────────────────────────────────────

async def run_flux_then_animate(job_id: str, flux_prompt: dict, animate_workflow_fn, target_path: str, face_path: str):
    """Run FLUX face swap, then feed result image into animation workflow."""
    import asyncio, glob

    # Step 1: Run FLUX face swap
    jobs[job_id] = {**jobs.get(job_id, {}), "status": "processing", "step": "1/2 — FLUX head swap", "started_at": datetime.now(timezone.utc).isoformat()}

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post("http://localhost:8188/prompt", json={"prompt": flux_prompt})
            r.raise_for_status()
            prompt_id = r.json()["prompt_id"]

        # Poll until FLUX job done
        output_image_path = None
        for _ in range(400):
            await asyncio.sleep(3)
            async with httpx.AsyncClient(timeout=30) as client:
                hist = await client.get(f"http://localhost:8188/history/{prompt_id}")
                data = hist.json()
            if prompt_id in data:
                entry = data[prompt_id]
                if entry.get("status", {}).get("completed"):
                    outputs = entry.get("outputs", {})
                    for node_out in outputs.values():
                        for img in node_out.get("images", []):
                            fname = img.get("filename", "")
                            subfolder = img.get("subfolder", "")
                            fpath = Path("/workspace/ComfyUI/output") / subfolder / fname
                            if fpath.exists():
                                output_image_path = str(fpath)
                    break
                if entry.get("status", {}).get("status_str") == "error":
                    raise Exception("FLUX face swap failed")

        if not output_image_path:
            raise Exception("FLUX face swap produced no output image")

        # Step 2: Build animation workflow using the swapped image
        jobs[job_id] = {**jobs.get(job_id, {}), "step": "2/2 — Animating"}

        # Copy swapped image to ComfyUI input
        swapped_filename = f"flux_swapped_{job_id[:8]}.png"
        swapped_path = f"/workspace/ComfyUI/input/{swapped_filename}"
        import shutil
        shutil.copy2(output_image_path, swapped_path)

        # Get animation workflow with the swapped image
        anim_workflow = animate_workflow_fn(swapped_filename)

        # Run animation
        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post("http://localhost:8188/prompt", json={"prompt": anim_workflow})
            r.raise_for_status()
            anim_prompt_id = r.json()["prompt_id"]

        # Poll animation
        for _ in range(600):
            await asyncio.sleep(3)
            async with httpx.AsyncClient(timeout=30) as client:
                hist = await client.get(f"http://localhost:8188/history/{anim_prompt_id}")
                data = hist.json()
            if anim_prompt_id in data:
                entry = data[anim_prompt_id]
                if entry.get("status", {}).get("completed"):
                    outputs = entry.get("outputs", {})
                    for node_out in outputs.values():
                        for vid in node_out.get("gifs", []) + node_out.get("videos", []):
                            fname = vid.get("filename", "")
                            subfolder = vid.get("subfolder", "")
                            fpath = Path("/workspace/ComfyUI/output") / subfolder / fname
                            if fpath.exists():
                                completed_at = datetime.now(timezone.utc)
                                started = datetime.fromisoformat(jobs[job_id].get("started_at", completed_at.isoformat()))
                                duration = round((completed_at - started).total_seconds(), 1)
                                url = f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/video/{subfolder}/{fname}" if subfolder else f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/video/{fname}"
                                jobs[job_id] = {"status": "completed", "url": url, "filename": str(fpath.name), "completed_at": completed_at.isoformat(), "duration_seconds": duration}
                                return
                    break
                if entry.get("status", {}).get("status_str") == "error":
                    raise Exception("Animation failed")

        raise Exception("Animation timed out")

    except Exception as e:
        jobs[job_id] = {**jobs.get(job_id, {}), "status": "failed", "error": str(e), "failed_at": datetime.now(timezone.utc).isoformat()}
    finally:
        Path(target_path).unlink(missing_ok=True)
        Path(face_path).unlink(missing_ok=True)


@app.post("/flux/face-swap/animate")
async def flux_face_swap_animate(
    background_tasks: BackgroundTasks,
    target_image: UploadFile = File(..., description="Template body image — body/pose stays, head gets replaced"),
    face_image: UploadFile = File(..., description="Source face — identity to transfer"),
    prompt: str = Form("the person moves naturally, smooth cinematic motion", description="Motion description for animation"),
    model: str = Form("ltx", description="Animation model: ltx or wan"),
    negative_prompt: str = Form("worst quality, low quality, blurry, distorted, inconsistent motion, jittery, flickering"),
    width: int = Form(544, description="Output video width. Portrait: 544 (LTX) or 704 (Wan)"),
    height: int = Form(960, description="Output video height. Portrait: 960 (LTX) or 1280 (Wan)"),
    seconds: int = Form(5, description="Video duration in seconds (1-10)"),
    seed: int = Form(-1, description="Random seed. -1 = random"),
    steps: int = Form(20, description="Inference steps"),
    cfg: float = Form(1.5, description="CFG scale. LTX: 1.0-2.0. Wan: 4.0-7.0"),
    audio: bool = Form(True, description="Generate audio. LTX only"),
    quality: str = Form("balanced", description="LTX quality preset: fast, balanced, high"),
):
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    flux_seed = uuid.uuid4().int % 2**32

    # Save images
    target_filename = f"flux_target_{uuid.uuid4().hex}.png"
    face_filename = f"flux_face_{uuid.uuid4().hex}.png"
    target_path = str(Path("/workspace/ComfyUI/input") / target_filename)
    face_path = str(Path("/workspace/ComfyUI/input") / face_filename)
    Path(target_path).write_bytes(await target_image.read())
    Path(face_path).write_bytes(await face_image.read())

    # Build FLUX swap prompt (same as /flux/face-swap)
    with open("/workspace/flux2_klein_face_swap.json") as f:
        wf_data = json.load(f)

    links = {}
    for link in wf_data.get("links", []):
        links[link[0]] = [str(link[1]), link[2]]

    flux_prompt = {}
    for node in wf_data["nodes"]:
        nid = str(node["id"])
        ntype = node["type"]
        wvals = node.get("widgets_values", [])
        inputs = {}
        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is not None and link_id in links:
                inputs[inp["name"]] = links[link_id]
        if ntype == "LoadImage":
            if node["id"] == 151:
                inputs["image"] = target_filename
                inputs["upload"] = "image"
            elif node["id"] == 121:
                inputs["image"] = face_filename
                inputs["upload"] = "image"
            else:
                inputs["image"] = face_filename
                inputs["upload"] = "image"
        elif ntype == "SaveImage":
            inputs["filename_prefix"] = f"images/flux_swap_{flux_seed}"
        elif ntype == "UNETLoader":
            inputs["unet_name"] = wvals[0] if wvals else ""
            inputs["weight_dtype"] = wvals[1] if len(wvals) > 1 else "default"
        elif ntype == "CLIPLoader":
            inputs["clip_name"] = wvals[0] if wvals else ""
            inputs["type"] = wvals[1] if len(wvals) > 1 else "flux2"
            inputs["device"] = wvals[2] if len(wvals) > 2 else "default"
        elif ntype == "VAELoader":
            inputs["vae_name"] = wvals[0] if wvals else ""
        elif ntype == "LoraLoader" or ntype == "LoraLoaderModelOnly":
            inputs["lora_name"] = wvals[0] if wvals else ""
            inputs["strength_model"] = wvals[1] if len(wvals) > 1 else 1.0
            if ntype == "LoraLoader":
                inputs["strength_clip"] = wvals[2] if len(wvals) > 2 else 1.0
        elif ntype == "CLIPTextEncode":
            inputs["text"] = wvals[0] if wvals else ""
        elif ntype == "FluxGuidance":
            inputs["guidance"] = wvals[0] if wvals else 4.0
        elif ntype == "LanPaint_KSampler":
            inputs["seed"] = flux_seed
            inputs["control_after_generate"] = wvals[1] if len(wvals) > 1 else "randomize"
            inputs["steps"] = wvals[2] if len(wvals) > 2 else 4
            inputs["cfg"] = wvals[3] if len(wvals) > 3 else 1.0
            inputs["sampler_name"] = wvals[4] if len(wvals) > 4 else "euler"
            inputs["scheduler"] = wvals[5] if len(wvals) > 5 else "simple"
            inputs["denoise"] = wvals[6] if len(wvals) > 6 else 1.0
            inputs["LanPaint_NumSteps"] = wvals[7] if len(wvals) > 7 else 2
            inputs["LanPaint_PromptMode"] = wvals[8] if len(wvals) > 8 else "Image First"
            inputs["Inpainting_mode"] = "🖼️ Image Inpainting"
            inputs["LanPaint_Info"] = wvals[9] if len(wvals) > 9 else "LanPaint KSampler"
        elif ntype == "ImageScaleToTotalPixels":
            inputs["upscale_method"] = wvals[0] if wvals else "lanczos"
            inputs["megapixels"] = megapixels
            inputs["resolution_steps"] = wvals[2] if len(wvals) > 2 else 1
        elif ntype == "ResizeImageMaskNode":
            inputs["resize_type"] = wvals[0] if wvals else "scale dimensions"
            inputs["width"] = wvals[1] if len(wvals) > 1 else 512
            inputs["height"] = wvals[2] if len(wvals) > 2 else 512
            inputs["crop"] = wvals[3] if len(wvals) > 3 else "center"
            inputs["scale_method"] = wvals[4] if len(wvals) > 4 else "bicubic"
            inputs["resize_type.width"] = wvals[1] if len(wvals) > 1 else 512
            inputs["resize_type.height"] = wvals[2] if len(wvals) > 2 else 512
            inputs["resize_type.crop"] = wvals[3] if len(wvals) > 3 else "center"
            inputs["resize_type.scale_method"] = wvals[4] if len(wvals) > 4 else "bicubic"
        elif ntype == "FloatConstant":
            inputs["value"] = wvals[0] if wvals else 2.0
        elif "EmptyFlux2Latent" in ntype or "EmptyLatent" in ntype:
            ar_w, ar_h = FLUX_ASPECT_RATIOS.get(aspect_ratio, (768, 1360))
            inputs["width"] = ar_w
            inputs["height"] = ar_h
            inputs["batch_size"] = wvals[2] if len(wvals) > 2 else 1
        elif ntype == "ComfySwitchNode":
            inputs["switch"] = wvals[0] if wvals else True
        flux_prompt[nid] = {"class_type": ntype, "inputs": inputs}

    # LTX quality preset
    if model == "ltx":
        if quality == "fast" and steps == 20:
            steps, cfg_val, use_lora = 8, 1.0, True
        elif quality == "high" and steps == 20:
            steps, cfg_val, use_lora = 30, 1.0, False
        else:
            cfg_val, use_lora = cfg, steps <= 25
    else:
        cfg_val = cfg if cfg != 1.5 else 6.0
        use_lora = True

    # Build animation workflow factory
    def make_animate_workflow(swapped_filename: str):
        if model == "wan":
            num_frames = ((seconds * 24 - 1) // 4) * 4 + 1
            return {
                "1": {"class_type": "LoadImage", "inputs": {"image": swapped_filename}},
                "4": {"class_type": "WanVideoModelLoader", "inputs": {"model": "wan2.2_ti2v_5B_fp16.safetensors", "base_precision": "bf16", "quantization": "disabled", "load_device": "offload_device"}},
                "5": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": "wan2.2_vae.safetensors", "precision": "bf16"}},
                "6": {"class_type": "LoadWanVideoT5TextEncoder", "inputs": {"model_name": "umt5-xxl-enc-bf16.safetensors", "precision": "bf16", "load_device": "offload_device", "quantization": "disabled"}},
                "7": {"class_type": "WanVideoTextEncode", "inputs": {"positive_prompt": prompt, "negative_prompt": negative_prompt, "t5": ["6", 0], "force_offload": True}},
                "8": {"class_type": "WanVideoImageToVideoEncode", "inputs": {"width": width, "height": height, "num_frames": num_frames, "noise_aug_strength": 0.0, "start_latent_strength": 1.0, "end_latent_strength": 1.0, "force_offload": True, "vae": ["5", 0], "start_image": ["1", 0]}},
                "9": {"class_type": "WanVideoSampler", "inputs": {"model": ["4", 0], "image_embeds": ["8", 0], "text_embeds": ["7", 0], "steps": steps, "cfg": cfg_val, "shift": 5.0, "seed": seed, "force_offload": True, "scheduler": "unipc", "riflex_freq_index": 0}},
                "10": {"class_type": "WanVideoDecode", "inputs": {"vae": ["5", 0], "samples": ["9", 0], "enable_vae_tiling": False, "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
                "11": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["10", 0], "frame_rate": 24, "loop_count": 0, "filename_prefix": f"video/flux_anim_wan_{seed}", "format": "video/h264-mp4", "pingpong": False, "save_output": True}}
            }
        else:
            length = seconds_to_frames(seconds)
            return {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}},
                "2": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}},
                "3": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors", "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors", "device": "default"}},
                "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "ltx-2.3-22b-distilled-lora-384.safetensors", "strength_model": 1.0 if use_lora else 0.0, "model": ["1", 0]}},
                "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
                "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["3", 0]}},
                "20": {"class_type": "LoadImage", "inputs": {"image": swapped_filename}},
                "8": {"class_type": "LTXVImgToVideo", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "vae": ["1", 2], "image": ["20", 0], "width": width, "height": height, "length": length, "batch_size": 1, "strength": 1.0}},
                "9": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": length, "frame_rate": 25, "batch_size": 1, "audio_vae": ["2", 0]}},
                "10": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["8", 2], "audio_latent": ["9", 0]}},
                "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
                "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_cfg_pp"}},
                "14": {"class_type": "LTXVScheduler", "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1, "latent": ["10", 0]}},
                "15": {"class_type": "CFGGuider", "inputs": {"cfg": cfg_val, "model": ["4", 0], "positive": ["8", 0], "negative": ["8", 1]}},
                "16": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["12", 0], "guider": ["15", 0], "sampler": ["13", 0], "sigmas": ["14", 0], "latent_image": ["10", 0]}},
                "17": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["16", 0]}},
                "18": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["17", 0], "vae": ["1", 2], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}},
                "19": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["17", 1], "audio_vae": ["2", 0]}},
                "23": {"class_type": "CreateVideo", "inputs": {"images": ["18", 0], **({"audio": ["19", 0]} if audio else {}), "fps": 24.0}},
                "24": {"class_type": "SaveVideo", "inputs": {"video": ["23", 0], "filename_prefix": f"video/flux_anim_ltx_{seed}", "format": "auto", "codec": "auto"}}
            }

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_flux_then_animate, job_id, flux_prompt, make_animate_workflow, target_path, face_path)

    return {
        "job_id": job_id,
        "status": "queued",
        "model": f"flux2-klein-9b+{model}",
        "type": "video",
        "created_at": jobs[job_id]["created_at"],
        "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"
    }
