import uuid, json, httpx, os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import websockets, asyncio

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
    width: int = 544
    height: int = 960
    seconds: int = 5
    seed: int = -1
    steps: int = 20
    cfg: float = 1.0
    enhance_prompt: bool = False
    audio: bool = True
    quality: str = "balanced"

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
    jobs[job_id] = {"status": "processing", "workflow": workflow}
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
                        jobs[job_id] = {"status": "completed", "url": url, "filename": filename}
                        if image_path:
                            Path(image_path).unlink(missing_ok=True)
                        return

        jobs[job_id] = {"status": "failed", "error": "No output found"}
    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}

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

    prompt = f"high quality, sharp, cinematic, 4k, smooth motion, professional video. {req.prompt}" if req.enhance_prompt else req.prompt
    workflow = get_workflow(prompt, req.negative_prompt, req.width, req.height, length, seed, cfg=cfg, steps=steps, audio=req.audio, use_lora=use_lora)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued"}
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
    jobs[job_id] = {"status": "queued"}
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
    jobs[job_id] = {"status": "queued"}
    background_tasks.add_task(run_job, job_id, workflow)
    return {
        "job_id": job_id,
        "status": "queued",
        "model": "wan2.2-ti2v-5b",
        "poll_url": f"https://t6pgge1y1kl2qt-7860.proxy.runpod.net/status/{job_id}"
    }

class WanT2IRequest(BaseModel):
    prompt: str
    negative_prompt: str = "worst quality, low quality, lowres, blurry, pixelated, jpeg artifacts, compression artifacts, noisy, grainy, unclear details, low contrast, low resolution, bad art, cartoon, anime, illustration, painting, sketch, drawing, cgi, render, 3d, comic, manga, watercolor, oil painting, digital art, concept art, artstation, octane render, cinema 4d, unreal engine, 2d, flat art, watermark, signature, text, logo, username, artist name, copyright, bad anatomy, bad proportions, deformed, disfigured, malformed, mutated, extra limbs, extra arms, extra legs, missing arms, missing legs, floating limbs, disconnected limbs, amputated, bad hands, poorly drawn hands, mutated hands, extra hands, missing hands, fused fingers, extra fingers, missing fingers, too many fingers, extra digits, fewer digits, long fingers, short fingers, malformed hands, bad face, poorly drawn face, cloned face, fused face, extra eyes, bad eyes, ugly eyes, deformed eyes, deformed pupils, deformed iris, cross-eyed, wall eye, asymmetrical face, uneven eyes, misaligned eyes, oversized eyes, tiny eyes, long neck, short neck, extra heads, multiple heads, multiple faces, bad feet, poorly drawn feet, extra feet, missing feet, unnatural pose, stiff pose, rigid pose, awkward pose, plastic skin, waxy skin, rubber skin, shiny skin, oily skin, unnatural skin tone, orange skin, gray skin, green skin, mannequin, doll, puppet, fake, artificial, fabric artifacts, wrinkled texture, unrealistic texture, bad cloth, distorted cloth, melting cloth, wrong material, unrealistic material, bad texture, bad background, distorted background, background inconsistency, bad architecture, distorted buildings, broken perspective, floating objects, impossible physics, unrealistic environment, wrong scale, disproportionate objects, overexposed, underexposed, washed out, oversaturated, desaturated, harsh lighting, flat lighting, bad lighting, unnatural lighting, color bleeding, chromatic aberration, color banding, monochrome when not intended, wrong colors, inconsistent motion, jittery, stuttering, flickering, frame drops, temporal inconsistency, ghosting, video compression artifacts, low framerate, choppy, freezing, looping artifacts, morphing artifacts, identity change, face distortion between frames, motion blur, out of focus, duplicate, clone, tiling, collage, split screen, vhs, old film, film grain, vintage, retro, lens flare, static, glitch, corrupted, broken, ugly, gross, creepy, disturbing"
    width: int = 704
    height: int = 1280
    seed: int = -1
    steps: int = 30
    cfg: float = 4.0

def get_wan_t2i_workflow(prompt, negative_prompt, width, height, seed, steps, cfg):
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
                "num_frames": 1
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
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": f"images/wan_image_{seed}"
            }
        }
    }

@app.post("/wan/t2i")
async def wan_text_to_image(req: WanT2IRequest, background_tasks: BackgroundTasks):
    seed = req.seed if req.seed != -1 else uuid.uuid4().int % 2**32
    workflow = get_wan_t2i_workflow(
        req.prompt, req.negative_prompt,
        req.width, req.height,
        seed, req.steps, req.cfg
    )
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued"}
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
