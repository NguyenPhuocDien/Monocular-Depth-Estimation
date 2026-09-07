import os
import sys
import io
import time
import base64
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.cm as cm

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Setup BTS paths
bts_dir = Path(__file__).resolve().parent.parent / "bts" / "pytorch"
if str(bts_dir) not in sys.path:
    sys.path.insert(0, str(bts_dir))

from bts import BtsModel

app = FastAPI(title="BTS Monocular Depth Estimation Demo", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ENGINE] Running on device: {device}")

# Model Registry
ROOT_DIR = Path(__file__).resolve().parent.parent

def find_checkpoint():
    candidates = [
        ROOT_DIR / "artifacts" / "checkpoints" / "nyu_densenet161_step271000_best_absrel_0.10967.pth",
        ROOT_DIR / "artifacts" / "checkpoints" / "nyu_densenet161_step302899_final.pth",
        ROOT_DIR / "kaggle_checkpoint_dataset" / "model-latest"
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

PRESETS = {
    "nyu": {
        "id": "nyu",
        "name": "NYU Depth V2 (Indoor)",
        "desc": "Huấn luyện trên 24,231 ảnh phòng nội thất, dải đo 0.001m - 10.0m",
        "max_depth": 10.0,
        "default_focal": 519.0,
        "ckpt_path": find_checkpoint(),
        "encoder": "densenet161_bts",
        "dataset": "nyu"
    },
    "kitti": {
        "id": "kitti",
        "name": "KITTI Benchmark (Outdoor)",
        "desc": "Môi trường ngoài trời & xe tự hành, dải đo 0.001m - 80.0m",
        "max_depth": 80.0,
        "default_focal": 715.0,
        "ckpt_path": find_checkpoint(),
        "encoder": "densenet161_bts",
        "dataset": "kitti"
    }
}

active_models = {}

def get_or_load_model(preset_key: str = "nyu"):
    if preset_key not in PRESETS:
        preset_key = "nyu"
    
    if preset_key in active_models:
        return active_models[preset_key], PRESETS[preset_key]
    
    cfg = PRESETS[preset_key]
    print(f"[ENGINE] Loading model preset '{preset_key}' from {cfg['ckpt_path']}...")
    
    class Args:
        encoder = cfg["encoder"]
        bts_size = 512
        max_depth = cfg["max_depth"]
        dataset = cfg["dataset"]
        num_threads = 1
        mode = "test"
    
    model = BtsModel(params=Args())
    if cfg["ckpt_path"].exists():
        ckpt = torch.load(str(cfg["ckpt_path"]), map_location="cpu", weights_only=False)
        state_dict = ckpt["model"]
        new_state = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "") if k.startswith("module.") else k
            new_state[key] = v
        model.load_state_dict(new_state)
        print(f"[ENGINE] Checkpoint loaded successfully! Step: {ckpt.get('global_step', -1)}")
    else:
        print(f"[ENGINE] WARNING: Checkpoint {cfg['ckpt_path']} not found, using initialized weights!")
    
    model = model.to(device)
    model.eval()
    active_models[preset_key] = model
    return model, cfg

# Pre-load NYU model
get_or_load_model("nyu")

def preprocess_image(image_bytes: bytes, target_h: int = 480, target_w: int = 640):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    t = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return t, img_resized, orig_w, orig_h

def apply_colormap(depth_map: np.ndarray, colormap_name: str = "plasma", max_d: float = 10.0):
    d_clipped = np.clip(depth_map, 0.001, max_d)
    norm_d = (d_clipped - d_clipped.min()) / (d_clipped.max() - d_clipped.min() + 1e-8)
    
    cmap_map = {
        "plasma": cm.plasma,
        "turbo": cm.turbo,
        "viridis": cm.viridis,
        "magma": cm.magma,
        "inferno": cm.inferno,
        "jet": cm.jet
    }
    cmap = cmap_map.get(colormap_name.lower(), cm.plasma)
    colored = (cmap(norm_d)[:, :, :3] * 255).astype(np.uint8)
    return colored

def img_to_base64(img_arr: np.ndarray, fmt: str = "JPEG") -> str:
    pil_img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

@app.get("/api/presets")
def get_presets():
    return {
        "presets": [
            {
                "id": k,
                "name": v["name"],
                "desc": v["desc"],
                "max_depth": v["max_depth"],
                "default_focal": v["default_focal"]
            }
            for k, v in PRESETS.items()
        ],
        "device": str(device)
    }

@app.get("/api/samples")
def get_samples():
    samples_dir = Path(__file__).resolve().parent / "static" / "samples"
    files = []
    if samples_dir.exists():
        for p in samples_dir.glob("*.jpg"):
            files.append({
                "filename": p.name,
                "url": f"/static/samples/{p.name}",
                "name": p.stem.replace("scene", "Bối cảnh ").replace("_", " ").title()
            })
    return {"samples": files}

@app.post("/api/predict")
async def predict_depth(
    file: UploadFile = File(...),
    preset: str = Form("nyu"),
    colormap: str = Form("plasma"),
    focal_length: Optional[float] = Form(None),
    grid_size: int = Form(160) # resolution for 3D point cloud grid (e.g. 160x120 or 240x180)
):
    start_time = time.time()
    model, cfg = get_or_load_model(preset)
    
    contents = await file.read()
    tensor, rgb_img, orig_w, orig_h = preprocess_image(contents)
    
    focal_val = focal_length if focal_length and focal_length > 10 else cfg["default_focal"]
    focal_tensor = torch.tensor([focal_val]).to(device)
    
    with torch.no_grad():
        _, _, _, _, depth = model(tensor.to(device), focal_tensor)
    
    depth_np = depth.squeeze().cpu().numpy()
    max_d = cfg["max_depth"]
    depth_np = np.clip(depth_np, 0.001, max_d)
    
    inference_time = (time.time() - start_time) * 1000.0
    
    colored_depth = apply_colormap(depth_np, colormap, max_d=max_d)
    rgb_arr = np.array(rgb_img)
    
    # Downsample for fast 3D web rendering
    h, w = depth_np.shape
    aspect = w / h
    target_gw = grid_size
    target_gh = int(grid_size / aspect)
    
    depth_small = cv2.resize(depth_np, (target_gw, target_gh), interpolation=cv2.INTER_AREA)
    rgb_small = cv2.resize(rgb_arr, (target_gw, target_gh), interpolation=cv2.INTER_AREA)
    
    rgb_b64 = img_to_base64(rgb_arr)
    depth_b64 = img_to_base64(colored_depth)
    
    stats = {
        "min_m": float(np.round(depth_np.min(), 2)),
        "max_m": float(np.round(depth_np.max(), 2)),
        "mean_m": float(np.round(depth_np.mean(), 2)),
        "median_m": float(np.round(np.median(depth_np), 2)),
        "std_m": float(np.round(depth_np.std(), 2)),
        "unit": "meters"
    }
    
    return {
        "status": "success",
        "preset": preset,
        "focal_length": focal_val,
        "inference_time_ms": round(inference_time, 1),
        "dimensions": {"width": w, "height": h, "orig_width": orig_w, "orig_height": orig_h},
        "stats": stats,
        "rgb_base64": rgb_b64,
        "depth_colormap_base64": depth_b64,
        "point_cloud_grid": {
            "width": target_gw,
            "height": target_gh,
            "depths": np.round(depth_small, 3).tolist(),
            "colors": (rgb_small.astype(np.uint8)).tolist()
        }
    }

@app.post("/api/export_ply")
async def export_ply(
    file: UploadFile = File(...),
    preset: str = Form("nyu"),
    focal_length: Optional[float] = Form(None),
    max_points: int = Form(100000)
):
    model, cfg = get_or_load_model(preset)
    contents = await file.read()
    tensor, rgb_img, _, _ = preprocess_image(contents)
    focal_val = focal_length if focal_length and focal_length > 10 else cfg["default_focal"]
    
    with torch.no_grad():
        _, _, _, _, depth = model(tensor.to(device), torch.tensor([focal_val]).to(device))
    
    depth_np = depth.squeeze().cpu().numpy()
    rgb_arr = np.array(rgb_img)
    H, W = depth_np.shape
    
    # Downsample step
    total_pix = H * W
    step = int(np.sqrt(total_pix / max_points))
    step = max(1, step)
    
    cx, cy = W / 2.0, H / 2.0
    fx, fy = focal_val, focal_val
    
    # Generate 3D coordinates
    u = np.arange(0, W, step)
    v = np.arange(0, H, step)
    uu, vv = np.meshgrid(u, v)
    
    z = depth_np[vv, uu]
    x = (uu - cx) * z / fx
    y = -(vv - cy) * z / fy # invert y for 3D graphic convention
    colors = rgb_arr[vv, uu]
    
    mask = (z > 0.001) & (z < cfg["max_depth"])
    x_valid, y_valid, z_valid = x[mask], y[mask], z[mask]
    c_valid = colors[mask]
    
    num_pts = len(x_valid)
    
    # PLY Header
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {num_pts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    
    buf = io.StringIO()
    buf.write(header)
    for i in range(num_pts):
        buf.write(f"{x_valid[i]:.4f} {y_valid[i]:.4f} {z_valid[i]:.4f} {c_valid[i,0]} {c_valid[i,1]} {c_valid[i,2]}\n")
    
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=bts_point_cloud_{preset}.ply"}
    )

# Static files
static_path = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
def serve_home():
    index_file = static_path / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>BTS Depth Estimation Demo Running</h1>")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)