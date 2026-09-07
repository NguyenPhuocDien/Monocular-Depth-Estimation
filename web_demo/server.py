import os
import sys
import io
import time
import base64
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

app = FastAPI(title="BTS Monocular Depth Estimation & Live 3D Webcam", version="2.5")

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

def find_preset_checkpoint(preset: str) -> Path:
    if preset == "kitti":
        candidates = [
            Path("E:/bts_kitti_session_08/session_8_handoff/best_abs_rel.pth"),
            Path("E:/bts_kitti_session_08/session_8_handoff/latest.pth"),
            Path("E:/bts_kitti_session_08/models/bts_kitti_densenet161_kaggle/model-242500-best_abs_rel_0.05748"),
            ROOT_DIR / "artifacts" / "checkpoints" / "bts_kitti_densenet161_step242500_best_absrel_0.05748.pth"
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]
    else:
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
        "name": "NYU Depth V2 (Indoor - Trong Nhà)",
        "desc": "Huấn luyện trên 24,231 ảnh phòng nội thất, dải đo 0.001m - 10.0m (Khuyên dùng cho Webcam)",
        "max_depth": 10.0,
        "default_focal": 519.0,
        "ckpt_path": find_preset_checkpoint("nyu"),
        "encoder": "densenet161_bts",
        "dataset": "nyu"
    },
    "kitti": {
        "id": "kitti",
        "name": "KITTI Benchmark (Outdoor - Ngoài Trời)",
        "desc": "Môi trường ngoài trời & xe tự hành, dải đo 0.001m - 80.0m (Peak Step 242,500)",
        "max_depth": 80.0,
        "default_focal": 715.0,
        "ckpt_path": find_preset_checkpoint("kitti"),
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
    ckpt_file = cfg["ckpt_path"]
    print(f"[ENGINE] Loading model preset '{preset_key}' from: {ckpt_file}...")
    
    class Args:
        encoder = cfg["encoder"]
        bts_size = 512
        max_depth = cfg["max_depth"]
        dataset = cfg["dataset"]
        num_threads = 1
        mode = "test"
    
    model = BtsModel(params=Args())
    if ckpt_file.exists():
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        new_state = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "") if k.startswith("module.") else k
            new_state[key] = v
        model.load_state_dict(new_state, strict=False)
        step_num = ckpt.get("global_step", -1) if isinstance(ckpt, dict) else -1
        print(f"[ENGINE] Successfully loaded '{preset_key}' checkpoint! Step: {step_num}")
    else:
        print(f"[ENGINE] WARNING: Checkpoint {ckpt_file} not found, using initialized weights!")
    
    model = model.to(device)
    model.eval()
    active_models[preset_key] = model
    return model, cfg

# Pre-load NYU model by default
try:
    get_or_load_model("nyu")
except Exception as e:
    print(f"[ENGINE] Preload NYU warning: {e}")

def preprocess_image(image_bytes: bytes, target_h: int = 480, target_w: int = 640):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
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
    pil_img.save(buf, format=fmt, quality=85)
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
                "default_focal": v["default_focal"],
                "is_active": k in active_models
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
            is_kitti = "kitti" in p.name.lower() or "outdoor" in p.name.lower()
            files.append({
                "filename": p.name,
                "url": f"/static/samples/{p.name}",
                "name": p.stem.replace("scene", "Bối cảnh ").replace("kitti_outdoor_street", "KITTI Phố ").replace("_", " ").title(),
                "domain": "kitti" if is_kitti else "nyu"
            })
    return {"samples": sorted(files, key=lambda x: (x["domain"] != "nyu", x["name"]))}

@app.post("/api/predict")
async def predict_depth(
    file: UploadFile = File(...),
    preset: str = Form("nyu"),
    colormap: str = Form("plasma"),
    focal_length: Optional[float] = Form(None),
    grid_size: int = Form(160)
):
    start_time = time.time()
    model, cfg = get_or_load_model(preset)
    
    contents = await file.read()
    tensor, rgb_img, orig_w, orig_h = preprocess_image(contents)
    
    focal_val = focal_length if focal_length and focal_length > 10 else cfg["default_focal"]
    focal_tensor = torch.tensor([focal_val]).to(device)
    
    with torch.inference_mode():
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

# FAST REAL-TIME WEBCAM INFERENCE ENDPOINT
@app.post("/api/predict_webcam")
async def predict_webcam(
    file: UploadFile = File(...),
    preset: str = Form("nyu"),
    colormap: str = Form("turbo"),
    focal_length: Optional[float] = Form(None)
):
    start_time = time.time()
    model, cfg = get_or_load_model(preset)
    
    contents = await file.read()
    # Fast resolution for high FPS live stream (384 x 288)
    tensor, rgb_img, orig_w, orig_h = preprocess_image(contents, target_h=288, target_w=384)
    
    focal_val = focal_length if focal_length and focal_length > 10 else cfg["default_focal"]
    focal_tensor = torch.tensor([focal_val]).to(device)
    
    with torch.inference_mode():
        _, _, _, _, depth = model(tensor.to(device), focal_tensor)
    
    depth_np = depth.squeeze().cpu().numpy()
    max_d = cfg["max_depth"]
    depth_np = np.clip(depth_np, 0.001, max_d)
    
    inference_time = (time.time() - start_time) * 1000.0
    
    colored_depth = apply_colormap(depth_np, colormap, max_d=max_d)
    depth_b64 = img_to_base64(colored_depth, fmt="JPEG")
    
    # Center pixel depth
    ch, cw = depth_np.shape[0] // 2, depth_np.shape[1] // 2
    center_depth = float(np.round(depth_np[ch, cw], 2))
    
    return {
        "status": "success",
        "preset": preset,
        "depth_b64": depth_b64,
        "center_depth_m": center_depth,
        "min_m": float(np.round(depth_np.min(), 2)),
        "max_m": float(np.round(depth_np.max(), 2)),
        "mean_m": float(np.round(depth_np.mean(), 2)),
        "latency_ms": round(inference_time, 1)
    }

@app.post("/api/export_ply")
async def export_ply(
    file: UploadFile = File(...),
    preset: str = Form("nyu"),
    focal_length: Optional[float] = Form(None),
    max_points: int = Form(50000)
):
    model, cfg = get_or_load_model(preset)
    contents = await file.read()
    tensor, rgb_img, orig_w, orig_h = preprocess_image(contents)
    
    focal_val = focal_length if focal_length and focal_length > 10 else cfg["default_focal"]
    focal_tensor = torch.tensor([focal_val]).to(device)
    
    with torch.inference_mode():
        _, _, _, _, depth = model(tensor.to(device), focal_tensor)
    
    depth_np = depth.squeeze().cpu().numpy()
    rgb_arr = np.array(rgb_img)
    
    h, w = depth_np.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u_norm = (u - w / 2.0) / focal_val
    v_norm = (v - h / 2.0) / focal_val
    
    x = u_norm * depth_np
    y = v_norm * depth_np
    z = depth_np
    
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb_arr.reshape(-1, 3)
    
    valid_mask = (points[:, 2] > 0.01) & (points[:, 2] <= cfg["max_depth"])
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    
    lines = [header]
    for (px, py, pz), (r, g, b) in zip(points, colors):
        lines.append(f"{px:.4f} {py:.4f} {pz:.4f} {int(r)} {int(g)} {int(b)}\n")
    
    ply_content = "".join(lines)
    return Response(
        content=ply_content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=bts_{preset}_pointcloud.ply"}
    )

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def read_root():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>BTS Web Demo Server Active</h1>")

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
    
    print(f"\n" + "=" * 60)
    print(f"🚀 BTS Depth Estimation & Live 3D Webcam running at:")
    print(f"👉 http://localhost:{port}")
    print(f"=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)