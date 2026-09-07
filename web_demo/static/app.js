// BTS Monocular Depth Estimation & 3D Point Cloud App
let currentFile = null;
let currentPrediction = null;
let scene, camera, renderer, controls, pointCloud, gridHelper;
let isAutoRotating = false;

// DOM Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const modelPresetSelect = document.getElementById("modelPresetSelect");
const focalInput = document.getElementById("focalInput");
const colormapSelect = document.getElementById("colormapSelect");
const colorModeSelect = document.getElementById("colorModeSelect");
const pointSizeSlider = document.getElementById("pointSizeSlider");
const pointSizeVal = document.getElementById("pointSizeVal");
const gridDensitySelect = document.getElementById("gridDensitySelect");

const rgbImg = document.getElementById("rgbImg");
const depthImg = document.getElementById("depthImg");
const depthViewport = document.getElementById("depthViewport");
const depthTooltip = document.getElementById("depthTooltip");

const minDepthVal = document.getElementById("minDepthVal");
const maxDepthVal = document.getElementById("maxDepthVal");
const meanDepthVal = document.getElementById("meanDepthVal");
const infTimeVal = document.getElementById("infTimeVal");
const inferenceSpeedPill = document.getElementById("inferenceSpeedPill");
const loadingOverlay = document.getElementById("loadingOverlay");
const samplesTrack = document.getElementById("samplesTrack");

// 1. INITIALIZE THREE.JS 3D ENVIRONMENT
function init3D() {
    const container = document.querySelector(".canvas-3d-wrapper");
    const canvas = document.getElementById("canvas3d");
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090d16);

    // Camera
    camera = new THREE.PerspectiveCamera(55, width / height, 0.05, 500);
    camera.position.set(0, 0, -0.5); // looking forward

    // Renderer
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 3.0); // look at 3m distance
    camera.position.set(0, -0.8, -0.5);
    controls.update();

    // Grid Floor
    gridHelper = new THREE.GridHelper(20, 40, 0x58a6ff, 0x21262d);
    gridHelper.position.y = -1.2;
    scene.add(gridHelper);

    // Coordinate Axes
    const axesHelper = new THREE.AxesHelper(1.0);
    axesHelper.position.set(-2, -1.2, 1.0);
    scene.add(axesHelper);

    // Animation Loop
    function animate() {
        requestAnimationFrame(animate);
        if (isAutoRotating) {
            controls.autoRotate = true;
            controls.autoRotateSpeed = 2.0;
        } else {
            controls.autoRotate = false;
        }
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // Resize listener
    window.addEventListener("resize", () => {
        if (!container) return;
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
}

// 2. BUILD 3D POINT CLOUD FROM PREDICTED DEPTH GRID
function buildPointCloud(pcData, colorMode = "rgb", pointSize = 2.5) {
    if (pointCloud) {
        scene.remove(pointCloud);
        if (pointCloud.geometry) pointCloud.geometry.dispose();
        if (pointCloud.material) pointCloud.material.dispose();
        pointCloud = null;
    }

    if (!pcData || !pcData.depths) return;

    const H = pcData.height;
    const W = pcData.width;
    const depths = pcData.depths;
    const colors = pcData.colors;
    const focal = parseFloat(focalInput.value) || 519.0;
    
    // Scale focal to grid resolution
    const fx = focal * (W / 640.0);
    const fy = focal * (H / 480.0);
    const cx = W / 2.0;
    const cy = H / 2.0;

    const positions = [];
    const colorAttrs = [];

    const minDepth = currentPrediction.stats.min_m;
    const maxDepth = currentPrediction.stats.max_m;

    for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
            const z = depths[r][c];
            if (z <= 0.05 || z >= 90.0) continue;

            const x = (c - cx) * z / fx;
            const y = -(r - cy) * z / fy; // invert Y for standard 3D coordinate system

            positions.push(x, y, z);

            if (colorMode === "rgb" && colors) {
                const rgb = colors[r][c];
                colorAttrs.push(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0);
            } else {
                // Depth Colormap (Plasma approximation)
                const normD = (z - minDepth) / (maxDepth - minDepth + 1e-6);
                const col = getPlasmaColor(normD);
                colorAttrs.push(col.r, col.g, col.b);
            }
        }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colorAttrs, 3));

    const material = new THREE.PointsMaterial({
        size: pointSize * 0.015,
        vertexColors: true,
        sizeAttenuation: true
    });

    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);

    // Reset camera target to center of room/scene
    controls.target.set(0, 0, currentPrediction.stats.mean_m || 3.0);
    controls.update();
}

function getPlasmaColor(t) {
    t = Math.max(0, Math.min(1, t));
    // Plasma RGB color ramp polynomial approximation
    const r = Math.min(1, Math.max(0, 0.05 + 2.5 * t - 1.6 * t * t));
    const g = Math.min(1, Math.max(0, 0.0 + 1.2 * t * t));
    const b = Math.min(1, Math.max(0, 0.5 + 1.2 * t - 1.8 * t * t));
    return { r, g, b };
}

// 3. API CALL & RUN PREDICTION
async function runInference(fileBlob) {
    if (!fileBlob) return;
    loadingOverlay.classList.add("active");

    const formData = new FormData();
    formData.append("file", fileBlob, "input.jpg");
    formData.append("preset", modelPresetSelect.value);
    formData.append("colormap", colormapSelect.value);
    formData.append("focal_length", focalInput.value);
    formData.append("grid_size", gridDensitySelect.value);

    try {
        const resp = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        if (!resp.ok) throw new Error("Inference failed with status " + resp.status);

        const data = await resp.json();
        currentPrediction = data;

        // Update 2D Images
        rgbImg.src = data.rgb_base64;
        depthImg.src = data.depth_colormap_base64;
        document.getElementById("rgbDim").innerText = `${data.dimensions.width} x ${data.dimensions.height}`;

        // Update Metrics
        minDepthVal.innerText = `${data.stats.min_m} m`;
        maxDepthVal.innerText = `${data.stats.max_m} m`;
        meanDepthVal.innerText = `${data.stats.mean_m} m`;
        infTimeVal.innerText = `${data.inference_time_ms} ms`;
        inferenceSpeedPill.innerText = `⏱ ${data.inference_time_ms} ms (${data.preset.toUpperCase()})`;

        // Build 3D Point Cloud
        buildPointCloud(
            data.point_cloud_grid,
            colorModeSelect.value,
            parseFloat(pointSizeSlider.value)
        );

    } catch (err) {
        alert("Lỗi khi chạy suy luận BTS: " + err.message);
        console.error(err);
    } finally {
        loadingOverlay.classList.remove("active");
    }
}

// 4. LOAD SAMPLE GALLERY
async function loadSamples() {
    try {
        const resp = await fetch("/api/samples");
        const data = await resp.json();
        samplesTrack.innerHTML = "";

        data.samples.forEach((sample, idx) => {
            const thumb = document.createElement("div");
            thumb.className = `sample-thumb ${idx === 0 ? "active" : ""}`;
            thumb.title = sample.name;
            thumb.innerHTML = `<img src="${sample.url}" alt="${sample.name}">`;

            thumb.addEventListener("click", async () => {
                document.querySelectorAll(".sample-thumb").forEach(t => t.classList.remove("active"));
                thumb.classList.add("active");
                
                // Fetch image as blob
                const imgResp = await fetch(sample.url);
                const blob = await imgResp.blob();
                currentFile = blob;
                runInference(blob);
            });

            samplesTrack.appendChild(thumb);
        });

        // Trigger first sample
        if (data.samples.length > 0) {
            const firstResp = await fetch(data.samples[0].url);
            const firstBlob = await firstResp.blob();
            currentFile = firstBlob;
            runInference(firstBlob);
        }
    } catch (e) {
        console.error("Failed to load samples:", e);
    }
}

// 5. EVENT LISTENERS
// File Upload & Drag-Drop
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
        currentFile = e.target.files[0];
        document.querySelectorAll(".sample-thumb").forEach(t => t.classList.remove("active"));
        runInference(currentFile);
    }
});

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        currentFile = e.dataTransfer.files[0];
        document.querySelectorAll(".sample-thumb").forEach(t => t.classList.remove("active"));
        runInference(currentFile);
    }
});

// Preset switcher
modelPresetSelect.addEventListener("change", () => {
    if (modelPresetSelect.value === "kitti") {
        focalInput.value = "715.0";
    } else {
        focalInput.value = "519.0";
    }
    if (currentFile) runInference(currentFile);
});

colormapSelect.addEventListener("change", () => {
    if (currentFile) runInference(currentFile);
});

gridDensitySelect.addEventListener("change", () => {
    if (currentFile) runInference(currentFile);
});

colorModeSelect.addEventListener("change", () => {
    if (currentPrediction && currentPrediction.point_cloud_grid) {
        buildPointCloud(
            currentPrediction.point_cloud_grid,
            colorModeSelect.value,
            parseFloat(pointSizeSlider.value)
        );
    }
});

pointSizeSlider.addEventListener("input", (e) => {
    pointSizeVal.innerText = e.target.value;
    if (pointCloud && pointCloud.material) {
        pointCloud.material.size = parseFloat(e.target.value) * 0.015;
    }
});

// Tabs Switching
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(c => c.style.display = "none");

        btn.classList.add("active");
        const tabId = btn.getAttribute("data-tab");
        const content = document.getElementById(tabId);
        if (content) content.style.display = "block";

        if (tabId === "3d-tab" && renderer && camera) {
            const container = document.querySelector(".canvas-3d-wrapper");
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
    });
});

// 3D Canvas Buttons
document.getElementById("resetCamBtn").addEventListener("click", () => {
    if (controls && currentPrediction) {
        camera.position.set(0, -0.8, -0.5);
        controls.target.set(0, 0, currentPrediction.stats.mean_m || 3.0);
        controls.update();
    }
});

document.getElementById("autoRotateBtn").addEventListener("click", () => {
    isAutoRotating = !isAutoRotating;
    document.getElementById("autoRotateBtn").style.borderColor = isAutoRotating ? "var(--accent-blue)" : "var(--border)";
});

document.getElementById("toggleGridBtn").addEventListener("click", () => {
    if (gridHelper) gridHelper.visible = !gridHelper.visible;
});

// Interactive 2D Hover Depth Tooltip
depthViewport.addEventListener("mousemove", (e) => {
    if (!currentPrediction || !currentPrediction.point_cloud_grid) return;
    const rect = depthImg.getBoundingClientRect();
    if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) {
        depthTooltip.style.display = "none";
        return;
    }

    const normX = (e.clientX - rect.left) / rect.width;
    const normY = (e.clientY - rect.top) / rect.height;

    const grid = currentPrediction.point_cloud_grid;
    const col = Math.floor(normX * grid.width);
    const row = Math.floor(normY * grid.height);

    if (row >= 0 && row < grid.height && col >= 0 && col < grid.width) {
        const d = grid.depths[row][col];
        depthTooltip.style.display = "block";
        depthTooltip.style.left = `${e.clientX - depthViewport.getBoundingClientRect().left}px`;
        depthTooltip.style.top = `${e.clientY - depthViewport.getBoundingClientRect().top}px`;
        depthTooltip.innerText = `Khoảng cách: ${d.toFixed(2)} m`;
    }
});

depthViewport.addEventListener("mouseleave", () => {
    depthTooltip.style.display = "none";
});

// Download depth map image
document.getElementById("downloadDepthBtn").addEventListener("click", () => {
    if (!depthImg.src) return;
    const a = document.createElement("a");
    a.href = depthImg.src;
    a.download = `bts_depth_map_${modelPresetSelect.value}.jpg`;
    a.click();
});

// Download .PLY Point Cloud
document.getElementById("downloadPlyBtn").addEventListener("click", async () => {
    if (!currentFile) {
        alert("Vui lòng chọn hoặc tải ảnh lên trước!");
        return;
    }
    const formData = new FormData();
    formData.append("file", currentFile);
    formData.append("preset", modelPresetSelect.value);
    formData.append("focal_length", focalInput.value);

    loadingOverlay.classList.add("active");
    document.getElementById("loadingSubtitle").innerText = "Đang xuất định dạng 3D Point Cloud (.PLY)...";

    try {
        const resp = await fetch("/api/export_ply", {
            method: "POST",
            body: formData
        });
        const blob = await resp.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `bts_point_cloud_${modelPresetSelect.value}.ply`;
        a.click();
        window.URL.revokeObjectURL(url);
    } catch (e) {
        alert("Lỗi khi tải PLY: " + e.message);
    } finally {
        loadingOverlay.classList.remove("active");
        document.getElementById("loadingSubtitle").innerText = "Xử lý mạng nơ-ron DenseNet161 + Local Planar Guidance";
    }
});

// App Startup
window.addEventListener("DOMContentLoaded", () => {
    init3D();
    loadSamples();
});