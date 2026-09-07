// BTS Monocular Depth Estimation & Live 3D Webcam Application
let currentFile = null;
let currentPrediction = null;
let scene, camera, renderer, controls, pointCloud, gridHelper;
let isAutoRotating = false;
let allSamples = [];

// Webcam State
let webcamStream = null;
let isWebcamRunning = false;
let webcamProcessing = false;
let lastFrameTime = performance.now();
let frameCount = 0;
let fpsTimer = performance.now();

// DOM Elements - Navigation & Presets
const modelPresetSelect = document.getElementById("modelPresetSelect");
const btnPresetNyu = document.getElementById("btnPresetNyu");
const btnPresetKitti = document.getElementById("btnPresetKitti");
const activeDomainText = document.getElementById("activeDomainText");
const ckptFileText = document.getElementById("ckptFileText");
const ckptMetricText = document.getElementById("ckptMetricText");
const ckptRangeText = document.getElementById("ckptRangeText");
const focalInput = document.getElementById("focalInput");
const focalHint = document.getElementById("focalHint");
const colormapSelect = document.getElementById("colormapSelect");
const colorModeSelect = document.getElementById("colorModeSelect");
const pointSizeSlider = document.getElementById("pointSizeSlider");
const pointSizeVal = document.getElementById("pointSizeVal");
const gridDensitySelect = document.getElementById("gridDensitySelect");

// DOM Elements - Upload & 2D
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const rgbImg = document.getElementById("rgbImg");
const depthImg = document.getElementById("depthImg");
const depthViewport = document.getElementById("depthViewport");
const depthTooltip = document.getElementById("depthTooltip");

// DOM Elements - Metrics
const minDepthVal = document.getElementById("minDepthVal");
const maxDepthVal = document.getElementById("maxDepthVal");
const meanDepthVal = document.getElementById("meanDepthVal");
const infTimeVal = document.getElementById("infTimeVal");
const inferenceSpeedPill = document.getElementById("inferenceSpeedPill");
const loadingOverlay = document.getElementById("loadingOverlay");
const samplesTrack = document.getElementById("samplesTrack");

// DOM Elements - Webcam
const btnStartWebcam = document.getElementById("btnStartWebcam");
const btnStopWebcam = document.getElementById("btnStopWebcam");
const btnFreeze3D = document.getElementById("btnFreeze3D");
const webcamVideo = document.getElementById("webcamVideo");
const webcamHiddenCanvas = document.getElementById("webcamHiddenCanvas");
const webcamDepthImg = document.getElementById("webcamDepthImg");
const webcamPlaceholder = document.getElementById("webcamPlaceholder");
const depthPlaceholder = document.getElementById("depthPlaceholder");
const webcamCrosshair = document.getElementById("webcamCrosshair");
const webcamDepthCrosshair = document.getElementById("webcamDepthCrosshair");
const crosshairDistLabel = document.getElementById("crosshairDistLabel");
const webcamResBadge = document.getElementById("webcamResBadge");
const webcamFps = document.getElementById("webcamFps");
const webcamLatency = document.getElementById("webcamLatency");
const webcamCenterDist = document.getElementById("webcamCenterDist");
const tabBtn3D = document.getElementById("tabBtn3D");

// 1. THREE.JS 3D INITIALIZATION
function init3D() {
    const container = document.querySelector(".canvas-3d-wrapper");
    const canvas = document.getElementById("canvas3d");
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 540;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090d16);

    camera = new THREE.PerspectiveCamera(55, width / height, 0.05, 500);
    camera.position.set(0, -0.6, -0.4);

    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 3.0);
    controls.update();

    gridHelper = new THREE.GridHelper(20, 40, 0x58a6ff, 0x21262d);
    gridHelper.position.y = -1.2;
    scene.add(gridHelper);

    const axesHelper = new THREE.AxesHelper(1.0);
    axesHelper.position.set(-2, -1.2, 1.0);
    scene.add(axesHelper);

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

    window.addEventListener("resize", () => {
        if (!container || !renderer || !camera) return;
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
}

// 2. BUILD 3D POINT CLOUD FROM DEPTH GRID
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
    const focal = parseFloat(focalInput.value) || (modelPresetSelect.value === "kitti" ? 715.0 : 519.0);

    const fx = focal * (W / 640.0);
    const fy = focal * (H / 480.0);
    const cx = W / 2.0;
    const cy = H / 2.0;

    const positions = [];
    const colorAttrs = [];

    const minDepth = currentPrediction.stats ? currentPrediction.stats.min_m : 0.1;
    const maxDepth = currentPrediction.stats ? currentPrediction.stats.max_m : 10.0;
    const maxAllowed = modelPresetSelect.value === "kitti" ? 85.0 : 12.0;

    for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
            const z = depths[r][c];
            if (z <= 0.05 || z >= maxAllowed) continue;

            const x = (c - cx) * z / fx;
            const y = -(r - cy) * z / fy;

            positions.push(x, y, z);

            if (colorMode === "rgb" && colors) {
                const rgb = colors[r][c];
                colorAttrs.push(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0);
            } else {
                const normD = (z - minDepth) / (maxDepth - minDepth + 1e-6);
                const col = getTurboColor(normD);
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

    const targetZ = (currentPrediction.stats && currentPrediction.stats.mean_m) ? currentPrediction.stats.mean_m : 3.0;
    controls.target.set(0, 0, Math.min(targetZ, 15.0));
    controls.update();
}

function getTurboColor(t) {
    t = Math.max(0, Math.min(1, t));
    // Turbo RGB color map approximation
    const r = Math.sin((t - 0.25) * Math.PI) * 0.5 + 0.5;
    const g = Math.sin((t - 0.5) * Math.PI) * 0.5 + 0.5;
    const b = Math.sin((t - 0.75) * Math.PI) * 0.5 + 0.5;
    return { r, g, b };
}

// 3. INFERENCE API CALL (HIGH RESOLUTION & 3D GRID)
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

        if (!resp.ok) throw new Error("Inference failed: " + resp.status);

        const data = await resp.json();
        currentPrediction = data;

        rgbImg.src = data.rgb_base64;
        depthImg.src = data.depth_colormap_base64;
        document.getElementById("rgbDim").innerText = `${data.dimensions.width} x ${data.dimensions.height}`;

        minDepthVal.innerText = `${data.stats.min_m} m`;
        maxDepthVal.innerText = `${data.stats.max_m} m`;
        meanDepthVal.innerText = `${data.stats.mean_m} m`;
        infTimeVal.innerText = `${data.inference_time_ms} ms`;
        inferenceSpeedPill.innerText = `⏱ ${data.inference_time_ms} ms (${data.preset.toUpperCase()})`;

        buildPointCloud(
            data.point_cloud_grid,
            colorModeSelect.value,
            parseFloat(pointSizeSlider.value)
        );

    } catch (err) {
        alert("Lỗi khi suy luận BTS: " + err.message);
        console.error(err);
    } finally {
        loadingOverlay.classList.remove("active");
    }
}

// 4. PRESET SWITCHING (NYU INDOOR VS KITTI OUTDOOR)
function setPreset(preset) {
    modelPresetSelect.value = preset;
    if (preset === "kitti") {
        btnPresetKitti.classList.add("active");
        btnPresetNyu.classList.remove("active");
        activeDomainText.innerText = "KITTI (Ngoài Trời • 80m)";
        ckptFileText.innerText = "bts_kitti_session_08_best";
        ckptMetricText.innerText = "0.05748 (AbsRel Peak)";
        ckptRangeText.innerText = "80.0 mét";
        focalInput.value = "715.0";
        focalHint.innerText = "Khuyến nghị fx=715 cho xe tự hành KITTI";
        if (gridHelper) {
            gridHelper.scale.set(4, 1, 4);
            gridHelper.position.y = -1.6;
        }
    } else {
        btnPresetNyu.classList.add("active");
        btnPresetKitti.classList.remove("active");
        activeDomainText.innerText = "NYUv2 (Trong Nhà • 10m)";
        ckptFileText.innerText = "nyu_densenet161_step271k";
        ckptMetricText.innerText = "0.10967 (AbsRel Peak)";
        ckptRangeText.innerText = "10.0 mét";
        focalInput.value = "519.0";
        focalHint.innerText = "Khuyến nghị fx=519 cho phòng nội thất / Webcam";
        if (gridHelper) {
            gridHelper.scale.set(1, 1, 1);
            gridHelper.position.y = -1.2;
        }
    }

    // Filter samples display matching preset
    filterSamples(preset);

    if (currentFile) {
        runInference(currentFile);
    }
}

btnPresetNyu.addEventListener("click", () => setPreset("nyu"));
btnPresetKitti.addEventListener("click", () => setPreset("kitti"));

// 5. LIVE WEBCAM ENGINE
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });

        webcamStream = stream;
        webcamVideo.srcObject = stream;
        webcamVideo.style.display = "block";
        webcamDepthImg.style.display = "block";
        webcamPlaceholder.style.display = "none";
        depthPlaceholder.style.display = "none";
        webcamCrosshair.style.display = "block";
        webcamDepthCrosshair.style.display = "block";

        btnStartWebcam.style.display = "none";
        btnStopWebcam.style.display = "inline-flex";

        isWebcamRunning = true;
        webcamProcessing = false;
        frameCount = 0;
        fpsTimer = performance.now();

        webcamVideo.onloadedmetadata = () => {
            webcamResBadge.innerText = `${webcamVideo.videoWidth} x ${webcamVideo.videoHeight} Live`;
            runWebcamLoop();
        };

    } catch (err) {
        alert("Không thể truy cập Webcam máy tính: " + err.message + "\nVui lòng cho phép quyền Camera trên trình duyệt!");
        console.error("Webcam Error:", err);
    }
}

function stopWebcam() {
    isWebcamRunning = false;
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    webcamVideo.style.display = "none";
    webcamDepthImg.style.display = "none";
    webcamPlaceholder.style.display = "flex";
    depthPlaceholder.style.display = "flex";
    webcamCrosshair.style.display = "none";
    webcamDepthCrosshair.style.display = "none";

    btnStartWebcam.style.display = "inline-flex";
    btnStopWebcam.style.display = "none";
    webcamResBadge.innerText = "Camera Off";
    webcamFps.innerText = "0.0";
    webcamLatency.innerText = "0 ms";
    webcamCenterDist.innerText = "-- m";
}

async function runWebcamLoop() {
    if (!isWebcamRunning) return;

    if (!webcamProcessing && webcamVideo.readyState >= 2) {
        webcamProcessing = true;

        // Capture frame at 384x288 onto hidden canvas
        webcamHiddenCanvas.width = 384;
        webcamHiddenCanvas.height = 288;
        const ctx = webcamHiddenCanvas.getContext("2d");
        
        // Draw mirrored image to canvas
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(webcamVideo, -384, 0, 384, 288);
        ctx.restore();

        webcamHiddenCanvas.toBlob(async (blob) => {
            if (!blob || !isWebcamRunning) {
                webcamProcessing = false;
                if (isWebcamRunning) requestAnimationFrame(runWebcamLoop);
                return;
            }

            const formData = new FormData();
            formData.append("file", blob, "frame.jpg");
            formData.append("preset", modelPresetSelect.value);
            formData.append("colormap", colormapSelect.value);
            formData.append("focal_length", focalInput.value);

            try {
                const resp = await fetch("/api/predict_webcam", {
                    method: "POST",
                    body: formData
                });

                if (resp.ok && isWebcamRunning) {
                    const data = await resp.json();
                    webcamDepthImg.src = data.depth_b64;

                    // Update metrics
                    webcamLatency.innerText = `${data.latency_ms} ms`;
                    webcamCenterDist.innerText = `${data.center_depth_m} m`;
                    crosshairDistLabel.innerText = `${data.center_depth_m} m`;

                    minDepthVal.innerText = `${data.min_m} m`;
                    maxDepthVal.innerText = `${data.max_m} m`;
                    meanDepthVal.innerText = `${data.mean_m} m`;
                    infTimeVal.innerText = `${data.latency_ms} ms`;

                    // Calculate live FPS
                    frameCount++;
                    const now = performance.now();
                    const elapsed = now - fpsTimer;
                    if (elapsed >= 1000) {
                        const fps = (frameCount * 1000) / elapsed;
                        webcamFps.innerText = fps.toFixed(1);
                        frameCount = 0;
                        fpsTimer = now;
                    }
                }
            } catch (e) {
                console.warn("Webcam frame error:", e);
            } finally {
                webcamProcessing = false;
                if (isWebcamRunning) {
                    // Smooth scheduling
                    setTimeout(runWebcamLoop, 15);
                }
            }
        }, "image/jpeg", 0.75);
    } else {
        if (isWebcamRunning) {
            requestAnimationFrame(runWebcamLoop);
        }
    }
}

// Freeze frame from webcam & build 3D Point Cloud
btnFreeze3D.addEventListener("click", () => {
    if (!isWebcamRunning || webcamVideo.readyState < 2) {
        alert("Vui lòng BẬT WEBCAM trước khi đóng băng khung hình!");
        return;
    }

    // Grab full resolution 640x480 frame
    const freezeCanvas = document.createElement("canvas");
    freezeCanvas.width = webcamVideo.videoWidth || 640;
    freezeCanvas.height = webcamVideo.videoHeight || 480;
    const ctx = freezeCanvas.getContext("2d");

    // Mirror to match preview
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(webcamVideo, -freezeCanvas.width, 0, freezeCanvas.width, freezeCanvas.height);
    ctx.restore();

    freezeCanvas.toBlob(async (blob) => {
        currentFile = blob;
        // Run full prediction and generate 3D point cloud
        await runInference(blob);
        // Switch to 3D tab
        tabBtn3D.click();
    }, "image/jpeg", 0.95);
});

btnStartWebcam.addEventListener("click", startWebcam);
btnStopWebcam.addEventListener("click", stopWebcam);

// 6. SAMPLE GALLERY MANAGEMENT
async function loadSamples() {
    try {
        const resp = await fetch("/api/samples");
        const data = await resp.json();
        allSamples = data.samples || [];
        renderSamples(allSamples);

        // Auto run first sample
        if (allSamples.length > 0) {
            const first = allSamples[0];
            const imgResp = await fetch(first.url);
            const blob = await imgResp.blob();
            currentFile = blob;
            runInference(blob);
        }
    } catch (e) {
        console.error("Failed to load samples:", e);
    }
}

function renderSamples(samples) {
    samplesTrack.innerHTML = "";
    samples.forEach((sample, idx) => {
        const thumb = document.createElement("div");
        thumb.className = `sample-thumb ${idx === 0 ? "active" : ""}`;
        thumb.title = sample.name;
        thumb.innerHTML = `
            <img src="${sample.url}" alt="${sample.name}">
            <span class="sample-domain-tag">${sample.domain === "kitti" ? "KITTI" : "NYU"}</span>
        `;

        thumb.addEventListener("click", async () => {
            document.querySelectorAll(".sample-thumb").forEach(t => t.classList.remove("active"));
            thumb.classList.add("active");

            // Auto switch preset if clicking domain sample
            if (sample.domain === "kitti" && modelPresetSelect.value !== "kitti") {
                setPreset("kitti");
            } else if (sample.domain === "nyu" && modelPresetSelect.value !== "nyu") {
                setPreset("nyu");
            }

            const imgResp = await fetch(sample.url);
            const blob = await imgResp.blob();
            currentFile = blob;
            runInference(blob);
        });

        samplesTrack.appendChild(thumb);
    });
}

function filterSamples(filterType) {
    document.querySelectorAll(".filter-chip").forEach(chip => {
        if (chip.getAttribute("data-filter") === filterType) {
            chip.classList.add("active");
        } else {
            chip.classList.remove("active");
        }
    });

    if (filterType === "all") {
        renderSamples(allSamples);
    } else {
        const filtered = allSamples.filter(s => s.domain === filterType);
        renderSamples(filtered);
    }
}

document.querySelectorAll(".filter-chip").forEach(chip => {
    chip.addEventListener("click", () => {
        const filter = chip.getAttribute("data-filter");
        filterSamples(filter);
    });
});

// 7. FILE UPLOAD & DRAG DROP
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

// 8. CONTROLS CHANGE LISTENERS
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

// 9. TAB SWITCHING
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
        camera.position.set(0, -0.6, -0.4);
        controls.target.set(0, 0, (currentPrediction.stats && currentPrediction.stats.mean_m) ? currentPrediction.stats.mean_m : 3.0);
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
    a.download = `bts_depth_${modelPresetSelect.value}.jpg`;
    a.click();
});

// Download .PLY Point Cloud
document.getElementById("downloadPlyBtn").addEventListener("click", async () => {
    if (!currentFile) {
        alert("Vui lòng tải ảnh lên hoặc đóng băng khung hình trước!");
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
        a.download = `bts_pointcloud_${modelPresetSelect.value}.ply`;
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