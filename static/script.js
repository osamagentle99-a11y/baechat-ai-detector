const form = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const resultImage = document.getElementById("resultImage");
const statusText = document.getElementById("status");
const objectList = document.getElementById("objectList");

// ========================================
// IMAGE UPLOAD AND OBJECT DETECTION
// ========================================

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.hidden = false;
        resultImage.hidden = true;
        objectList.replaceChildren();
        statusText.textContent = "Image selected. Ready to detect.";
    }
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = imageInput.files[0];
    if (!file) {
        statusText.textContent = "Please select an image first.";
        return;
    }
    const data = new FormData();
    data.append("image", file);
    const button = form.querySelector("button");
    button.disabled = true;
    statusText.textContent = "AI is detecting objects...";
    objectList.replaceChildren();
    resultImage.hidden = true;
    try {
        const response = await fetch("/detect", { method: "POST", body: data });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Detection failed.");
        const objects = result.objects || [];
        statusText.textContent = `Detection complete: ${objects.length} object(s) found.`;
        if (result.image_url) {
            resultImage.src = result.image_url + "?t=" + Date.now();
            resultImage.hidden = false;
            resultImage.alt = "Image with detected objects";
        }
        objects.forEach((object) => {
            const item = document.createElement("li");
            item.textContent = `${object.name} — ${object.confidence}% confidence`;
            objectList.appendChild(item);
        });
        if (objects.length === 0) {
            statusText.textContent = "No objects detected. Try another image.";
        }
    } catch (error) {
        statusText.textContent = error.message;
        console.error("Image detection error:", error);
    } finally {
        button.disabled = false;
    }
});

// ========================================
// LIVE CAMERA ELEMENTS
// ========================================

const camera = document.getElementById("camera");
const overlay = document.getElementById("cameraOverlay");
const startCamera = document.getElementById("startCamera");
const stopCamera = document.getElementById("stopCamera");
const cameraStatus = document.getElementById("cameraStatus");
const cameraObjects = document.getElementById("cameraObjects");

let cameraStream = null;
let cameraRunning = false;
let detectingFrame = false;
let smoothedFace = null;

const SMOOTHING = 0.25;
const captureCanvas = document.createElement("canvas");

// ========================================
// START CAMERA
// ========================================

startCamera.addEventListener("click", async () => {
    try {
        cameraStatus.textContent = "Opening camera...";
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" }, audio: false
        });
        camera.srcObject = cameraStream;
        await camera.play();
        overlay.width = camera.videoWidth;
        overlay.height = camera.videoHeight;
        cameraRunning = true;
        smoothedFace = null;
        startCamera.disabled = true;
        stopCamera.disabled = false;
        cameraStatus.textContent = "Camera is running. Detecting...";
        detectCameraFrame();
    } catch (error) {
        console.error("Camera error:", error);
        cameraStatus.textContent = `Camera access failed: ${error.message}`;
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
    }
});

// ========================================
// STOP CAMERA
// ========================================

stopCamera.addEventListener("click", () => {
    cameraRunning = false;
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    camera.srcObject = null;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    cameraObjects.replaceChildren();
    smoothedFace = null;
    cameraStatus.textContent = "Camera is stopped.";
    startCamera.disabled = false;
    stopCamera.disabled = true;
});

// ========================================
// LIVE FRAME DETECTION - FIXED VERSION
// ========================================

async function detectCameraFrame() {
    if (!cameraRunning || detectingFrame) return;
    detectingFrame = true;
    try {
        const width = camera.videoWidth;
        const height = camera.videoHeight;
        if (!width || !height) return;

        captureCanvas.width = width;
        captureCanvas.height = height;
        const captureContext = captureCanvas.getContext("2d");
        captureContext.drawImage(camera, 0, 0, width, height);

        const blob = await new Promise(resolve => {
            captureCanvas.toBlob(resolve, "image/jpeg", 0.75);
        });
        if (!blob || !cameraRunning) return;

        const formData = new FormData();
        formData.append("frame", blob, "camera.jpg");

        const response = await fetch("/detect-frame", { method: "POST", body: formData });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Camera detection failed.");
        if (!cameraRunning) return;

        const ctx = overlay.getContext("2d");
        if (overlay.width !== width || overlay.height !== height) {
            overlay.width = width;
            overlay.height = height;
        }
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        cameraObjects.replaceChildren();

        const seen = new Map();
        const objects = result.objects || [];

        // OBJECT BOXES (purple)
        objects.forEach(object => {
            if (!Array.isArray(object.box) || object.box.length < 4) return;
            const [x1, y1, x2, y2] = object.box;
            const boxWidth = x2 - x1;
            const boxHeight = y2 - y1;
            ctx.strokeStyle = "#a78bfa";
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, boxWidth, boxHeight);
            const label = `${object.name} ${object.confidence}%`;
            ctx.font = "bold 16px Arial";
            const labelWidth = ctx.measureText(label).width + 12;
            const labelY = Math.max(0, y1 - 25);
            ctx.fillStyle = "#7c3aed";
            ctx.fillRect(x1, labelY, labelWidth, 25);
            ctx.fillStyle = "#ffffff";
            ctx.fillText(label, x1 + 6, labelY + 17);
            seen.set(object.name, (seen.get(object.name) || 0) + 1);
        });

        // --- FIXED: DRAW MATCHED FACE ONLY AROUND FACE ---
        const matchedFaces = result.matched_faces || [];
        if (matchedFaces.length > 0) {
            const face = matchedFaces[0];
            const target = {
                x: Number(face.x), y: Number(face.y),
                w: Number(face.w), h: Number(face.h)
            };
            if (Number.isFinite(target.x) && Number.isFinite(target.y) && Number.isFinite(target.w) && Number.isFinite(target.h) && target.w > 0 && target.h > 0) {
                if (!smoothedFace) {
                    smoothedFace = { ...target };
                } else {
                    smoothedFace.x += (target.x - smoothedFace.x) * SMOOTHING;
                    smoothedFace.y += (target.y - smoothedFace.y) * SMOOTHING;
                    smoothedFace.w += (target.w - smoothedFace.w) * SMOOTHING;
                    smoothedFace.h += (target.h - smoothedFace.h) * SMOOTHING;
                }
                const { x, y, w, h } = smoothedFace;

                // RECTANGLE ONLY ON FACE - NOT FULL SCREEN
                ctx.strokeStyle = "#ff0000";
                ctx.lineWidth = 3;
                ctx.strokeRect(x, y, w, h);

                // Label on top of face
                const personName = face.name || "osama";
                const faceLabel = `MATCH FOUND: ${personName} ${face.similarity ? '(' + Math.round(face.similarity) + '%)' : ''}`;
                ctx.font = "bold 16px Arial";
                const textWidth = ctx.measureText(faceLabel).width + 14;
                const labelY = Math.max(0, y - 26);
                ctx.fillStyle = "#ff0000";
                ctx.fillRect(x, labelY, textWidth, 22);
                ctx.fillStyle = "#ffffff";
                ctx.fillText(faceLabel, x + 7, labelY + 16);
            }
        } else {
            smoothedFace = null;
        }

        seen.forEach((count, name) => {
            const item = document.createElement("li");
            item.textContent = `${name}: ${count}`;
            cameraObjects.appendChild(item);
        });

        if (matchedFaces.length > 0) {
            const name = matchedFaces[0].name || "osama";
            cameraStatus.textContent = `MATCH FOUND: ${name} | ${objects.length} object(s) detected.`;
        } else {
            cameraStatus.textContent = `Detected ${objects.length} object(s). No matching face found.`;
        }

    } catch (error) {
        if (cameraRunning) {
            cameraStatus.textContent = `Detection error: ${error.message}`;
            console.error("Live detection error:", error);
        }
    } finally {
        detectingFrame = false;
        if (cameraRunning) setTimeout(detectCameraFrame, 250);
    }
}

// ========================================
// REFERENCE PHOTO UPLOAD
// ========================================

const referenceForm = document.getElementById("referenceForm");
const referenceImage = document.getElementById("referenceImage");
const referenceStatus = document.getElementById("referenceStatus");

referenceForm.addEventListener("submit", async event => {
    event.preventDefault();
    if (!referenceImage.files.length) {
        referenceStatus.textContent = "Please select a reference photo.";
        return;
    }
    const formData = new FormData();
    formData.append("image", referenceImage.files[0]);
    referenceStatus.textContent = "Uploading reference photo...";
    try {
        const response = await fetch("/upload-reference", { method: "POST", body: formData });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Reference upload failed.");
        referenceStatus.textContent = result.message || "Reference photo uploaded successfully.";
    } catch (error) {
        referenceStatus.textContent = error.message;
        console.error("Reference upload error:", error);
    }
});