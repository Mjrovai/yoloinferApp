import * as inf from './inference.js';

const video          = document.getElementById('video');
const staticImg      = document.getElementById('static-img');
const overlay        = document.getElementById('overlay');
const ctx            = overlay.getContext('2d');
const placeholder    = document.getElementById('placeholder');
const loader         = document.getElementById('loader');
const loaderMsg      = document.getElementById('loader-msg');
const errorScreen    = document.getElementById('error-screen');
const errorMsg       = document.getElementById('error-msg');
const fpsBadge       = document.getElementById('fps-badge');
const modelBadge     = document.getElementById('model-badge');
const btnCamera      = document.getElementById('btn-camera');
const btnUpload      = document.getElementById('btn-upload');
const btnModel       = document.getElementById('btn-model');
const btnCapture     = document.getElementById('btn-capture');
const fileInput      = document.getElementById('file-input');
const modelInput     = document.getElementById('model-input');
const modelSelect    = document.getElementById('model-select');
const confSlider     = document.getElementById('conf-slider');
const confVal        = document.getElementById('conf-val');
const iouSlider      = document.getElementById('iou-slider');
const iouVal         = document.getElementById('iou-val');
const bboxToggleRow  = document.getElementById('bbox-toggle-row');
const bboxToggle     = document.getElementById('bbox-toggle');
const viewport       = document.getElementById('viewport');
const btnTheme       = document.getElementById('btn-theme');

let session            = null;
let cameraRunning      = false;
let inferenceRunning   = false;
let rafId              = null;
let fpsSmooth          = 0;
let lastInfStart       = 0;
let displayedW         = 1;
let displayedH         = 1;
let offsetX            = 0;
let offsetY            = 0;
let currentModelName   = 'detection';

function updateDisplayMetrics() {
  const state = inf.getLastPreprocessState();
  const srcW = state.srcW;
  const srcH = state.srcH;
  const canvasW = overlay.width;
  const canvasH = overlay.height;
  const srcAspect = srcW / srcH;
  const canvasAspect = canvasW / canvasH;

  if (srcAspect > canvasAspect) {
    displayedW = canvasW;
    displayedH = canvasW / srcAspect;
  } else {
    displayedH = canvasH;
    displayedW = canvasH * srcAspect;
  }

  offsetX = (canvasW - displayedW) / 2;
  offsetY = (canvasH - displayedH) / 2;

  console.log(`updateDisplayMetrics: src=(${srcW}x${srcH}) canvas=(${canvasW}x${canvasH}) displayed=(${displayedW.toFixed(1)}x${displayedH.toFixed(1)}) offset=(${offsetX.toFixed(1)}, ${offsetY.toFixed(1)})`);
}

function toCanvasCoords(x, y) {
  const state = inf.getLastPreprocessState();
  const srcW = state.srcW;
  const srcH = state.srcH;
  const canvasW = overlay.width;
  const canvasH = overlay.height;
  const srcAspect = srcW / srcH;
  const canvasAspect = canvasW / canvasH;

  let dW, dH, oX, oY;
  if (srcAspect > canvasAspect) {
    dW = canvasW;
    dH = canvasW / srcAspect;
  } else {
    dH = canvasH;
    dW = canvasH * srcAspect;
  }
  oX = (canvasW - dW) / 2;
  oY = (canvasH - dH) / 2;

  // Letterbox state uses { scale, padX, padY }; crop state uses { sx, sy, sw, sh }
  let srcX, srcY;
  if (state.scale !== undefined) {
    srcX = (x - state.padX) / state.scale;
    srcY = (y - state.padY) / state.scale;
  } else {
    srcX = state.sx + x * (state.sw / 192);
    srcY = state.sy + y * (state.sh / 192);
  }

  return {
    x: oX + srcX * (dW / srcW),
    y: oY + srcY * (dH / srcH),
  };
}

// ── Theme ─────────────────────────────────────────────────

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  btnTheme.textContent = theme === 'dark' ? '☀' : '🌙';
  localStorage.setItem('yoloinfer-theme', theme);
}

btnTheme.addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
});

// ─────────────────────────────────────────────────────────

inf.setConf(0.35);
inf.setIou(0.45);
updateConfDisplay();
updateIouDisplay();

btnUpload.querySelector('span').textContent = '📤 Upload';
btnModel.querySelector('span').textContent = '🤖 Model';

const COLORS = ['#ff1744', '#ffd600', '#00e676', '#00b0ff', '#d500f9'];
const CLASS_NAMES = ['box', 'wheel'];
const POSE_NAMES  = ['person'];
const COCO_NAMES  = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];
const BUILTIN_MODELS = {
  detection:    '/models/yolov8n.onnx',
  pose:         '/models/yolov8n-pose.onnx',
  segmentation: '/models/yolov8n-seg.onnx',
};
const MODEL_LABELS = { detection: 'Detection', pose: 'Pose', segmentation: 'Segmentation' };

let showBboxes = true;
let normalizedDetCoords = true; // true = custom model (0-1 → ×192); false = standard YOLO (pixel)
let detectionClassNames = CLASS_NAMES; // updated at load time based on detected numClasses

function hexToRgb(hex) {
  return [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)];
}

function getClassName(classId) {
  const mk = modelSelect.value;
  if (mk === 'pose') return POSE_NAMES[classId] ?? 'c' + classId;
  if (mk === 'segmentation') return COCO_NAMES[classId] ?? 'c' + classId;
  return detectionClassNames[classId] ?? 'c' + classId;
}
const kptPairs = [
    [0,1],[0,2],[1,3],[2,4],          // face
    [5,6],                             // shoulder–shoulder
    [5,7],[6,8],[7,9],[8,10],         // arms
    [5,11],[6,12],[11,12],            // torso + hip–hip
    [11,13],[12,14],[13,15],[14,16],  // legs
];

function updateConfDisplay() {
  inf.setConf(parseFloat(confSlider.value) / 100);
  confVal.textContent = (parseFloat(confSlider.value) / 100).toFixed(2);
}

function updateIouDisplay() {
  inf.setIou(parseFloat(iouSlider.value) / 100);
  iouVal.textContent = (parseFloat(iouSlider.value) / 100).toFixed(2);
}

confSlider.addEventListener('input', updateConfDisplay);
iouSlider.addEventListener('input', updateIouDisplay);

confSlider.addEventListener('change', () => {
  if (!staticImg.hidden) runStaticImage();
});

iouSlider.addEventListener('change', () => {
  if (!staticImg.hidden) runStaticImage();
});

function resizeCanvas() {
  const rect = overlay.getBoundingClientRect();
  overlay.width    = Math.max(1, Math.round(rect.width));
  overlay.height   = Math.max(1, Math.round(rect.height));
  updateDisplayMetrics();
}

function updateModelBadge(modelType) {
  let filename;
  if (BUILTIN_MODELS[currentModelName]) {
    filename = BUILTIN_MODELS[currentModelName].split('/').pop();
  } else {
    filename = currentModelName + '.onnx';
  }
  const typeLabel = MODEL_LABELS[modelType] || modelType;
  modelBadge.textContent = filename + '  ·  ' + typeLabel;
  modelBadge.hidden = false;
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'environment',
      width:  { ideal: 1280 },
      height: { ideal: 720  },
    },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise(r => { video.onloadedmetadata = r; });
  await video.play();
  // Match viewport to camera's actual aspect ratio so the feed fills the box
  if (video.videoWidth && video.videoHeight) {
    viewport.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  }
  viewport.classList.add('camera-live');
}

function stopCamera() {
  cameraRunning  = false;
  inferenceRunning = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  video.hidden = true;
  fpsBadge.hidden = true;
  btnCamera.textContent = '📷 Start';
  btnCamera.classList.remove('running');
  btnCapture.classList.remove('active');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  viewport.style.aspectRatio = '';
  viewport.classList.remove('camera-live');
}

async function runFrame() {
  if (!cameraRunning) return;
  if (inferenceRunning) {
    rafId = requestAnimationFrame(runFrame);
    return;
  }
  inferenceRunning = true;
  const t0 = performance.now();
  try {
    const results = await doInference(video);
    drawResults(results);
    if (lastInfStart > 0) {
      const dt = t0 - lastInfStart;
      fpsSmooth = fpsSmooth * 0.85 + (1000 / dt) * 0.15;
      fpsBadge.textContent = Math.round(fpsSmooth) + ' FPS';
    }
    lastInfStart = t0;
  } catch (e) {
    console.error(e);
  }
  inferenceRunning = false;
  if (cameraRunning) rafId = requestAnimationFrame(runFrame);
}

async function runStaticImage() {
  if (!staticImg.complete || staticImg.naturalWidth === 0) return;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  try {
    const results = await doInference(staticImg);
    drawResults(results);
  } catch (e) {
    console.error(e);
  }
}

function drawResults(results) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!results || results.length === 0) return;
  results.forEach((r, idx) => {
    // Use instance index for color so multiple same-class objects get distinct colors
    const c = COLORS[idx % COLORS.length];
    const drawBox_ = r.type === 'bbox' || (showBboxes && (r.type === 'keypoints' || r.type === 'mask'));
    if (drawBox_) drawBox(r, c);
    if (r.type === 'keypoints') drawKeypoints(r, c);
    if (r.type === 'mask')      drawMask(r, c);
  });
}

function drawBox(r, c) {
  const p1 = toCanvasCoords(r.x1, r.y1);
  const p2 = toCanvasCoords(r.x2, r.y2);
  const w = p2.x - p1.x;
  const h = p2.y - p1.y;

  ctx.strokeStyle = c;
  ctx.lineWidth    = 2;
  ctx.setLineDash([]);
  ctx.strokeRect(p1.x, p1.y, w, h);
  const lbl = getClassName(r.classId) + ': ' + (r.conf * 100).toFixed(0) + '%';
  ctx.font    = 'bold 10px -apple-system, sans-serif';
  const lw = ctx.measureText(lbl).width;
  const lx = p1.x;
  const ly = Math.max(5, p1.y - 22);
  ctx.fillStyle = c;
  ctx.fillRect(lx, ly, lw + 8, 17);
  ctx.fillStyle = '#000';
  ctx.fillText(lbl, lx + 4, ly + 13);
}

function drawKeypoints(r, c) {
  ctx.strokeStyle = c;
  ctx.lineWidth    = 1.5;
  ctx.setLineDash([]);
  kptPairs.forEach(([i, j]) => {
    const kp1 = r.kpts[i], kp2 = r.kpts[j];
    if (kp1.conf > 0.2 && kp2.conf > 0.2) {
      const p1 = toCanvasCoords(kp1.x, kp1.y);
      const p2 = toCanvasCoords(kp2.x, kp2.y);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  });
  ctx.fillStyle = c;
  r.kpts.forEach(k => {
    if (k.conf > 0.2) {
      const p = toCanvasCoords(k.x, k.y);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

const _maskCanvas = document.createElement('canvas');
_maskCanvas.width = 48;
_maskCanvas.height = 48;
const _maskCtx = _maskCanvas.getContext('2d');

function drawMask(r, c) {
  if (!r.mask) return;
  const imgData = _maskCtx.createImageData(48, 48);
  const [cr, cg, cb] = hexToRgb(c);
  for (let i = 0; i < 48 * 48; i++) {
    const base = i * 4;
    imgData.data[base]     = cr;
    imgData.data[base + 1] = cg;
    imgData.data[base + 2] = cb;
    imgData.data[base + 3] = r.mask[i] > 0.5 ? 110 : 0;
  }
  _maskCtx.putImageData(imgData, 0, 0);

  // Draw full mask without bbox clip so sigmoid contours follow object shape
  const tl = toCanvasCoords(0, 0);
  const br = toCanvasCoords(192, 192);
  ctx.drawImage(_maskCanvas, tl.x, tl.y, br.x - tl.x, br.y - tl.y);
}

// Run once after every model load to detect type, coordinate convention,
// and preprocessing mode so the very first inference frame is correct.
async function configureLoadedModel() {
  const conventions = await inf.detectOutputConventions();
  if (!conventions) return;
  const { type, normalizedCoords, numClasses } = conventions;

  normalizedDetCoords = normalizedCoords;
  if (type === 'detection') {
    detectionClassNames = numClasses === 80 ? COCO_NAMES : CLASS_NAMES;
  }

  // Preprocessing: custom normalized detection → crop; everything else → letterbox
  inf.setPreprocessMode(type === 'detection' && normalizedCoords ? 'crop' : 'letterbox');

  modelSelect.value = type;
  bboxToggleRow.hidden = type === 'detection';
  if (type !== 'detection') {
    showBboxes = false;
    bboxToggle.checked = false;
  }
  updateModelBadge(type);
}

// ── Camera ────────────────────────────────────────────────

btnCamera.addEventListener('click', async () => {
  if (cameraRunning) {
    stopCamera();
    placeholder.hidden = false;
    return;
  }
  if (!session) {
    errorMsg.textContent = 'No model loaded. Upload a model first using the "Model" button.';
    errorScreen.hidden = false;
    return;
  }
  staticImg.hidden = true;
  placeholder.hidden = true;
  video.hidden      = false;
  fpsBadge.hidden   = false;
  try {
    await startCamera();
  } catch (e) {
    video.hidden = true;
    placeholder.hidden = true;
    fpsBadge.hidden = true;
    errorMsg.textContent = 'Camera requires HTTPS connection. Use a local network connection.';
    errorScreen.hidden = false;
    return;
  }
  cameraRunning   = true;
  btnCamera.textContent = '⏹ Stop';
  btnCamera.classList.add('running');
  fpsSmooth = 0;
  lastInfStart = 0;
  resizeCanvas();
  rafId = requestAnimationFrame(runFrame);
});

// ── Image Upload ──────────────────────────────────────────

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file || !session) return;
  stopCamera();
  placeholder.hidden = true;
  const url = URL.createObjectURL(file);
  staticImg.src = url;
  await new Promise(r => { staticImg.onload = r; });
  staticImg.hidden = false;
  resizeCanvas();
  btnCapture.classList.add('active');
  await runStaticImage();
  URL.revokeObjectURL(url);
  fileInput.value = '';
});

// ── Model Upload ──────────────────────────────────────────

async function loadModelFromFile(file) {
  if (!file) return;
  if (cameraRunning) stopCamera();
  const url = URL.createObjectURL(file);
  const filename = file.name.replace(/\.[^/.]+$/, '');
  currentModelName = filename;
  errorScreen.hidden = true;
  loader.hidden = false;
  loaderMsg.textContent = 'Loading model...';
  try {
    if (session) { session.endSession?.(); session = null; }
    session = await inf.loadModel(url);
    await configureLoadedModel();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    loader.hidden = true;
    btnCamera.disabled = false;
    btnCamera.textContent = '📷 Start';
    btnUpload.classList.remove('disabled');
    btnModel.classList.remove('disabled');
    btnCapture.classList.remove('active');
    fpsBadge.hidden = true;
  } catch (e) {
    errorScreen.hidden = false;
    errorMsg.textContent = 'Could not load model. Use a YOLOv8 ONNX model with input [1,3,192,192]. ' + e.message;
    modelBadge.textContent = 'Error';
  }
  URL.revokeObjectURL(url);
  modelInput.value = '';
}

// Use File System Access API on desktop (no action sheet, direct file dialog).
// Fall back to <input type="file"> on iOS/unsupported browsers.
btnModel.addEventListener('click', async () => {
  if (window.showOpenFilePicker) {
    try {
      const [handle] = await window.showOpenFilePicker({
        types: [{ description: 'ONNX Model', accept: { 'application/octet-stream': ['.onnx'] } }],
        multiple: false,
      });
      await loadModelFromFile(await handle.getFile());
    } catch (e) {
      if (e.name !== 'AbortError') console.error(e);
    }
  } else {
    modelInput.click();
  }
});

modelInput.addEventListener('change', async (e) => {
  await loadModelFromFile(e.target.files[0]);
});

// ── Model Selector ────────────────────────────────────────

modelSelect.addEventListener('change', async () => {
  const modelType = modelSelect.value;
  const modelPath = BUILTIN_MODELS[modelType];
  if (!modelPath) return;
  if (cameraRunning) stopCamera();
  errorScreen.hidden = true;
  loader.hidden = false;
  loaderMsg.textContent = 'Loading ' + MODEL_LABELS[modelType] + ' model...';
  currentModelName = modelType;
  try {
    if (session) { session.endSession?.(); session = null; }
    session = await inf.loadModel(modelPath);
    await configureLoadedModel();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    loader.hidden = true;
    btnCamera.disabled = false;
    btnCamera.textContent = '📷 Start';
    btnUpload.classList.remove('disabled');
    btnModel.classList.remove('disabled');
    btnCapture.classList.remove('active');
    fpsBadge.hidden = true;
  } catch (e) {
    loader.hidden = true;
    errorScreen.hidden = false;
    errorMsg.textContent = 'Could not load ' + MODEL_LABELS[modelType] + ' model: ' + e.message;
    modelBadge.textContent = 'Error';
  }
});

bboxToggle.addEventListener('change', () => {
  showBboxes = bboxToggle.checked;
  if (!staticImg.hidden) runStaticImage();
});

// ── Capture Button ────────────────────────────────────────

btnCapture.addEventListener('click', () => {
  if (session) runStaticImage();
});

// ── Resize ────────────────────────────────────────────────

window.addEventListener('resize', resizeCanvas);

// ── Inference ─────────────────────────────────────────────

async function doInference(source) {
  if (!session) return [];
  const input = inf.preprocess(source);
  const feeds = { [session.inputNames[0]]: input };
  const out = await session.run(feeds);
  const output = out[session.outputNames[0]];
  const numBoxes = output.dims[2];
  const channels = output.dims[1];
  const cTV = inf.getConf();
  const iTV = inf.getIou();

  // Determine type from output shape — more reliable than dropdown
  // (handles manual model uploads that don't change the dropdown)
  let detectedType;
  if (session.outputNames.length > 1) {
    detectedType = 'segmentation';
  } else {
    const impliedPoseClasses = channels - 4 - 17 * 3;
    detectedType = (impliedPoseClasses >= 1 && impliedPoseClasses <= 5) ? 'pose' : 'detection';
  }

  // Sync dropdown + badge when model was loaded without using the selector
  if (modelSelect.value !== detectedType) {
    modelSelect.value = detectedType;
    bboxToggleRow.hidden = detectedType === 'detection';
    inf.setPreprocessMode(detectedType === 'detection' && normalizedDetCoords ? 'crop' : 'letterbox');
    if (detectedType !== 'detection') {
      showBboxes = false;
      bboxToggle.checked = false;
    }
    currentModelName = detectedType;
    updateModelBadge(detectedType);
  }

  if (detectedType === 'segmentation') {
    const protos = out[session.outputNames[1]];
    return inf.postprocessSegmentation(output, protos.data, numBoxes, cTV, iTV);
  }
  if (detectedType === 'pose') {
    return inf.postprocessPose(output, numBoxes, cTV, iTV);
  }
  return inf.postprocessDetection(output, numBoxes, cTV, iTV, normalizedDetCoords ? 192 : 1);
}

// ── Init ──────────────────────────────────────────────────

async function init() {
  resizeCanvas();
  errorScreen.hidden = true;
  fpsBadge.hidden = true;
  modelBadge.hidden = true;
  loader.hidden = false;
  try {
    session = await inf.loadModel();
    await configureLoadedModel();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    loader.hidden = true;
    btnCamera.disabled = false;
    btnCamera.textContent = '📷 Start';
    btnUpload.classList.remove('disabled');
    btnModel.classList.remove('disabled');
  } catch (e) {
    loader.hidden = false;
    errorScreen.hidden = false;
    errorMsg.textContent = 'Model not found at /models/yolov8n.onnx. ' +
      'Upload a YOLOv8 .onnx model using the "Model" button. ' +
      'Expected: input [1,3,192,192], output [1,84,756]. ' +
      e.message;
    fpsBadge.hidden = true;
    modelBadge.textContent = 'No Model';
    modelBadge.hidden = false;
  }
}

init();
