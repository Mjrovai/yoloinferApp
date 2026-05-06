import * as ort from 'onnxruntime-web';

const INPUT_W = 192;
const INPUT_H = 192;

let session = null;
let customPath = null;
let lastPreprocessState = { sx: 0, sy: 0, sw: 192, sh: 192, srcW: 192, srcH: 192 };
let preprocessMode = 'crop'; // 'crop' (center-square) | 'letterbox' (scale+pad)

export function setPreprocessMode(mode) { preprocessMode = mode; }
export function getPreprocessMode()      { return preprocessMode; }

// Run a dummy inference immediately after loading to detect:
//   - model type (detection / pose / segmentation)
//   - coordinate convention (normalized 0-1 vs pixel 0-192)
// Returns { type, normalizedCoords } or null if no session loaded.
export async function detectOutputConventions() {
  if (!session) return null;
  const dummy = new ort.Tensor('float32', new Float32Array(3 * INPUT_H * INPUT_W).fill(0.45), [1, 3, INPUT_H, INPUT_W]);
  const out = await session.run({ [session.inputNames[0]]: dummy });
  const o = out[session.outputNames[0]];
  const channels = o.dims[1];
  const numBoxes = o.dims[2];
  const hasTwoOutputs = session.outputNames.length > 1;

  let type;
  if (hasTwoOutputs) {
    type = 'segmentation';
  } else {
    const impliedPoseClasses = channels - 4 - 17 * 3;
    type = (impliedPoseClasses >= 1 && impliedPoseClasses <= 5) ? 'pose' : 'detection';
  }

  // Detection models: check if bbox cx values are normalized (0-1) or pixel (0-192).
  // Normalized custom models have max cx << 1; standard Ultralytics outputs have max cx >> 1.
  let normalizedCoords = true;
  if (type === 'detection') {
    let maxCx = 0;
    const end = Math.min(numBoxes, 200);
    for (let i = 0; i < end; i++) maxCx = Math.max(maxCx, o.data[i]);
    normalizedCoords = maxCx < 1.5;
  }

  const numClasses = type === 'detection' ? channels - 4 : null;
  return { type, normalizedCoords, numClasses };
}

export function getLastPreprocessState() {
  return lastPreprocessState;
}

export async function loadModel(overridePath) {
  ort.env.wasm.wasmPaths =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  customPath = overridePath || null;
  const modelPath = overridePath || '/models/yolov8n.onnx';
  
  const providers = ['wasm'];
  
  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: providers,
    graphOptimizationLevel: 'all',
  });
  return session;
}

export function getSession() {
  return session;
}

export function setCustomPath(url) {
  customPath = url;
}

export function getCustomPath() {
  return customPath;
}

const _canvas = document.createElement('canvas');
_canvas.width   = INPUT_W;
_canvas.height = INPUT_H;
const _ctx = _canvas.getContext('2d', { willReadFrequently: true });

let _lastSource = null;

export function preprocess(source) {
  _lastSource = source;

  const srcW = source.videoWidth || source.naturalWidth || source.width || INPUT_W;
  const srcH = source.videoHeight || source.naturalHeight || source.height || INPUT_H;

  if (preprocessMode === 'letterbox') {
    // Scale full frame to fit 192×192 with black padding (standard YOLO eval mode)
    const scale = Math.min(INPUT_W / srcW, INPUT_H / srcH);
    const newW  = Math.round(srcW * scale);
    const newH  = Math.round(srcH * scale);
    const padX  = Math.floor((INPUT_W - newW) / 2);
    const padY  = Math.floor((INPUT_H - newH) / 2);
    lastPreprocessState = { srcW, srcH, scale, padX, padY };
    _ctx.fillStyle = '#000';
    _ctx.fillRect(0, 0, INPUT_W, INPUT_H);
    _ctx.drawImage(source, padX, padY, newW, newH);
  } else {
    // Center square crop (used for custom detection model)
    const minDim = Math.min(srcW, srcH);
    const sx = Math.round((srcW - minDim) / 2);
    const sy = Math.round((srcH - minDim) / 2);
    lastPreprocessState = { sx, sy, sw: minDim, sh: minDim, srcW, srcH };
    _ctx.drawImage(source, sx, sy, minDim, minDim, 0, 0, INPUT_W, INPUT_H);
  }

  const imgData = _ctx.getImageData(0, 0, INPUT_W, INPUT_H).data;
  const pxCount = INPUT_W * INPUT_H;
  const f32 = new Float32Array(3 * pxCount);

  for (let i = 0; i < pxCount; i++) {
    const s = i * 4;
    f32[i]               = imgData[s]     / 255.0;
    f32[i + pxCount]     = imgData[s + 1] / 255.0;
    f32[i + pxCount * 2] = imgData[s + 2] / 255.0;
  }

  return new ort.Tensor('float32', f32, [1, 3, INPUT_H, INPUT_W]);
}

let confThresh = 0.35;
let iouThresh  = 0.45;

export function getConf() { return confThresh; }
export function getIou()   { return iouThresh; }
export function setConf(v) { confThresh = v; }
export function setIou(v)   { iouThresh = v; }

export function nms(boxes, iouTV) {
  const byConf = [...boxes].sort((a, b) => b.conf - a.conf);
  const kept   = [];
  const pruned = new Set();
  
  for (let i = 0; i < byConf.length; i++) {
    if (pruned.has(i)) continue;
    kept.push(byConf[i]);
    for (let j = i + 1; j < byConf.length; j++) {
      if (pruned.has(j)) continue;
      if (computeIoU(byConf[i], byConf[j]) < iouTV) continue;
      pruned.add(j);
    }
  }
  return kept;
}

function computeIoU(a, b) {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const aA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const bA = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / Math.max(aA + bA - inter, 0.0001);
}

export function postprocessDetection(outputTensor, numBoxes, confTV, iouTV, coordScale = 192) {
  const data = outputTensor.data;
  const numClasses = outputTensor.dims[1] - 4;
  const candidates = [];

  for (let i = 0; i < numBoxes; i++) {
    const cx = data[0 * numBoxes + i] * coordScale;
    const cy = data[1 * numBoxes + i] * coordScale;
    const w  = data[2 * numBoxes + i] * coordScale;
    const h  = data[3 * numBoxes + i] * coordScale;
    
    let bestScore = 0;
    let bestClassId = 0;
    for (let c = 0; c < numClasses; c++) {
      const si = (4 + c) * numBoxes + i;
      if (data[si] > bestScore) {
        bestScore = data[si];
        bestClassId = c;
      }
    }

    const conf = Math.max(0, bestScore);
    if (conf < confTV) continue;

    candidates.push({
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
      classId: bestClassId,
      conf,
      type: 'bbox',
    });
  }
  
  return nms(candidates, iouTV);
}

export function postprocessPose(outputTensor, numBoxes, confTV, iouTV) {
  const data = outputTensor.data;
  const kptCount = 17;
  const numClasses = outputTensor.dims[1] - 4 - kptCount * 3;
  const candidates = [];

  for (let i = 0; i < numBoxes; i++) {
    // pose/seg models output absolute pixel coords (0-192), not normalized
    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w  = data[2 * numBoxes + i];
    const h  = data[3 * numBoxes + i];

    let bestScore = 0;
    let bestClassId = 0;
    for (let c = 0; c < numClasses; c++) {
      const si = (4 + c) * numBoxes + i;
      if (data[si] > bestScore) {
        bestScore = data[si];
        bestClassId = c;
      }
    }

    const conf = Math.max(0, bestScore);
    if (conf < confTV) continue;

    const kpts = [];
    for (let k = 0; k < kptCount; k++) {
      const k0 = (4 + numClasses + k * 3) * numBoxes + i;
      const k1 = (4 + numClasses + k * 3 + 1) * numBoxes + i;
      const k2 = (4 + numClasses + k * 3 + 2) * numBoxes + i;
      kpts.push({ x: data[k0], y: data[k1], conf: data[k2] });
    }

    candidates.push({
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
      classId: bestClassId,
      conf,
      kpts,
      type: 'keypoints',
    });
  }

  return nms(candidates, iouTV);
}

export function postprocessSegmentation(outputTensor, protosData, numBoxes, confTV, iouTV) {
  const data = outputTensor.data;
  const maskDims = 32;
  const PROTO_H = 48, PROTO_W = 48;
  const numClasses = outputTensor.dims[1] - 4 - maskDims;
  const candidates = [];

  for (let i = 0; i < numBoxes; i++) {
    // pose/seg models output absolute pixel coords (0-192), not normalized
    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w  = data[2 * numBoxes + i];
    const h  = data[3 * numBoxes + i];

    let bestScore = 0;
    let bestClassId = 0;
    for (let c = 0; c < numClasses; c++) {
      const si = (4 + c) * numBoxes + i;
      if (data[si] > bestScore) {
        bestScore = data[si];
        bestClassId = c;
      }
    }

    const conf = Math.max(0, bestScore);
    if (conf < confTV) continue;

    const maskCoeffs = new Float32Array(maskDims);
    for (let m = 0; m < maskDims; m++) {
      maskCoeffs[m] = data[(4 + numClasses + m) * numBoxes + i];
    }

    // Compute 48×48 instance mask: sigmoid(coefficients · protos)
    // Pixels outside the detection bbox are zeroed to prevent bleeding into
    // neighboring objects (proto space = same extent as 192×192 input)
    const scale = PROTO_H / 192;
    const bpx1 = Math.max(0,  Math.floor((cx - w / 2) * scale));
    const bpy1 = Math.max(0,  Math.floor((cy - h / 2) * scale));
    const bpx2 = Math.min(PROTO_W - 1, Math.ceil((cx + w / 2) * scale));
    const bpy2 = Math.min(PROTO_H - 1, Math.ceil((cy + h / 2) * scale));

    const mask = new Float32Array(PROTO_H * PROTO_W);
    for (let py = 0; py < PROTO_H; py++) {
      for (let px = 0; px < PROTO_W; px++) {
        if (py < bpy1 || py > bpy2 || px < bpx1 || px > bpx2) continue;
        let val = 0;
        for (let m = 0; m < maskDims; m++) {
          val += maskCoeffs[m] * protosData[m * PROTO_H * PROTO_W + py * PROTO_W + px];
        }
        mask[py * PROTO_W + px] = 1 / (1 + Math.exp(-val));
      }
    }

    candidates.push({
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
      classId: bestClassId,
      conf,
      mask,
      type: 'mask',
    });
  }

  return nms(candidates, iouTV);
}
