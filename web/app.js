// ONNX tensor names (verified from export)
const ONNX_INPUT = 'float_input';
const ONNX_PROBS = 'probabilities';

let session = null;
let labels = [];

// Normalization identical to hand_utils.py extract_landmarks()
function extractLandmarks(landmarks) {
  const wx = landmarks[0].x, wy = landmarks[0].y, wz = landmarks[0].z;
  const centered = landmarks.map(lm => [lm.x - wx, lm.y - wy, lm.z - wz]);
  const norms = centered.map(([x, y, z]) => Math.sqrt(x * x + y * y + z * z));
  const maxDist = Math.max(...norms);
  const flat = centered.flatMap(([x, y, z]) =>
    maxDist > 0 ? [x / maxDist, y / maxDist, z / maxDist] : [x, y, z]
  );
  return new Float32Array(flat);
}

async function predict(landmarks) {
  if (!session || labels.length === 0) return null;
  const input = extractLandmarks(landmarks);
  const tensor = new ort.Tensor('float32', input, [1, 63]);
  const results = await session.run({ [ONNX_INPUT]: tensor });
  const probs = Array.from(results[ONNX_PROBS].data);
  const maxIdx = probs.indexOf(Math.max(...probs));
  return labels[maxIdx] ?? null;
}

function setStatus(text) {
  const el = document.getElementById('status-overlay');
  if (el) el.textContent = text;
}

async function main() {
  const mainVideo = document.getElementById('main-video');
  const bgVideo = document.getElementById('bg-video');
  const overlay = document.getElementById('overlay');
  const ctx = overlay.getContext('2d');
  const predictionEl = document.getElementById('prediction');

  setStatus('Chargement du modèle...');

  // Load labels and ONNX model in parallel
  const [labelsRes] = await Promise.all([
    fetch('./models/labels.json'),
    ort.InferenceSession.create('./models/asl_model.onnx').then(s => { session = s; })
  ]);
  labels = await labelsRes.json();

  setStatus('Accès caméra...');

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
  } catch (err) {
    setStatus('Autorisez l\'accès à la caméra dans votre navigateur.');
    predictionEl.textContent = '!';
    return;
  }
  mainVideo.srcObject = stream;
  bgVideo.srcObject = stream;

  await new Promise(resolve => { mainVideo.onloadedmetadata = resolve; });

  setStatus('');

  const hands = new Hands({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
  });

  hands.onResults(async results => {
    overlay.width = mainVideo.videoWidth;
    overlay.height = mainVideo.videoHeight;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const lms = results.multiHandLandmarks[0];

      drawConnectors(ctx, lms, HAND_CONNECTIONS, { color: '#cc0000', lineWidth: 2 });
      drawLandmarks(ctx, lms, { color: '#ff0000', radius: 4, fillColor: '#ff4444' });

      const letter = await predict(lms);
      if (letter) predictionEl.textContent = letter;
    } else {
      predictionEl.textContent = '--';
    }
  });

  const camera = new Camera(mainVideo, {
    onFrame: async () => { await hands.send({ image: mainVideo }); },
    width: 640,
    height: 480
  });

  camera.start();
}

main().catch(err => {
  console.error(err);
  setStatus('Erreur : ' + err.message);
  document.getElementById('prediction').textContent = '!';
});
