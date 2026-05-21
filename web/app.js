const ONNX_INPUT = 'float_input';
const ONNX_PROBS = 'probabilities';

let session = null;
let labels = [];
let predicting = false;

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

  const [labelsRes] = await Promise.all([
    fetch('./models/labels.json'),
    ort.InferenceSession.create('./models/asl_model.onnx').then(s => { session = s; })
  ]);
  labels = await labelsRes.json();

  setStatus('Accès caméra...');

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } }
    });
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
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  // Synchronous results handler — never awaits, never blocks the next frame
  hands.onResults(results => {
    // Only resize canvas when video dimensions actually change (expensive)
    if (overlay.width !== mainVideo.videoWidth) overlay.width = mainVideo.videoWidth;
    if (overlay.height !== mainVideo.videoHeight) overlay.height = mainVideo.videoHeight;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const lms = results.multiHandLandmarks[0];

      drawConnectors(ctx, lms, HAND_CONNECTIONS, { color: '#ffffff', lineWidth: 2 });
      drawLandmarks(ctx, lms, { color: '#fc1f1f', radius: 4, fillColor: '#ff4444' });

      // Fire prediction without blocking drawing — updates whenever ONNX is ready
      if (!predicting) {
        predicting = true;
        predict(lms).then(letter => {
          if (letter) predictionEl.textContent = letter;
          predicting = false;
        });
      }
    } else {
      predictionEl.textContent = '-';
    }
  });

  const camera = new Camera(mainVideo, {
    onFrame: async () => { await hands.send({ image: mainVideo }); },
    width: 1280,
    height: 720
  });

  camera.start();
}

main().catch(err => {
  console.error(err);
  setStatus('Erreur : ' + err.message);
  document.getElementById('prediction').textContent = '!';
});
