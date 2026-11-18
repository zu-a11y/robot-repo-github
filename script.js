
const robotFace = document.getElementById("robotFace");
const statusText = document.getElementById("status");
const aiStatus = document.getElementById("aiStatus");
const confidenceText = document.getElementById("confidence");
const camera = document.getElementById("camera");
const canvas = document.getElementById("captureCanvas");
const ctx = canvas.getContext("2d");

const eyes = document.querySelectorAll(".eye");

const LABELS = ["angry", "happy", "neutral", "sad", "surprise"];

let model = null;
let cameraStream = null;
let detectionInterval = null;
let currentExpression = "normal";
let isPredicting = false;
let noFaceCount = 0;
let predictionHistory = [];

const EXPRESSION_MAP = {
    angry: "angry",
    happy: "happy",
    neutral: "normal",
    sad: "sad",
    surprise: "surprised"
};
// MEDIAPIPE FACE DETECTOR
let faceDetector = null;

async function initializeFaceDetector() {
    try {
        const vision = await import(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14"
        );

        const fileset = await vision.FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );

        faceDetector = await vision.FaceDetector.createFromOptions(fileset, {
            baseOptions: {
                modelAssetPath:
                    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            },
            runningMode: "VIDEO"
        });

        console.log("Face detector ready");

	} catch (error) {
    console.error("Face detector loading failed:", error);
    throw error;
	}
}
// =====================================
// LOAD MODEL
// =====================================

async function loadModel() {
    try {
        aiStatus.textContent = "AI Status: Loading model...";
        await initializeFaceDetector();
        model = await tf.loadGraphModel("./web_model/model.json");

        aiStatus.textContent = "AI Status: Model ready";

        console.log("V3 model loaded successfully");
    } catch (error) {
        console.error(error);
        aiStatus.textContent = "AI Status: Model loading failed";
    }
}

// =====================================
// CAMERA
// =====================================

async function startCamera() {
    try {
        if (!model) {
            aiStatus.textContent = "AI Status: Model not ready";
            return;
        }

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: 640,
                height: 480
            },
            audio: false
        });

        camera.srcObject = cameraStream;

        await camera.play();

        document.getElementById("startCamera").disabled = true;
        document.getElementById("stopCamera").disabled = false;

        aiStatus.textContent = "AI Status: Camera active";

        detectionInterval = setInterval(predictEmotion, 300);

    } catch (error) {
        console.error(error);
        aiStatus.textContent = "AI Status: Camera access failed";
    }
}

// =====================================
// STOP CAMERA
// =====================================

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    camera.srcObject = null;

    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }

    document.getElementById("startCamera").disabled = false;
    document.getElementById("stopCamera").disabled = true;

    aiStatus.textContent = "AI Status: Camera stopped";
    confidenceText.textContent = "Confidence: --";

    setExpression("normal");
}

// =====================================
// PREPROCESS IMAGE
// =====================================

function preprocessImage() {
    if (!faceDetector || camera.readyState < 2) {
        return null;
    }

    const result = faceDetector.detectForVideo(
        camera,
        performance.now()
    );

    if (!result.detections || result.detections.length === 0) {
        return null;
    }

    const box = result.detections[0].boundingBox;

    const padding = 0.20;

    const x = Math.max(0, box.originX - box.width * padding);
    const y = Math.max(0, box.originY - box.height * padding);

    const width = Math.min(
        camera.videoWidth - x,
        box.width * (1 + padding * 2)
    );

    const height = Math.min(
        camera.videoHeight - y,
        box.height * (1 + padding * 2)
    );

    const size = Math.min(width, height);

    const cropX = x + (width - size) / 2;
    const cropY = y + (height - size) / 2;

    canvas.width = 48;
    canvas.height = 48;

    ctx.drawImage(
        camera,
        cropX, cropY, size, size,
        0, 0, 48, 48
    );

    const image = ctx.getImageData(0, 0, 48, 48);
    const pixels = [];

    for (let i = 0; i < image.data.length; i += 4) {
        const gray = (
            0.299 * image.data[i] +
            0.587 * image.data[i + 1] +
            0.114 * image.data[i + 2]
        ) / 255;

        pixels.push(gray, gray, gray);
    }

    return tf.tensor4d(pixels, [1, 48, 48, 3]);
}

// =====================================
// PREDICT EMOTION
// =====================================

	async function predictEmotion() {
    if (isPredicting || !model || camera.readyState < 2) return;

    isPredicting = true;

    let input;
    let output;

    try {
        input = preprocessImage();


	if (!input) {
    noFaceCount++;

    confidenceText.textContent = "Confidence: Face not detected";

    if (noFaceCount >= 3) {
        predictionHistory = [];
        setExpression("normal");
    }

    return;
}

noFaceCount = 0;

        output = model.predict(input);

        const probabilities = await output.data();

        let maxIndex = 0;

        for (let i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }

        const label = LABELS[maxIndex];
        const confidence = probabilities[maxIndex];

        predictionHistory.push({ label, confidence });

        if (predictionHistory.length > 3) {
            predictionHistory.shift();
        }

        const stablePrediction = getStablePrediction();

        confidenceText.textContent =
            `Confidence: ${(confidence * 100).toFixed(1)}%`;

if (stablePrediction && stablePrediction.confidence >= 0.55) {
    setExpression(EXPRESSION_MAP[stablePrediction.label]);
}
    } catch (error) {
        console.error("Prediction error:", error);
        aiStatus.textContent = "AI Status: Prediction error";

    	} finally {
        if (input) input.dispose();
        if (output) output.dispose();
        isPredicting = false;
    }
}

// =====================================
// STABLE PREDICTION
// =====================================

function getStablePrediction() {
    if (predictionHistory.length < 3) return null;

    const counts = {};

    predictionHistory.forEach(prediction => {
        const label = prediction.label;

        if (!counts[label]) {
            counts[label] = {
                count: 0,
                confidence: 0
            };
        }

        counts[label].count++;
        counts[label].confidence += prediction.confidence;
    });

    const bestLabel = Object.keys(counts).reduce((a, b) =>
        counts[a].count > counts[b].count ? a : b
    );

    if (counts[bestLabel].count < 2) return null;

    return {
        label: bestLabel,
        confidence:
            counts[bestLabel].confidence / counts[bestLabel].count
    };
}

// =====================================
// ROBOT EXPRESSION
// =====================================

function setExpression(expression) {
    currentExpression = expression;

    robotFace.className = `robot-face ${expression}`;

    statusText.textContent =
        `Current expression: ${expression.toUpperCase()}`;
}

// =====================================
// EYE MOVEMENT
// =====================================

function moveEyes(x, y) {
    eyes.forEach(eye => {
        eye.style.setProperty("--eye-x", `${x}px`);
        eye.style.setProperty("--eye-y", `${y}px`);
    });
}

function randomEyeMovement() {
    const x = Math.floor(Math.random() * 13) - 6;
    const y = Math.floor(Math.random() * 9) - 4;

    moveEyes(x, y);
}

setInterval(() => {
    if (currentExpression !== "confused") {
        randomEyeMovement();
    }
}, 2500);

// =====================================
// BLINK
// =====================================

function blink() {
    robotFace.classList.add("blink");

    setTimeout(() => {
        robotFace.classList.remove("blink");
    }, 180);
}

setInterval(blink, 5000);

// =====================================
// BUTTON EVENTS
// =====================================

document.getElementById("startCamera")
    .addEventListener("click", startCamera);

document.getElementById("stopCamera")
    .addEventListener("click", stopCamera);

// =====================================
// INITIALIZE
// =====================================

document.getElementById("stopCamera").disabled = true;

setExpression("normal");

loadModel();
