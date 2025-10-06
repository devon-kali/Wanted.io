let opencvReady = false;
let cascadeReady = false;
let classifier;

function updateStatus() {
  const status = document.getElementById('status');
  if (opencvReady && cascadeReady) {
    status.innerText = '✅ Ready for training and recognition';
    document.querySelectorAll('button').forEach(b => b.disabled = false);
  } else {
    status.innerText = '⏳ Loading OpenCV and cascade...';
    document.querySelectorAll('button').forEach(b => b.disabled = true);
  }
}

function loadCascade() {
  const cascadeUrl = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml';
  fetch(cascadeUrl)
    .then(res => res.text())
    .then(data => {
      cv.FS_createDataFile('/', 'haarcascade_frontalface_default.xml', data, true, false, false);
      classifier = new cv.CascadeClassifier();
      classifier.load('haarcascade_frontalface_default.xml');
      cascadeReady = true;
      updateStatus();
    })
    .catch(err => console.error('Cascade load error:', err));
}

function loadOpenCV() {
  const script = document.createElement('script');
  script.src = 'https://docs.opencv.org/4.x/opencv.js';
  script.async = true;
  script.onload = () => {
    cv['onRuntimeInitialized'] = () => {
      opencvReady = true;
      updateStatus();
      loadCascade();
    };
  };
  document.body.appendChild(script);
}

window.onload = loadOpenCV;

const trainData = {};
document.getElementById('addTrain').onclick = async () => {
  const files = document.getElementById('trainImages').files;
  const name = document.getElementById('personName').value.trim();
  if (!files.length || !name) return alert('Upload images and enter a name.');

  if (!trainData[name]) trainData[name] = [];

  for (const file of files) {
    const img = await createImageBitmap(file);
    const mat = cv.imread(await fileToCanvas(img));
    const faces = detectFaces(mat);
    if (faces.length) {
      const face = cropFace(mat, faces[0]);
      trainData[name].push(face);
    }
    mat.delete();
  }
  alert(`Added ${files.length} images for ${name}`);
};

document.getElementById('trainBtn').onclick = () => {
  if (!opencvReady || !cascadeReady) return alert('OpenCV or cascade not yet loaded.');
  computeHistograms();
};

document.getElementById('recognizeBtn').onclick = async () => {
  if (!opencvReady || !cascadeReady) return alert('OpenCV or cascade not yet loaded.');
  const file = document.getElementById('testImage').files[0];
  if (!file) return alert('Upload a test image.');
  const img = await createImageBitmap(file);
  const mat = cv.imread(await fileToCanvas(img));
  recognizeFaces(mat);
  mat.delete();
};

async function fileToCanvas(imgBitmap) {
  const canvas = document.createElement('canvas');
  canvas.width = imgBitmap.width;
  canvas.height = imgBitmap.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imgBitmap, 0, 0);
  return canvas;
}

function detectFaces(mat) {
  const gray = new cv.Mat();
  cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
  const faces = new cv.RectVector();
  classifier.detectMultiScale(gray, faces, 1.1, 4);
  const result = [];
  for (let i = 0; i < faces.size(); i++) result.push(faces.get(i));
  gray.delete();
  faces.delete();
  return result;
}

function cropFace(mat, rect) {
  const roi = mat.roi(rect);
  const gray = new cv.Mat();
  cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);
  const resized = new cv.Mat();
  cv.resize(gray, resized, new cv.Size(100, 100));
  roi.delete();
  gray.delete();
  return resized;
}

const histograms = {};

function computeHistograms() {
  for (const name in trainData) {
    const allFaces = trainData[name];
    const hists = allFaces.map(face => lbpHistogram(face));
    const avg = averageHistogram(hists);
    histograms[name] = avg;
  }
  alert('Training complete!');
}

function lbpHistogram(faceMat) {
  const rows = faceMat.rows, cols = faceMat.cols;
  const hist = new Array(256).fill(0);
  for (let i = 1; i < rows - 1; i++) {
    for (let j = 1; j < cols - 1; j++) {
      let center = faceMat.ucharPtr(i, j)[0];
      let code = 0;
      code |= (faceMat.ucharPtr(i - 1, j - 1)[0] > center) << 7;
      code |= (faceMat.ucharPtr(i - 1, j)[0] > center) << 6;
      code |= (faceMat.ucharPtr(i - 1, j + 1)[0] > center) << 5;
      code |= (faceMat.ucharPtr(i, j + 1)[0] > center) << 4;
      code |= (faceMat.ucharPtr(i + 1, j + 1)[0] > center) << 3;
      code |= (faceMat.ucharPtr(i + 1, j)[0] > center) << 2;
      code |= (faceMat.ucharPtr(i + 1, j - 1)[0] > center) << 1;
      code |= (faceMat.ucharPtr(i, j - 1)[0] > center);
      hist[code]++;
    }
  }
  const sum = hist.reduce((a, b) => a + b, 0);
  return hist.map(v => v / sum);
}

function averageHistogram(hists) {
  const sum = new Array(256).fill(0);
  for (const h of hists) for (let i = 0; i < 256; i++) sum[i] += h[i];
  return sum.map(v => v / hists.length);
}

function chiSquare(hist1, hist2) {
  let d = 0;
  for (let i = 0; i < hist1.length; i++) {
    const a = hist1[i], b = hist2[i];
    if (a + b !== 0) d += ((a - b) ** 2) / (a + b);
  }
  return d / 2;
}

function recognizeFaces(mat) {
  const faces = detectFaces(mat);
  const canvas = document.getElementById('canvas');
  cv.imshow(canvas, mat);
  const ctx = canvas.getContext('2d');
  const threshold = parseFloat(document.getElementById('threshold').value);

  faces.forEach(rect => {
    const face = cropFace(mat, rect);
    const hist = lbpHistogram(face);
    let bestName = 'Unknown';
    let bestDist = Infinity;

    for (const name in histograms) {
      const d = chiSquare(histograms[name], hist);
      if (d < bestDist) {
        bestDist = d;
        bestName = name;
      }
    }

    if (bestDist > threshold) bestName = 'Unknown';

    ctx.strokeStyle = '#1e88e5';
    ctx.lineWidth = 2;
    ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    ctx.fillStyle = 'rgba(30,136,229,0.7)';
    ctx.fillRect(rect.x, rect.y - 20, rect.width, 20);
    ctx.fillStyle = '#fff';
    ctx.fillText(`${bestName}`, rect.x + 5, rect.y - 5);
  });
}
