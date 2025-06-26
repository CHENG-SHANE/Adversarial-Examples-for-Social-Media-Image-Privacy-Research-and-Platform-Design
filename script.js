document.addEventListener('DOMContentLoaded', () => {
  const chooseBtn        = document.getElementById('choose-btn');
  const fileInput        = document.getElementById('file-upload');
  const previewContainer = document.getElementById('preview-container');
  const confirmSection   = document.getElementById('confirm-section');
  const confirmBtn       = document.getElementById('confirm-btn');
  const qrCodeSection    = document.getElementById('qr-code-section');
  const qrCodeContainer  = document.getElementById('qr-code-container');
  const loadingMask      = document.getElementById('loading-mask');

  const cameraBtn        = document.getElementById('camera-btn');
  const video            = document.getElementById('video');
  const captureBtn       = document.getElementById('capture-btn');
  const canvas           = document.getElementById('canvas');

  const API_BASE_URL = 'http://localhost:5000';

  let stream = null;
  // 存檔案或拍照，以及它們的dataUrl與filename
  const previewItems = [];

  // 檔案選擇
  chooseBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    previewItems.length = 0;         // 清空舊的
    previewContainer.innerHTML = ''; // 清掉畫面
    qrCodeSection.style.display = 'none';

    Array.from(fileInput.files).forEach(file => {
      if (!file.type.startsWith('image/')) return;
      const reader = new FileReader();
      reader.onload = e => {
        previewItems.push({
          type:     'file',
          file:     file,
          filename: file.name,
          dataUrl:  e.target.result
        });
        renderPreview();
      };
      reader.readAsDataURL(file);
    });
  });

  // 相機開／關
  cameraBtn.addEventListener('click', async () => {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
      video.style.display   = 'none';
      captureBtn.style.display = 'none';
      cameraBtn.textContent = 'Open Camera';
    } else {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.style.display   = 'block';
        captureBtn.style.display = 'inline-block';
        cameraBtn.textContent = 'Close Camera';
      } catch (err) {
        alert('無法開啟相機：' + err.message);
      }
    }
  });

  // 拍照
  captureBtn.addEventListener('click', () => {
    const ctx = canvas.getContext('2d');
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
      const filename = `camera_${Date.now()}.png`;
      const dataUrl  = canvas.toDataURL('image/png');

      previewItems.push({
        type:     'camera',
        blob:     blob,
        filename: filename,
        dataUrl:  dataUrl
      });
      renderPreview();
    }, 'image/png');
  });

  // 確認上傳
  confirmBtn.addEventListener('click', async () => {
    if (previewItems.length === 0) {
      alert('請先選擇或拍攝圖片');
      return;
    }

    const formData = new FormData();
    previewItems.forEach(item => {
      if (item.type === 'file') {
        formData.append('files', item.file,     item.filename);
      } else {
        formData.append('files', item.blob,     item.filename);
      }
    });

    loadingMask.style.display = 'flex';
    try {
      const res = await fetch(`${API_BASE_URL}/upload/`, {
        method: 'POST',
        body:   formData
      });
      if (!res.ok) throw new Error(await res.text());
      const { processed_files, results } = await res.json();
      if (!processed_files.length) {
        alert('請上傳包含人臉的圖片');
        return;
      }
      displayProcessedFiles(results);
    } catch (err) {
      console.error(err);
      alert('上傳失敗：' + err.message);
    } finally {
      loadingMask.style.display = 'none';
    }
  });


  function renderPreview() {
    previewContainer.innerHTML = '';
    previewItems.forEach((item, idx) => {
      const wrapper = document.createElement('div');
      wrapper.classList.add('preview-item');

      const img = document.createElement('img');
      img.src = item.dataUrl;
      wrapper.appendChild(img);

      const del = document.createElement('button');
      del.classList.add('delete-btn');
      del.innerHTML = '×';
      del.onclick = () => {
        previewItems.splice(idx, 1);
        renderPreview();
      };
      wrapper.appendChild(del);

      previewContainer.appendChild(wrapper);
    });

    confirmSection.style.display = previewItems.length ? 'block' : 'none';
  }

  // 顯示結果
  function displayProcessedFiles(results) {
    document.querySelector('.upload-section').style.display   = 'none';
    previewContainer.style.display    = 'none';  // 隱藏預覽區
    confirmSection.style.display      = 'none';
    qrCodeContainer.innerHTML         = '';
    qrCodeSection.style.display       = 'block';

    results.forEach(result => {
      const url = `${API_BASE_URL}/download/${encodeURIComponent(result.filename)}`;
      // 用 original_filename 找 dataUrl
      const match = previewItems.find(p => p.filename === result.original_filename);
      const imgSrc = match ? match.dataUrl : '';

      const itemDiv = document.createElement('div');
      itemDiv.classList.add('qr-item');

      // 置信度
      const confValue = result.average_confidence ?? result.confidence ?? 0;
      const confP = document.createElement('p');
      confP.textContent = `FaceNet Confidence: ${(confValue * 100).toFixed(2)}%`;
      confP.classList.add('confidence-text');
      itemDiv.appendChild(confP);

      // 圖 + QR
      const blockDiv = document.createElement('div');
      blockDiv.classList.add('qr-code-block');

      const origImg = document.createElement('img');
      origImg.src = imgSrc;
      origImg.classList.add('orig-img');
      blockDiv.appendChild(origImg);

      const canvasQR = document.createElement('canvas');
      blockDiv.appendChild(canvasQR);
      QRCode.toCanvas(canvasQR, url, err => {
        if (err) console.error('QR Error:', err);
      });

      itemDiv.appendChild(blockDiv);

      // 下載按鈕
      const btn = document.createElement('button');
      btn.textContent = 'Download';
      btn.classList.add('download-btn');
      btn.onclick = () => {
        const a = document.createElement('a');
        a.href     = url;
        a.download = result.filename;
        a.click();
      };
      itemDiv.appendChild(btn);

      qrCodeContainer.appendChild(itemDiv);
    });
  }
});
