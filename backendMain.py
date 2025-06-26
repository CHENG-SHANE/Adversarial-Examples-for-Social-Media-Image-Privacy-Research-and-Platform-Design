from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from tqdm import tqdm
import sys


# 初始化Flask
app = Flask(__name__)
CORS(app)

# 資料夾路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')

# 用戶上傳與處理的資料夾
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 載入模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
yolo_model = YOLO(r'C:/Users/shane/OneDrive/桌面/adv/best.pt')

# 預處理
attack_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])
embed_transform = attack_transform

# 檢查檔案類型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 反正規化
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor * 0.5 + 0.5, 0, 1)

# 將tensor轉為PIL圖片
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = denormalize(tensor.cpu()).detach()
    return transforms.ToPILImage()(tensor)

# 獲取人臉嵌入向量
def get_embedding_tensor(img_pil: Image.Image) -> torch.Tensor:
    img_tensor = embed_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet(img_tensor)
    return embedding

# 偵測並裁切人臉
def detect_and_crop_face(image_path: str):
    results = yolo_model(image_path, conf=0.3)[0]
    img = Image.open(image_path).convert('RGB')
    faces = []
    for box in results.boxes:
        if int(box.cls[0].item()) == 0:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            face = img.crop((x1, y1, x2, y2))
            faces.append(((x1, y1, x2, y2), face))
    return img, faces

# 白盒攻擊函數
def white_box_attack_record(image, target_model, target_embedding, orig_embedding,
                            epsilon, alpha, num_iterations, desc='Attacking'):
    image_tensor = attack_transform(image).unsqueeze(0).to(device)
    perturbed_image = image_tensor.clone().detach().to(device).requires_grad_(True)
    for _ in tqdm(range(num_iterations), desc=desc, file=sys.stdout):
        curr_emb = target_model(perturbed_image)
        loss = -F.cosine_similarity(curr_emb, target_embedding).mean()
        loss.backward()
        grad = perturbed_image.grad.sign()
        perturbed_image = torch.clamp(perturbed_image.detach() + alpha * grad,
                                      image_tensor - epsilon, image_tensor + epsilon)
        perturbed_image = torch.clamp(perturbed_image, -1, 1).requires_grad_(True)
    return perturbed_image


@app.route('/upload/', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files part'}), 400

        files = request.files.getlist('files')
        # 攻擊參數(可以微調，出來的效果不一樣)
        epsilon = float(request.form.get('epsilon', 0.022))
        alpha = float(request.form.get('alpha', 0.001))
        num_iter = int(request.form.get('num_iterations', 120))
        cosine_threshold = float(request.form.get('success_threshold', 0.6))
        raw_conf_thr = (cosine_threshold + 1.0) / 2.0
        confidence_threshold = float(np.clip(raw_conf_thr, 0.0, 1.0))

        processed_files = []
        results = []

        for file in files:
            if not (file and allowed_file(file.filename)):
                continue

            # 儲存上傳檔案
            unique_fn = f"{uuid.uuid4().hex}_{file.filename}"
            save_path = os.path.join(UPLOAD_FOLDER, unique_fn)
            file.save(save_path)

            # 偵測並裁切人臉
            original_img, faces = detect_and_crop_face(save_path)
            if not faces:
                continue

            # 對每張臉做攻擊並貼回
            adv_img = original_img.copy()
            cos_list = []
            conf_list = []
            success_list = []

            for (x1, y1, x2, y2), face in faces:
                orig_emb = get_embedding_tensor(face)
                target_emb = torch.randn(1, 512).to(device)

                adv_tensor = white_box_attack_record(
                    face, facenet, target_emb, orig_emb,
                    epsilon, alpha, num_iter
                )

                # 計算攻擊後的人臉confidence
                adv_emb = facenet(adv_tensor)
                cos_sim = F.cosine_similarity(orig_emb, adv_emb, dim=1).item()
                raw_conf = (cos_sim + 1.0) / 2.0
                confidence = float(np.clip(raw_conf, 0.0, 1.0))
                success = confidence < confidence_threshold

                # 貼回對抗後人臉
                adv_face = tensor_to_pil(adv_tensor).resize((x2 - x1, y2 - y1))
                adv_img.paste(adv_face, (x1, y1))

                cos_list.append(cos_sim)
                conf_list.append(confidence)
                success_list.append(success)

            # 整張圖儲存與回傳
            out_fn = f"adv_{unique_fn}"
            out_path = os.path.join(PROCESSED_FOLDER, out_fn)
            adv_img.save(out_path)

            avg_cos = float(np.mean(cos_list))
            avg_conf = float(np.mean(conf_list))
            overall_success = all(success_list)

            processed_files.append(out_fn)
            results.append({
                'filename': out_fn,
                'average_cosine_similarity': avg_cos,
                'average_confidence': avg_conf,
                'success': overall_success,
                'download_url': f'/download/{out_fn}',
                'original_filename': file.filename
            })

        return jsonify({'processed_files': processed_files, 'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True)

@app.route('/', methods=['GET'])
def index():
    return '後端運作中，請由前端發送 POST 請求至 /upload/'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
