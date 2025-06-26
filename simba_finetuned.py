import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime
import csv
import pandas as pd
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from concurrent.futures import ProcessPoolExecutor

IMAGE_DIR = "D:/Adversarial-Example/funtionTest/simba/facesDATA"
SAVE_DIR = "D:/Adversarial-Example/funtionTest/simba/adv_faces"
RESULT_DIR = "D:/Adversarial-Example/funtionTest/simba/attack_results"
YOLO_WEIGHTS = "D:/Adversarial-Example/model/theNewYOLO/best.pt"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
yolo_model = YOLO(YOLO_WEIGHTS)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_embedding_direct(pil_image):
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet(img_tensor)
    return F.normalize(embedding, p=2, dim=1)

def cosine_similarity(a, b):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return (a * b).sum(dim=1)

def to_pil(tensor_img):
    img = tensor_img.squeeze(0).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)

def simba_attack(image_tensor, original_embedding,
                 max_iters, epsilon, block_size=1,
                 early_stop_threshold=0.7,
                 filename="", face_id=0):
    image = image_tensor.clone().detach().to(device)
    perturb = torch.zeros_like(image, device=device)
    flat_dim = perturb.numel()
    indices = torch.randperm(flat_dim // block_size) * block_size

    best_dist = 1 - cosine_similarity(get_embedding_direct(to_pil(image)), original_embedding).item()
    best_perturb = perturb.clone()
    history = [best_dist]
    snapshot_interval = 1000

    for i in tqdm(range(min(max_iters, len(indices))),
                  desc=f"[{filename}][face {face_id}] SimBA", leave=False):
        idx = indices[i].item()
        if idx + block_size > flat_dim:
            continue

        perturb_flat = perturb.view(-1)

        for j in range(block_size): perturb_flat[idx + j] += epsilon
        plus_img = torch.clamp(image + perturb, -1, 1)
        dist_plus = 1 - cosine_similarity(get_embedding_direct(to_pil(plus_img)), original_embedding)

        for j in range(block_size): perturb_flat[idx + j] -= 2 * epsilon
        minus_img = torch.clamp(image + perturb, -1, 1)
        dist_minus = 1 - cosine_similarity(get_embedding_direct(to_pil(minus_img)), original_embedding)

        for j in range(block_size): perturb_flat[idx + j] += epsilon
        direction = epsilon if dist_plus > dist_minus else -epsilon
        for j in range(block_size): perturb_flat[idx + j] += direction

        current_img = torch.clamp(image + perturb, -1, 1)
        current_dist = 1 - cosine_similarity(get_embedding_direct(to_pil(current_img)), original_embedding).item()
        history.append(current_dist)

        if (i + 1) % snapshot_interval == 0 or current_dist >= early_stop_threshold:
            save_dir = os.path.join(SAVE_DIR, f"{filename}_face{face_id}")
            os.makedirs(save_dir, exist_ok=True)

            adv_snapshot = to_pil(current_img)
            adv_snapshot.save(os.path.join(save_dir, f"adv_iter_{i+1}.jpg"))

            perturb_np = perturb.squeeze(0).detach().cpu().numpy()
            perturb_np = np.transpose(perturb_np, (1, 2, 0))
            mask_norm = ((perturb_np - perturb_np.min()) / (perturb_np.ptp() + 1e-8) * 255)
            mask_img = Image.fromarray(mask_norm.clip(0,255).astype(np.uint8))
            mask_img.save(os.path.join(save_dir, f"perturb_mask_iter_{i+1}.png"))

            diff = current_img - image
            diff_np = diff.squeeze(0).detach().cpu().numpy()
            diff_np = np.transpose(diff_np, (1, 2, 0))
            diff_norm = ((diff_np - diff_np.min()) / (diff_np.ptp() + 1e-8) * 255)
            diff_img = Image.fromarray(diff_norm.clip(0,255).astype(np.uint8))
            diff_img.save(os.path.join(save_dir, f"diff_only_iter_{i+1}.png"))

        if current_dist > best_dist:
            best_dist = current_dist
            best_perturb = perturb.clone()
        if current_dist >= early_stop_threshold:
            break

    history_csv = os.path.join(RESULT_DIR, f"{filename}_face{face_id}_history.csv")
    os.makedirs(os.path.dirname(history_csv), exist_ok=True)
    with open(history_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', '1_minus_cosine_similarity'])
        for iter_idx, dist in enumerate(history): writer.writerow([iter_idx, dist])

    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("1 - Cosine Similarity")
    plt.title("SimBA Attack Progress")
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, f"simba_progress_{filename}_face{face_id}.png"))
    plt.close()

    return torch.clamp(image + best_perturb, -1, 1), best_dist, len(history) - 1, current_dist >= early_stop_threshold

def apply_yolo_crop(image_path):
    results = yolo_model(image_path)[0]
    orig_img = Image.open(image_path).convert("RGB")
    faces = []
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        faces.append((orig_img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)))
    return orig_img, faces

def process_image(filename):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")): return
    path = os.path.join(IMAGE_DIR, filename)
    try:
        original_img, faces = apply_yolo_crop(path)
        if not faces: print(f"[Warning] No face detected in {filename}"); return

        summary_rows = []
        start_time = time.time()
        for i, (face, box) in enumerate(faces):
            image_tensor = transform(face).unsqueeze(0).to(device)
            original_embedding = get_embedding_direct(face)
            before_sim = cosine_similarity(original_embedding, original_embedding).item()

            adv_tensor, best_dist, iters, early_stopped = simba_attack(
                image_tensor, original_embedding, max_iters=100000,
                epsilon=0.001, filename=filename.split('.')[0], face_id=i
            )

            adv_pil = to_pil(adv_tensor)
            adv_embedding = get_embedding_direct(adv_pil)
            after_sim = cosine_similarity(original_embedding, adv_embedding).item()
            save_path = os.path.join(SAVE_DIR, f"adv_face_{i}_{filename}")
            adv_pil.save(save_path)

            elapsed = time.time() - start_time
            summary_rows.append({
                "filename": filename,
                "face_id": i,
                "before_sim": before_sim,
                "after_sim": after_sim,
                "best_dist": best_dist,
                "iterations": iters,
                "early_stopped": early_stopped,
                "time_sec": round(elapsed, 2)
            })

        summary_df = pd.DataFrame(summary_rows)
        excel_path = os.path.join(RESULT_DIR, "attack_summary.xlsx")
        if os.path.exists(excel_path): summary_df = pd.concat([pd.read_excel(excel_path), summary_df], ignore_index=True)
        summary_df.to_excel(excel_path, index=False)
    except Exception as e:
        print(f"[Error] Failed to process {filename}: {str(e)}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ CUDA 可用！目前使用的 GPU 裝置：{torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("❌ CUDA 不可用，已切換至 CPU 模式。")

files = os.listdir(IMAGE_DIR)
for file in tqdm(files, desc="Total Progress"): process_image(file)
