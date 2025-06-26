from fastapi import FastAPI, File, UploadFile
from facenet_model import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import torch
import io
import uvicorn

app = FastAPI()

# FaceNet 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.post("/predict")
async def get_embedding(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor).squeeze(0).cpu().numpy()

    return {"embedding": embedding.tolist()}

if __name__ == "__main__":
    uvicorn.run("facenet_api_server:app", host="0.0.0.0", port=8000, reload=True)
