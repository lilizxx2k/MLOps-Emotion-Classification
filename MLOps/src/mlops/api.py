from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image
import torch
from loguru import logger
from torchvision import transforms
from pydantic import BaseModel

# import model and labels
from mlops.model import Model
from mlops.labels import IDX_TO_CLASS


app = FastAPI(title="Emotion Classification API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

logger.info(f"Using device: {device}")

class InferenceResponse(BaseModel):
    emotion: str
    confidence: float

# preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # add batch dimension
    return tensor.to(device)

# prediction endpoint 
@app.on_event("startup")
def load_model():
    global model
    try:
        model = Model(output_dim=8)  # 8 classes
        model.load_state_dict(
            torch.load("models/TrainedModel.pth", map_location=device)
        )
        model.to(device)
        model.eval()
        logger.success("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


@app.post("/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

        emotion = IDX_TO_CLASS[pred_idx]
        logger.info(f"Prediction: {emotion} (confidence: {confidence:.3f})")

        return InferenceResponse(
            emotion=emotion,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    logger.info("Health check requested")
    return {"status": "ok"}
