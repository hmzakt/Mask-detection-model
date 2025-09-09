import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from model import ResNet18Clone
from pathlib import Path

# ----------------------------
# Device and Model Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18Clone(num_classes=2)
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "face_mask_detection_model.pth"


model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Preprocessing
# ----------------------------
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),   # Resize to match training
    transforms.ToTensor(),           
    transforms.Normalize(              
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ["with_mask", "without_mask"]

# ----------------------------
# Face Detector
# ----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # type: ignore


# ----------------------------
# Webcam Loop
# ----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop face
        face_img = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        # Preprocess and predict
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device) # type: ignore
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            label = class_names[preds[0].item()] # type: ignore

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
