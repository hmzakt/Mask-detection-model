# Face Mask Detection (PyTorch + OpenCV)

> A real-time face mask detection system using a custom ResNet18 deep learning model and OpenCV.  
> This project detects whether a person is wearing a mask or not using webcam input.

---

## ✨ Features
- Real-time face detection using OpenCV
- Classifies `Mask` / `No Mask` using a **ResNet18Clone** model
- Lightweight, fast inference
- Easily extendable for webcam or video feeds

---

## 🛠 Tech Stack
| Category | Technology |
|----------|------------|
| Deep Learning | ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?style=flat&logo=PyTorch&logoColor=white) |
| Computer Vision | ![OpenCV](https://img.shields.io/badge/OpenCV-%230064FF?style=flat&logo=opencv&logoColor=white) |
| Data Handling | ![NumPy](https://img.shields.io/badge/NumPy-%23013243?style=flat&logo=numpy&logoColor=white) |
| Image Processing | ![Pillow](https://img.shields.io/badge/Pillow-%23B72E26?style=flat&logo=python&logoColor=white) |

---

## 🗂 Dataset
- Collected and prepared face mask dataset with three classes:
  1. `with_mask`
  2. `without_mask`
- Split into **train / validation / test** sets
- Applied transforms: Resize, ToTensor, Normalize


## 📁 Project Structure  
vs/  
├── model/  
│ └── face_detection_model.pth # Trained model weights  
├── scripts/  
│ ├── model_def.py # ResNet18Clone architecture  
│ ├── load_model.py # Model loader  
│ ├── webcam_test.py # Test webcam feed  
│ └── inference.py # Live mask detection  
├── requirements.txt  


---

## ⚡ Installation
1. Clone the repo:
```bash
git clone <your-repo-url>
cd face_mask_detector

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt


# TEST WEBCAM FEED  
`python scripts/webcam_test.py`

