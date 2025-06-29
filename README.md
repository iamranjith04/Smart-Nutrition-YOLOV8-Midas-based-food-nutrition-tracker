# Smart-Nutrition-YOLOV8-Midas-based-food-nutrition-tracker
[Click here to watch the demo video](./demo.mp4)

This project estimates food nutrition (e.g., calories, weight, volume) from a **single image**, using a combination of:

- **YOLOv8** for food and reference object detection
- **MiDaS** (monocular depth estimation)
- **Pinhole Camera Model** with known reference object (e.g., credit card) for accurate size/scale

---

## 🚀 Key Features

- 📷 **One Image Input** — No need for dual-camera or depth sensors
- 🍔 **Food Detection using YOLOv8**, trained on [Food-101 dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
- 💳 **Reference Object Detection** (e.g., credit card) using YOLOv8, trained on [Credit Card Dataset](https://github.com/soumyajit4419/Credit-Card-Detection)
- 🔎 **Depth Estimation using MiDaS** (Monocular Depth Estimation)  
- 📐 **Volume and Size Estimation** using the Pinhole Camera Model with known object size and camera focal length
- 📊 **Calorie/Nutrition Estimation** using computed weight and nutritional data

---

## 🧠 Project Architecture
![image](https://github.com/user-attachments/assets/935a6f7d-347b-4fde-bacf-d531afbe18fa)

## 📷 How it Works

1. User uploads an image containing food and a reference object (like a credit card).
2. YOLOv8 detects all objects, bounding boxes are generated.
3. MiDaS predicts depth map of the image.
4. Using the known size of the reference object and focal length, the real-world size of food is computed.
5. Based on food volume and its density, weight is estimated.
6. Caloric value is estimated using known nutritional information.

---

## ⚙️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/iamranjith04/Smart-Nutrition-YOLOV8-Midas-based-food-nutrition-tracker.git

cd Smart-Nutrition-YOLOV8-Midas-based-food-nutrition-tracker

# Install dependencies
pip install -r requirements.txt

# Place your trained model weights inside the 'Models/' folder

