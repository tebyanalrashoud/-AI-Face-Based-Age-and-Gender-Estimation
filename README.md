#  AI Face-Based Age and Gender Estimation

A smart and elegant **AI-powered system** that detects human faces in an image and intelligently **predicts both age and gender** using deep learning models.

Built using **Python, OpenCV, and DNN-based pretrained models**, this project showcases real-time inference on static images and can be easily adapted for webcam or video input.

---

<p align="center">
 
  <img src="https://github.com/user-attachments/assets/0beaab77-2ea2-4e8d-8fce-3c071f253eb9" width="600" hight="600"/>
</p>




---

## 🚀 Features

-  **Face Detection** using Haar Cascade Classifier  
- **Age Prediction** using OpenCV DNN with pre-trained Caffe model  
- **Gender Classification** using deep learning  
- Supports **static image input** with bounding box annotations  
- Shows **age and gender directly on image**  
- Lightweight and easy to extend for webcam or live video  
- Ready to be deployed or integrated into larger AI pipelines  

---

##  How It Works

This system uses a two-step pipeline:

1. **Face Detection**  
   - Uses OpenCV's Haar cascade to locate faces in an image.

2. **Age and Gender Estimation**  
   - Crops face region  
   - Preprocesses input to required blob format  
   - Feeds into pre-trained **Caffe-based models**  
   - Outputs probabilities for **gender** (male/female) and **age range**  

---

##  Example Output

The system draws a rectangle around the detected face and displays the predicted **gender and age** above it.

