# Automatic License Plate Recognition (ALPR) using YOLOv8 + EasyOCR

##  Overview

This project performs **Automatic License Plate Recognition (ALPR)** from images or videos using:

- **YOLOv8** for vehicle + number plate detection
- **EasyOCR** for reading number plate text
- Optional tracking for smoother visualization

The system detects vehicles, locates the number plates, performs text recognition, and returns results in video/image form along with structured plate data.

---

##  Core Features

✔ Detects vehicles using **YOLOv8** pretrained model  
✔ Detects number plates using a **custom YOLOv8 plate model**  
✔ Extracts text from plates using **EasyOCR**  
✔ Works on **images & videos**  
✔ Outputs:  
   - Processed image/video with overlays  
   - JSON/CSV containing recognized plates  
✔ Suitable for **traffic systems**, **parking automation**, **law enforcement**, etc.

---

##  Architecture

1. **Vehicle Detection (YOLOv8n Pretrained)**
2. **Plate Detection (Custom YOLOv8 Model)**
3. **Text Recognition (EasyOCR)**
4. **(Optional) Tracking & Post Processing**
5. **Output Generation**

---

## Project Flow

1. **Set up Environment**  
   - Create Python environment 
   - Install dependencies  

2. **Prepare Models**  
   - `yolov8n.pt` → vehicle detection  
   - `license_plate_detector.pt` → license plate detection  

3. **Provide Input**  
   - Replace the sample video/image with your own file  
   - Run `main.py`  

4. **Run Detection Pipeline**  
   - Detect vehicles using YOLOv8  
   - Track vehicles using SORT  
   - Detect license plates and crop the plate region  
   - Recognize text using EasyOCR  

5. **Save Results**  
   - Output CSV with frame, vehicle ID, plate text, and bounding boxes  

6. ** Smoothing**  
   - Run `PreProcessingCSV.py` to fill missing frames and smooth tracking  

7. **Visualize Output**  
   - Run `Output.py` to generate final processed video or images  



