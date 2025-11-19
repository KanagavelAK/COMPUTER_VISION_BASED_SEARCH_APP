# COMPUTER_VISION_BASED_SEARCH_APP
##  Image Search Application using YOLOv11 & Streamlit

This project is a Computer Vision–based Image Search System that allows users to upload an image, detect objects using a YOLOv11 model, and search through a local dataset based on detected objects. The application is built with Streamlit, making it easy to use through a web interface.

The app supports:

Image Upload

Object Detection (YOLOv11)

Similar Image Search

Visualization of Detection & Metadata

Local filesystem-based search

## Features
### 1. Upload Image

Users can upload any image (JPEG/PNG).
The app automatically performs YOLOv11 inference.

### 2. Object Detection (YOLOv11)

Model: YOLOv11 (or YOLOv8/YOLOv5 depending on your model file)

### 3. Image Search Engine

Searches similar images from your local dataset based on:

Detected object classes

Confidence threshold

Metadata stored in JSON/YAML

### 4. Metadata Management

Each detected image saves:

Classes & counts

File path

Timestamp

Embeddings (optional)

### 5. Streamlit UI

Simple, fast, and interactive:

Sidebar inputs

Results gallery

Confidence threshold slider

## Project Structure
```
Yolov11_Image_search/
│
├── app.py                         # Streamlit application
├── configs/
│   └── default.yaml               # Configuration file
│
├── src/
│   ├── inference.py               # YOLOv11 inference wrapper
│   ├── config.py                  # Configuration loader
│   ├── utils.py                   # Helper functions
│   └── search.py                  # Search engine logic (if any)
│
├── data/
│   └── images/                    # Dataset of images for searching
│
├── outputs/
│   └── metadata.json              # Stored metadata from detections
│
└── requirements.txt
```
## Installation
1. Create a Python Environment
conda create -n cvsearch python=3.10
conda activate cvsearch

2. Install Requirements
pip install -r requirements.txt


If you don’t have a requirements.txt, here is a recommended one:

streamlit
numpy
opencv-python
pillow
pyyaml
ultralytics
scikit-learn

## Running the Application

From the project root, run:

streamlit run app.py


If the folder path includes spaces:

streamlit run "C:\Users\Admin\Desktop\deep yolo\Yolov11_Image_search\app.py"


The UI opens at:

http://localhost:8501

## How It Works

User uploads an image.

YOLOv11 model detects objects.

Detected classes are extracted.

Metadata is saved (class counts, filenames, etc.).

Search engine compares detected classes with dataset metadata.

Matching images are returned with confidence scores.

## Output:
<img width="1919" height="937" alt="image" src="https://github.com/user-attachments/assets/63483ef4-6039-4ec0-a4d9-9d01800ec8cf" />

<img width="1919" height="526" alt="image" src="https://github.com/user-attachments/assets/52ce54c7-dd93-463d-bfe1-b48d0f3f85c9" />

<img width="1912" height="852" alt="image" src="https://github.com/user-attachments/assets/318ef420-36f4-4147-b181-7bb5f1c179e2" />

<img width="1919" height="921" alt="image" src="https://github.com/user-attachments/assets/1246d930-8d9f-4672-bc0a-00c5a7ab0b7d" />

<img width="1919" height="926" alt="image" src="https://github.com/user-attachments/assets/c900d6d0-b098-4cf1-a996-901d349bbf4d" />

## Result:
Thus, the Image Search Application using YOLOv11 & Streamlit is successfully achieved.
