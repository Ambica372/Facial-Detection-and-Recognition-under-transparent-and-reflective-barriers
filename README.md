# Facial Detection and Recognition Under Barriers  
### MSRUAS – Final Research Project

## Overview
This project presents a robust AI-based facial detection and recognition pipeline designed to operate under challenging visual conditions caused by **transparent and reflective barriers** such as glass or plastic. These barriers introduce atmospheric noise, glare, and light reflections that typically degrade the performance of conventional face recognition systems.

The proposed system overcomes these challenges through specialized image preprocessing, adaptive face detection, and deep feature-based recognition.

---

## Project Highlights
- **High Accuracy:** Achieved a verification score of **1.0000** using the ArcFace recognition model.
- **Robust Pre-processing:** Uses a custom **Laplacian sharpening kernel** to enhance facial features and recover landmarks obscured by reflections.
- **Adaptive Detection:** Implements YOLOv8 with a reduced detection threshold (**0.3**) to localize faces in low-contrast and reflective environments.
- **Barrier-aware Recognition:** Maintains recognition reliability despite reflections and transparency effects.

---

## Repository Structure
- **`final_project.ipynb`** – Primary research notebook containing the complete 5-module AI pipeline  
- **`MSRUAS_Success_Output.png`** – Visual proof of final system output with verified similarity score  
- **`yolov8n.pt`** – Pre-trained YOLOv8 weights used for adaptive face detection  
- **`test_face.jpg`** – Standardized input image used for experimental validation  

---

## Technical Pipeline

### Module 1 & 2: Image Acquisition and Reconstruction  
- Image enhancement using **weighted fusion**
- Noise reduction and contrast improvement under reflective conditions  

### Module 3: Adaptive Face Detection  
- YOLOv8-based face detection  
- Detection threshold tuned to **0.3** for low-contrast environments  

### Module 4: Face Alignment and Landmark Recovery  
- Accurate facial landmark localization
- Alignment for consistent embedding extraction  

### Module 5: Face Recognition  
- 512-dimensional **ArcFace embeddings**
- Identity verification using **cosine similarity**

---

## Results
- Successful face detection and recognition through transparent barriers
- Perfect similarity score (1.0000) under experimental conditions
- Demonstrates robustness against reflection-induced distortion

---

## Applications
- Secure access systems behind glass barriers
- Surveillance in controlled indoor environments
- Research in robust biometric recognition
- Barrier-aware computer vision systems

---

## Authors
- **Ambica Natraj**  
- **Sudharani**

**Department of Computer Science and Engineering**  
**M. S. Ramaiah University of Applied Sciences (MSRUAS)**
