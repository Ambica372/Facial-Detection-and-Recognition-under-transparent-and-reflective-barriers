Facial Detection and Recognition Under Barriers
MSRUAS Final Research Project

This repository contains a robust AI pipeline designed to detect and recognize human faces through transparent and reflective barriers (such as glass or plastic). This project addresses the challenges of atmospheric noise and light reflection that typically cause standard facial recognition systems to fail.

🚀 Project Highlights
High Accuracy: Achieved a 1.0000 verification score using the ArcFace recognition model.

Robust Pre-processing: Implements a custom Laplacian sharpening kernel to recover facial landmarks obscured by reflections.

Adaptive Detection: Optimized with a reduced detection threshold of 0.3 to localize faces in low-contrast environments.

📂 Repository Structure
final_project.ipynb: The primary research notebook containing the full 5-module AI pipeline.

MSRUAS_Success_Output.png: Visual proof of the final system output with verified similarity.

yolov8n.pt: Pre-trained YOLOv8 weights used for the adaptive detection module.

test_face.jpg: Standardized input image used for research validation.

🛠️ Technical Pipeline
Module 1 & 2: Image Acquisition and Reconstruction using Weighted Fusion.

Module 3: Adaptive Face Detection using YOLOv8 with adjusted det_thresh.

Module 4: Face Alignment and Landmark recovery.

Module 5: Recognition using 512-D ArcFace embeddings and Cosine Similarity.

👥 Authors
Ambica

Sudharani

MSRUAS Department of Computer Science and Engineering
