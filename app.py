import cv2
import numpy as np
import gradio as gr
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Initialize InsightFace model
app_model = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)
app_model.prepare(ctx_id=0, det_thresh=0.3)

def reflection_aware_preprocessing(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    sharpened = cv2.filter2D(img_rgb, -1, kernel)

    enhanced = cv2.addWeighted(
        img_rgb, 0.7,
        sharpened, 0.3,
        0
    )

    return enhanced

def verify_faces(img1, img2):
    if img1 is None or img2 is None:
        return "Please upload both images."

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    img1_proc = reflection_aware_preprocessing(img1)
    img2_proc = reflection_aware_preprocessing(img2)

    faces1 = app_model.get(img1_proc)
    faces2 = app_model.get(img2_proc)

    if len(faces1) == 0 or len(faces2) == 0:
        return "Face not detected in one or both images."

    emb1 = faces1[0].embedding.reshape(1, -1)
    emb2 = faces2[0].embedding.reshape(1, -1)

    similarity = cosine_similarity(emb1, emb2)[0][0]

    threshold = 0.5

    if similarity > threshold:
        result = "Same Person"
    else:
        result = "Different Person"

    return f"Cosine Similarity: {similarity:.4f}\nResult: {result}"

interface = gr.Interface(
    fn=verify_faces,
    inputs=[
        gr.Image(type="numpy", label="Upload Image 1"),
        gr.Image(type="numpy", label="Upload Image 2")
    ],
    outputs="text",
    title="Reflection-Aware Face Verification",
    description="Face matching robust to reflections or glass distortions."
)

if __name__ == "__main__":
    interface.launch()
