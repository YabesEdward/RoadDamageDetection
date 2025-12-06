import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best.pt')

CLASS_INFO = {
    'Longitudinal_Crack': '🔹 Retak Memanjang',
    'Transverse_Crack': '↔️ Retak Melintang',
    'Alligator_Crack': '🐊 Retak Buaya',
    'Pothole': '🕳️ Lubang'
}

def detect(image, conf):
    if image is None:
        return None, "No image"
    
    results = model.predict(image, conf=conf, verbose=False, imgsz=640)
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    detections = {}
    for box in results[0].boxes:
        cls = model.names[int(box.cls[0])]
        detections[cls] = detections.get(cls, 0) + 1
    
    if not detections:
        text = "✅ No damage detected"
    else:
        text = f"🎯 Total: {sum(detections.values())}\n\n"
        for cls, count in detections.items():
            text += f"{CLASS_INFO.get(cls, cls)}: {count}x\n"
    
    return annotated, text

demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(label="Upload or Use Webcam", sources=["upload", "webcam"]),
        gr.Slider(0.1, 1.0, 0.25, 0.05, label="Confidence")
    ],
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Results", lines=5)
    ],
    title="🛣️ Road Damage Detection",
    description="Upload image or use webcam to detect road damage"
)

demo.launch(server_name="0.0.0.0", server_port=7860)