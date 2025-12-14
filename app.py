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
        return None, "❌ Tidak ada gambar yang diupload"
    
    results = model.predict(image, conf=conf, verbose=False, imgsz=640)
    annotated = results[0].plot()
    annotated = cv2.CVtColor(annotated, cv2.COLOR_BGR2RGB)
    
    detections = {}
    for box in results[0].boxes:
        cls = model.names[int(box.cls[0])]
        detections[cls] = detections.get(cls, 0) + 1
    
    if not detections:
        text = "✅ Tidak ada kerusakan jalan terdeteksi"
    else:
        text = f"🎯 Total Deteksi: {sum(detections.values())}\n\n"
        for cls, count in detections.items():
            text += f"{CLASS_INFO.get(cls, cls)}: {count}x\n"
    
    return annotated, text

demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(label="Upload Gambar Jalan", type="numpy", sources=["upload"]),
        gr.Slider(0.1, 1.0, 0.25, 0.05, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(label="Hasil Deteksi"),
        gr.Textbox(label="Ringkasan Deteksi", lines=6)
    ],
    title="🛣️ Deteksi Kerusakan Jalan",
    description="📸 Upload gambar jalan untuk mendeteksi kerusakan seperti retak dan lubang",
    theme="soft",
    examples=[
        # Anda bisa menambahkan contoh gambar di sini jika ada
    ]
)

import os
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
