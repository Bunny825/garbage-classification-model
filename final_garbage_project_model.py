import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2

# Load YOLO model
model = YOLO("final_model_garbage.pt")

st.title("Garbage Classification Model")

# Camera input (for Streamlit Cloud)
image = st.camera_input("Take a picture")

if image is not None:
    # Convert image to OpenCV format
    img = Image.open(image)
    frame = np.array(img)

    # Run YOLO inference
    results = model(frame)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]}: {conf:.2f}"
            
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert back to PIL image for display
    frame_rgb = Image.fromarray(frame)
    st.image(frame_rgb, caption="Processed Image", use_column_width=True)
