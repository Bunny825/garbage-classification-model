import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image

# Load YOLO model
model = YOLO("final_model_garbage.pt")

st.title("Garbage Classification Model")

# Open the webcam
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

# Process video frames in real-time
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture video")
        break

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

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # Display frame in Streamlit
    frame_placeholder.image(img, caption="Live Video Feed", use_column_width=True)

    # Stop video processing if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
