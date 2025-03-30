import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
model = YOLO("final_model_garbage.pt")
st.title("Garbage Classification Model")


cap = cv2.VideoCapture(0)

# Create a placeholder for the video feed
video_placeholder = st.empty()

while True:
    ret, frame = cap.read()
    
    if not ret:
        st.write("Error: Failed to capture frame")
        break

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


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

