import os
import cv2
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Load the YOLOv8 model for hand sign recognition
chosen_model = YOLO("Model/HandSign.pt")  # Update the path to your hand sign model

# Load COCO labels
with open("Model/coco_label.txt", "r") as f:
    classes = f.read().splitlines()

@st.cache_data()
def predict(_chosen_model, img, conf=0.5):
    # Resize the image to 640x480
    img = cv2.resize(img, (640, 480))
    results = _chosen_model.predict(img, conf=conf, save_txt=False)

    return results

def predict_and_detect(chosen_model, img, conf=0.5):
    # Resize the image to 640x480
    img = cv2.resize(img, (640, 480))
    results = predict(chosen_model, img, conf=conf)

    detected_signs = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = classes[class_id]
            confidence = box.conf[0]
            if confidence >= conf:
                detected_signs.append(class_name)
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
                cv2.putText(img, str(class_name), (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
    return img, detected_signs

def process_frame(frame):
    result_frame, detected_signs = predict_and_detect(chosen_model, frame)
    return result_frame, detected_signs

def main():
    st.title("Hand Sign Language Recognition")

    option = st.selectbox("Select Option", ["Upload Video", "Upload Image", "Live Camera"])

    if option == "Live Camera":
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        detected_signs_text = st.empty()
        stop_button = st.button("Stop Live Camera")
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_frame, detected_signs = process_frame(frame_rgb)
            stframe.image(result_frame)
            detected_signs_text.markdown(f"**Detected Signs:** {', '.join(detected_signs)}")
        cap.release()

    elif option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            img_array = np.array(image)
            result_frame, detected_signs = process_frame(img_array)
            st.image(result_frame)
            st.markdown(f"**Detected Signs:** {', '.join(detected_signs)}")

    elif option == "Upload Video":
      uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
      if uploaded_file is not None:
          with tempfile.NamedTemporaryFile(delete=False) as temp_file:
              temp_file.write(uploaded_file.read())
              temp_file_path = temp_file.name

          cap = cv2.VideoCapture(temp_file_path)
          stframe = st.empty()
          detected_signs_text = st.empty()
          while cap.isOpened():
              ret, frame = cap.read()
              if not ret:
                  break
              frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              result_frame, detected_signs = process_frame(frame_rgb)
              stframe.image(result_frame)
              detected_signs_text.markdown(f"**Detected Signs:** {', '.join(detected_signs)}")
          cap.release()

          # Delete the temporary file
          os.unlink(temp_file_path)

if __name__ == '__main__':
    main()