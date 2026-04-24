# Logic with UI
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

st.set_page_config(page_title="Driver Drowsiness Detection")

st.title("🚗 Driver Drowsiness Detection")

@st.cache_resource
def load_my_model():
    base_model = MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.build((None,224,224,3))
    model.load_weights("drowsiness_weights.weights.h5")

    return model

model = load_my_model()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

run = st.checkbox("Start Camera")

frame_window = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]

        img = cv2.resize(face,(224,224))
        img = img.astype("float32")
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)[0][0]

        if pred > 0.35:
            label = "Drowsy"
            color = (0,0,255)
        else:
            label = "Alert"
            color = (0,255,0)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

cap.release()
cv2.destroyAllWindows()
