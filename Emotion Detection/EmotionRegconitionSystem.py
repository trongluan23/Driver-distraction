import cv2
from keras.models import load_model
import numpy as np

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model("emotion_detection_model.h5")
eye_tracking_model = load_model("eye_tracking_model.h5")

# Labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
eye_labels = ['left_look', 'right_look', 'forward_look', 'close_look']

def detect_and_predict_emotion(frame):
    face_images = cv2.resize(frame, (48, 48))
    face_images = cv2.cvtColor(face_images, cv2.COLOR_BGR2GRAY)
    face_images = np.expand_dims(face_images, axis=0)
    face_images = np.expand_dims(face_images, axis=-1)
    face_images = face_images.astype('float32') / 255.0
    predictions = emotion_model.predict(face_images)
    emotion_index = np.argmax(predictions[0])
    emotion_label = emotion_labels[emotion_index]
    return emotion_label

def detect_and_predict_eye_direction(eye_roi):
    eye_images = cv2.resize(eye_roi, (48, 48))
    eye_images = cv2.cvtColor(eye_images, cv2.COLOR_BGR2GRAY)
    eye_images = np.expand_dims(eye_images, axis=0)
    eye_images = np.expand_dims(eye_images, axis=-1)
    eye_images = eye_images.astype('float32') / 255.0
    predictions = eye_tracking_model.predict(eye_images)
    eye_index = np.argmax(predictions[0])
    eye_label = eye_labels[eye_index]
    return eye_label

cap = cv2.VideoCapture(0)  # Use 0 for the default camera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = frame[y:y + h, x:x + w]
        emotion_label = detect_and_predict_emotion(face_roi)

        # Detect eyes within the face region
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye_roi = face_roi[ey:ey + eh, ex:ex + ew]
            eye_label = detect_and_predict_eye_direction(eye_roi)
            cv2.putText(frame, eye_label, (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Emotion and Eye Direction Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()