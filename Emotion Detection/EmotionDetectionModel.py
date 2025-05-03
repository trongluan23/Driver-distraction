import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

emotion_folder = [
    r"C:\Users\GIGABYTE\Desktop\Emotion Detection\train\angry",
    r"C:\Users\GIGABYTE\Desktop\Emotion Detection\train\disgust",
    r"C:\Users\GIGABYTE\Desktop\Emotion Detection\train\fear",
    r"C:\Users\GIGABYTE\Desktop\Emotion Detection\train\happy",
    r"C:\Users\GIGABYTE\Desktop\Emotion Detection\train\neutral",
    r"C:\Users\GIGABYTE\Desktop\Emotion Detection\train\sad", 
    r"C:\Users\GIGABYTE\Desktop\Emotion Detection\train\surprise"
    ]# Replace with the path to your dataset

def load_data(emotion_folder):
    images = []
    labels = []

    for label, folder in enumerate(emotion_folder):
        for filename in os.listdir(folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))  # Resize to 48x48
                images.append(img)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Normalize pixel values to be between 0 and 1
    images = images.astype('float32') / 255.0

    # Reshape images to add a channel dimension
    images = np.expand_dims(images, axis=-1) # (num_samples, 48, 48, 1)

    return images, labels

images, labels = load_data(emotion_folder)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))
model.save('C:/Users/GIGABYTE/Desktop/Driver-distraction/Trainer/emotion_detection_model.h5')


