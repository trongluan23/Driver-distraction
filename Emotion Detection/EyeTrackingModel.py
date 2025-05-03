import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Thư mục chứa dữ liệu cho 4 hướng nhìn
left_eye_folder = r"C:\Users\GIGABYTE\Desktop\Driver-distraction\Emotion Detection\eyedataset\Eye dataset\left_look"
right_eye_folder = r"C:\Users\GIGABYTE\Desktop\Driver-distraction\Emotion Detection\eyedataset\Eye dataset\right_look"
forward_eye_folder = r"C:\Users\GIGABYTE\Desktop\Driver-distraction\Emotion Detection\eyedataset\Eye dataset\forward_look"
close_eye_folder = r"C:\Users\GIGABYTE\Desktop\Driver-distraction\Emotion Detection\eyedataset\Eye dataset\close_look"
# Thay đổi đường dẫn đến thư mục chứa dữ liệu của bạn

def load_data(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (24, 24))  # Thay đổi kích thước thành 24x24
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Tải dữ liệu từ 4 thư mục
images_left, labels_left = load_data(left_eye_folder, 0)  # Nhãn 0: left_look
images_right, labels_right = load_data(right_eye_folder, 1)  # Nhãn 1: right_look
images_forward, labels_forward = load_data(forward_eye_folder, 2)  # Nhãn 2: forward_look
images_close, labels_close = load_data(close_eye_folder, 3)  # Nhãn 3: close_look

# Kết hợp dữ liệu từ 4 hướng
images_combined = np.concatenate((images_left, images_right, images_forward, images_close), axis=0)
labels_combined = np.concatenate((labels_left, labels_right, labels_forward, labels_close), axis=0)

# Chuẩn hóa dữ liệu
images_normalized = images_combined.reshape(-1, 24, 24, 1).astype('float32') / 255  # Thay đổi kích thước thành 24x24x1
labels_categorical = to_categorical(labels_combined, num_classes=4)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(images_normalized, labels_categorical, test_size=0.2, random_state=42)

# Xây dựng mô hình CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)))  # Thay đổi input_shape thành 24x24x1
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))  # 4 lớp đầu ra cho 4 hướng

# Biên dịch và huấn luyện mô hình
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, y_test))

# Lưu mô hình
model.save("eye_tracking_model.h5")