from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Đường dẫn dữ liệu
data_path = 'dataset/Eye/train'
directories = ['/closed', '/open']

# Hàm kiểm tra dữ liệu
def plot_img(dir, top=10):
    all_item_in_dir = os.listdir(dir)
    for file in all_item_in_dir:
        item_file = [os.path.join(dir, file)][:5]

for j in directories:
    plot_img(data_path + j)

# Tạo ImageDataGenerator
BS = 128
train_data = ImageDataGenerator(horizontal_flip=True, rescale=1./255, zoom_range=0.2, validation_split=0.1)
test_data = ImageDataGenerator(rescale=1./255)

train_data_path = 'dataset/Eye/train'
test_data_path = 'dataset/Eye/val'

train_set = train_data.flow_from_directory(train_data_path, target_size=(24, 24), batch_size=BS, color_mode='grayscale', class_mode='categorical')
test_set = test_data.flow_from_directory(test_data_path, target_size=(24, 24), batch_size=BS, color_mode='grayscale', class_mode='categorical')

# Xây dựng mô hình
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(24, 24, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

print(model.summary())

# Biên dịch mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Lưu mô hình tốt nhất
model_path = "Trainer/eye_detection.h5"
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]

# Huấn luyện mô hình
EPOCHS = 15
training_steps = train_set.n // train_set.batch_size
validation_steps = test_set.n // test_set.batch_size

history = model.fit(train_set, epochs=EPOCHS, steps_per_epoch=training_steps, validation_data=test_set, validation_steps=validation_steps, callbacks=callback_list)

# Vẽ biểu đồ loss và accuracy
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer: Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.savefig("plot.png")
plt.show()