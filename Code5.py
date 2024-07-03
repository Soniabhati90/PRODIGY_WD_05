import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_images_from_folder(folder, image_size):
    images = []
    labels = []
    for food_item in os.listdir(folder):
        food_item_folder = os.path.join(folder, food_item)
        for filename in os.listdir(food_item_folder):
            img = cv2.imread(os.path.join(food_item_folder, filename))
            if img is not None:
                img = cv2.resize(img, (image_size, image_size))
                images.append(img)
                labels.append(food_item)
    return images, labels

data_folder = 'path/to/food_dataset'

image_size = 64

images, labels = load_images_from_folder(data_folder, image_size)

images = np.array(images)
labels = np.array(labels)

images = images / 255.0

label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
labels = np.array([label_mapping[label] for label in labels])
labels = to_categorical(labels, num_classes=len(label_mapping))

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_mapping), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=64)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', Validation'], loc='upper left')
plt.show()

model.save('food_recognition_model.h5')

calorie_mapping = {
    'apple': 52,
    'banana': 89,
    'burger': 295,
    'pizza': 266,
    'salad': 33,
    'sushi': 145
}

model = tf.keras.models.load_model('food_recognition_model.h5')

def predict_food_and_calories(image_path, model, label_mapping, calorie_mapping, image_size):
   
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (image_size, image_size))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, image_size, image_size, 3))

    prediction = model.predict(img_reshaped)
    food_idx = np.argmax(prediction)
    food_item = [key for key, value in label_mapping.items() if value == food_idx][0]

    calories = calorie_mapping.get(food_item, "Unknown")

    return food_item, calories

image_path = 'path/to/food_image.jpg'
food_item, calories = predict_food_and_calories(image_path, model, label_mapping, calorie_mapping, image_size)
print(f"Predicted food item: {food_item}, Estimated calories: {calories}")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    food_item, calories = predict_food_and_calories(frame, model, label_mapping, calorie_mapping, image_size)

    cv2.putText(frame, f'Food: {food_item}, Calories: {calories}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Food Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
