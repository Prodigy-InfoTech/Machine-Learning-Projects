import tensorflow as tf
import numpy as np
import cv2

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

base_model = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet', input_shape=(32, 32, 3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

def classify_sample_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    return class_idx

sample_image_path = "airplaneImage.jpg"
predicted_class = classify_sample_image(sample_image_path, model)

class_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print(f"Predicted class: {class_labels[predicted_class]}")
