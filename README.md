# CNN for MNIST Handwritten Digit Classification

A Deep Learning project using a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset with high accuracy.

---

# 📌 Project Overview

This project demonstrates how CNNs can be used for image classification tasks.
The model is trained on the MNIST dataset containing handwritten digits from 0 to 9.

The project includes:

* Dataset loading and preprocessing
* CNN model creation
* Model training and validation
* Accuracy and loss analysis
* Prediction and testing
* Output visualization

---

# 🧠 Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

# 📂 Dataset

The MNIST dataset contains:

* 60,000 training images
* 10,000 testing images
* 28 × 28 grayscale handwritten digit images

Dataset is directly loaded using Keras:

```python id="f0q8hk"
from tensorflow.keras.datasets import mnist
```

---

# ⚙️ Data Preprocessing

Steps performed:

1. Normalize pixel values (0–255 → 0–1)
2. Reshape images for CNN input
3. Split training and validation data

Example:

```python id="p2j08f"
X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
```

---

# 🏗️ CNN Architecture

The CNN model consists of:

* Convolution layers
* MaxPooling layers
* Flatten layer
* Dense layers
* Softmax output layer

```python id="a9m1dc"
model = keras.Sequential([
    
    keras.layers.Conv2D(32,(3,3),activation='relu',
                        input_shape=(28,28,1)),
    
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Flatten(),
    
    keras.layers.Dense(64,activation='relu'),
    
    keras.layers.Dense(10,activation='softmax')
])
```

---

# 🚀 Model Training

```python id="m1x15z"
model_fit = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_split=0.2
)
```

---

# 📊 Accuracy and Loss Analysis

The project visualizes:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

### Accuracy Graph

```python id="6b7r6f"
plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['val_accuracy'])

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Training','Validation'])

plt.show()
```

### Loss Graph

```python id="cfm8q9"
plt.plot(model_fit.history['loss'])
plt.plot(model_fit.history['val_loss'])

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend(['Training','Validation'])

plt.show()
```

---

# 🔍 Prediction and Testing

```python id="7lwyuj"
y_pred = model.predict(X_test)
```

Convert probabilities into predicted labels:

```python id="6i0y1n"
y_pred_labels = [np.argmax(i) for i in y_pred]
```

---

# 📈 Output Visualization

The project includes:

* Sample image prediction
* Accuracy graph
* Loss graph
* Confusion matrix visualization

Example:

```python id="y4of0l"
plt.imshow(X_test[0].reshape(28,28), cmap='gray')

plt.title(f"Predicted Digit: {y_pred_labels[0]}")

plt.show()
```

---

# ✅ Results

* Achieved high classification accuracy on the MNIST dataset
* Successfully recognized handwritten digits
* CNN performed efficiently for image classification tasks

---

# 📌 Conclusion

This project demonstrates the effectiveness of Convolutional Neural Networks in image recognition tasks. CNN automatically extracts image features and provides accurate handwritten digit classification.

---

# 🔗 Future Improvements

* Add Dropout layers to reduce overfitting
* Use Data Augmentation
* Experiment with deeper CNN architectures
* Deploy model using Flask or Streamlit

---

# 👨‍💻 Author

Deep Learning CNN Project using the MNIST dataset.
