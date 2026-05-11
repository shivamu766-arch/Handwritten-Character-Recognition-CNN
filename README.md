# Handwritten-Character-Recognition-CNN
# =========================================================
# HANDWRITTEN CHARACTER RECOGNITION USING CNN
# ALL-IN-ONE SINGLE FILE PROJECT
# =========================================================

# INSTALL REQUIRED LIBRARIES FIRST:
# pip install tensorflow opencv-python flask pillow numpy matplotlib

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template_string
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# =========================================================
# LOAD DATASET
# =========================================================

print("Loading Dataset...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# One-hot encoding
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

print("Dataset Loaded Successfully!")

# =========================================================
# BUILD CNN MODEL
# =========================================================

print("Building CNN Model...")

model = Sequential()

# First Convolution Layer
model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(28,28,1)
))

model.add(MaxPooling2D((2,2)))

# Second Convolution Layer
model.add(Conv2D(
    64,
    (3,3),
    activation='relu'
))

model.add(MaxPooling2D((2,2)))

# Flatten Layer
model.add(Flatten())

# Dense Layer
model.add(Dense(128, activation='relu'))

# Dropout Layer
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(10, activation='softmax'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("CNN Model Built Successfully!")

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTraining Started...\n")

model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test,y_test),
    batch_size=64
)

print("\nTraining Completed!")

# =========================================================
# SAVE MODEL
# =========================================================

model.save("handwritten_cnn_model.h5")

print("\nModel Saved Successfully!")

# =========================================================
# TEST ACCURACY
# =========================================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy * 100, "%")

# =========================================================
# FLASK WEB APPLICATION
# =========================================================

app = Flask(__name__)

# Load trained model
model = load_model("handwritten_cnn_model.h5")

HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Handwritten Character Recognition</title>

    <style>

        body{
            background:#0f172a;
            color:white;
            text-align:center;
            font-family:Arial;
            padding-top:50px;
        }

        h1{
            color:#38bdf8;
        }

        form{
            margin-top:30px;
        }

        input{
            padding:10px;
            background:white;
        }

        button{
            padding:10px 20px;
            background:#38bdf8;
            border:none;
            color:white;
            cursor:pointer;
            margin-top:20px;
            border-radius:10px;
        }

        .result{
            margin-top:30px;
            font-size:30px;
            color:#4ade80;
        }

    </style>

</head>
<body>

    <h1>Handwritten Digit Recognition using CNN</h1>

    <form method="POST" enctype="multipart/form-data">

        <input type="file" name="file">

        <br>

        <button type="submit">Predict</button>

    </form>

    {% if prediction %}

    <div class="result">

        Prediction : {{prediction}}

    </div>

    {% endif %}

</body>
</html>
'''

# =========================================================
# HOME ROUTE
# =========================================================

@app.route('/', methods=['GET','POST'])

def home():

    prediction = None

    if request.method == 'POST':

        file = request.files['file']

        if file:

            filepath = "temp.png"

            file.save(filepath)

            # Read Image
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # Resize
            img = cv2.resize(img,(28,28))

            # Invert colors
            img = 255 - img

            # Normalize
            img = img / 255.0

            # Reshape
            img = img.reshape(1,28,28,1)

            # Prediction
            pred = model.predict(img)

            prediction = np.argmax(pred)

    return render_template_string(
        HTML_PAGE,
        prediction=prediction
    )

# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("\n===================================")
    print("SERVER STARTED SUCCESSFULLY")
    print("===================================")

    print("\nOpen Browser:")
    print("http://127.0.0.1:5000")

    app.run(debug=True)
