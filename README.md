# -Neural-Networks-Deep-Learning-Using-Keras-Convolutional-NNs-in-Python-to-create-an-MNIST-model
To build a Convolutional Neural Network (CNN) using Keras in Python for classifying the MNIST dataset, follow these steps. We'll use Keras (with TensorFlow as the backend) to define and train a CNN model.
Steps:

    Import libraries: We'll need Keras and some basic libraries for data manipulation and visualization.
    Load the MNIST dataset: The MNIST dataset contains 70,000 images of handwritten digits (0-9) and is pre-packaged in Keras.
    Preprocess the data: We'll normalize the data and reshape it for the CNN.
    Define the CNN model: Using Keras to define a Convolutional Neural Network.
    Compile and train the model: We'll compile the model with an optimizer and loss function, and then train it.
    Evaluate the model: Finally, we'll test the model on the test set.

Step 1: Install TensorFlow (if not installed)

You can install TensorFlow using the following pip command:

pip install tensorflow

Step 2: Import Required Libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

Step 3: Load and Preprocess the MNIST Dataset

Keras provides a simple interface to load the MNIST dataset. We'll load the dataset, normalize it, and reshape it to fit the requirements of a CNN.

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize the images to be between 0 and 1 by dividing by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the images to include a channel dimension (since MNIST is grayscale, we set channels=1)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

Step 4: Define the CNN Model

Now, we define the Convolutional Neural Network (CNN) using Keras. The architecture includes convolutional layers, max-pooling layers, dropout for regularization, and a dense fully-connected layer at the end.

model = models.Sequential()

# First Convolutional Layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the 3D feature maps to 1D
model.add(layers.Flatten())

# Fully connected layer
model.add(layers.Dense(64, activation='relu'))

# Output layer with softmax activation for multi-class classification
model.add(layers.Dense(10, activation='softmax'))

# Summarize the model
model.summary()

Explanation of Layers:

    Conv2D layers: These are convolutional layers that help in feature extraction. We use 3x3 kernels, which are standard for CNNs.
    MaxPooling2D layers: Pooling layers reduce the spatial dimensions of the feature maps, which helps to reduce the number of parameters and computation.
    Flatten layer: This layer flattens the 3D feature map into a 1D vector to be passed into the fully connected layer.
    Dense layer: Fully connected layer with 64 units to capture high-level features from the flattened data.
    Softmax activation: This is used in the output layer to get the probability distribution for the 10 possible digit classes (0-9).

Step 5: Compile the Model

Before training the model, you need to compile it by specifying the optimizer, loss function, and metrics.

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    Adam optimizer: A popular optimizer that adapts the learning rate during training.
    Categorical Crossentropy: This loss function is used for multi-class classification problems (since MNIST has 10 classes).
    Accuracy metric: We want to track the accuracy during training.

Step 6: Train the Model

Now, we train the model on the training data:

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

    Epochs: The model will train for 5 epochs. You can increase this number for better accuracy.
    Batch Size: Number of samples per gradient update. We use a batch size of 64 here.
    Validation Data: We also specify validation data (test set) to monitor performance on unseen data during training.

Step 7: Evaluate the Model

Once the model has finished training, we can evaluate its performance on the test dataset.

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

Step 8: Visualize the Training Results

We can plot the training and validation accuracy over the epochs to visualize the model’s performance during training.

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

Step 9: Make Predictions

After training, you can use the model to make predictions on new data.

# Make predictions on the test set
predictions = model.predict(x_test)

# Get the predicted class for the first image
predicted_class = np.argmax(predictions[0])
print(f"Predicted class for the first test image: {predicted_class}")

Step 10: Visualize the Results

You can also visualize the first few images from the test set and compare the predicted and actual labels.

# Visualize the first 5 test images and their predicted labels
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"True label: {np.argmax(y_test[i])}, Predicted label: {predicted_class}")
    plt.show()

Final Result:

Once the model is trained and evaluated, you should be able to see a high accuracy (~98-99%) on the MNIST dataset if the model is trained for a sufficient number of epochs.
Full Code Example:

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess the MNIST Dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Step 2: Define the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 3: Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Step 5: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Step 6: Visualize the Training History
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 7: Make Predictions
predictions = model.predict(x_test)
predicted_class = np.argmax(predictions[0])

# Visualize Results
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"True label: {np.argmax(y_test[i])}, Predicted label: {predicted_class}")
    plt.show()

Conclusion:

This code builds a Convolutional Neural Network (CNN) in Keras to classify handwritten digits from the MNIST dataset. We used basic CNN layers, including convolutional layers, max-pooling layers, and dense layers, and trained the model to achieve high accuracy. This is a fundamental example of using deep learning for image classification.
