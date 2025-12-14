import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Printing the shape
print("training_images shape:", train_images.shape)
print("training_labels shape:", train_labels.shape)
print("test_images shape:", test_images.shape)
print("test_labels shape:", test_labels.shape)

# Displaying the first 9 images of dataset
fig = plt.figure(figsize=(10, 10))

nrows = 3
ncols = 3
for i in range(9):
    fig.add_subplot(nrows, ncols, i+1)
    plt.imshow(train_images[i])
    plt.title("Digit: {}".format(train_labels[i]))
    plt.axis(False)
plt.show()

# Converting images pixel values to 0 - 1
train_images = train_images / 255
test_images = test_images / 255

print("First Label before Conversion:")
print(train_labels[0])

# Converting labels to one-hot encoding vectors
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

print("First Label after Conversion:")
print(train_labels[0])


# Using Sequential() to build layer one after another
model = tf.keras.Sequential([

    # Flatten Layer that converts to 1D array
    tf.keras.layers.Flatten(),
    
    # Hidden Layer with 512 units and relu activation
    tf.keras.layers.Dense(512, activation=tf.nn.relu),

    # Output Layer with 10 units for 10 classes and softmax activation
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

history = model.fit(
    x = train_images,
    y = train_labels,
    epochs = 10
)

# Showing plot for loss
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.legend(['Loss'])
plt.show()

#  Showing plot for accuracy
plt.plot(history.history['accuracy'], color='orange')
plt.xlabel('Epochs')
plt.legend(['Accuracy'])
plt.show()

# Call evaluate to find the accuracy on test images
test_loss, test_accuracy = model.evaluate(
    x = test_images,
    y = test_labels
)

print("Test Loss: %.4f"%test_loss)
print("Test Accuracy: %.4f"%test_accuracy)

# ----- part 7 ------
predicted_probabilities = model.predict(test_images)
predicted_classes = tf.argmax(predicted_probabilities, axis=-1).numpy()

index = 11

# Showing image
plt.imshow(test_images[index])

# Printing Probabilities
print("Predicted Probabilities:",index)
print(predicted_probabilities[index])

# Printing Predicted Class
print("Predicted Class:", index)
print(predicted_classes[index])