import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
from utils import predict_and_print


dataset = tfds.load('mnist', as_supervised=True)

# Extract train and test sets
train_dataset, test_dataset = dataset['train'], dataset['test']

model = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Preprocess function: normalize images to [0, 1] range and reshape to include channel dimension
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  
    image = tf.reshape(image, (28, 28, 1))    
    return image, label

train_dataset = train_dataset.map(preprocess_image).batch(32)
test_dataset = test_dataset.map(preprocess_image).batch(32)


# Evaluate the model on the test dataset before training
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy before training: {test_acc}")

# Train the model
model.fit(train_dataset, epochs=1, validation_data=test_dataset)

# Evaluate the model on the test dataset after training
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy after training: {test_acc}")

predict_and_print(model, test_dataset)
