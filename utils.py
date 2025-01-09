import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np

# Show the prediction for the first image and print it
def predict_and_print(model, test_dataset):
    for image, label in test_dataset.take(1):
     predictions = model.predict(image)
    
    predicted_class = tf.argmax(predictions[0]).numpy()
    print(f'Predicted class: {predicted_class}')

# Print the first image (28 * 28)   
    plt.imshow(image[0, :, :, 0], cmap='gray')  
    plt.title(f'Predicted class: {predicted_class}, True label: {label[0]}')  # Title with predicted and true label
    plt.show()

def embedding_model(model, test_dataset):
   embedding_model = models.Model(inputs=model.inputs, outputs=model.layers[5].output)
   embeddings = embedding_model.predict(test_dataset)
   return embeddings



