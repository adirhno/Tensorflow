import tensorflow as tf
import matplotlib.pyplot as plt

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
