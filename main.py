# Garbage Classification Model using EfficientNetV2B2

# Import all required libraries
import numpy as np  # Importing NumPy for numerical operations and array manipulations
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting graphs and visualizations
import seaborn as sns  # Importing Seaborn for statistical data visualization, built on top of Matplotlib
import tensorflow as tf  # Importing TensorFlow for building and training machine learning models
from tensorflow import keras  # Importing Keras, a high-level API for TensorFlow, to simplify model building
from tensorflow.keras import Layer  # Importing Layer class for creating custom layers in Keras
from tensorflow.keras.models import Sequential  # Importing Sequential model for building neural networks layer-by-layer
from tensorflow.keras.layers import Rescaling, GlobalAveragePooling2D
from tensorflow.keras import layers, optimizers, callbacks  # Importing various modules for layers, optimizers, and callbacks in Keras
from sklearn.utils.class_weight import compute_class_weight  # Importing function to compute class weights for imbalanced datasets
from tensorflow.keras.applications import EfficientNetV2B2  # Importing EfficientNetV2B2 model for transfer learning
from sklearn.metrics import confusion_matrix, classification_report  # Importing functions to evaluate model performance
import gradio as gr  # Importing Gradio for creating interactive web interfaces for machine learning models

# Additional imports needed for this implementation
import os  # For file and directory operations
import warnings  # To handle warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Dataset path
DATASET_PATH = r"C:\garbage classification aicte"

# Configuration
IMG_SIZE = 224  # EfficientNetV2B2 input size
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

print("Loading dataset...")

# Load and prepare the dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Get class names
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Data preprocessing and augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# Preprocess datasets
def preprocess_dataset(dataset, augment=False):
    # Rescale pixel values to [0, 1]
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    # Performance optimization
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Apply preprocessing
train_dataset = preprocess_dataset(train_dataset, augment=True)
validation_dataset = preprocess_dataset(validation_dataset, augment=False)

print("Building model...")

# Create the model
def create_model():
    # Load pre-trained EfficientNetV2B2
    base_model = EfficientNetV2B2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create and compile the model
model = create_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
print("\nModel Summary:")
model.summary()

# Define callbacks
callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_garbage_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\nStarting training...")

# Train the model
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=callbacks_list,
    verbose=1
)

print("\nFine-tuning the model...")

# Fine-tune the model by unfreezing some layers
base_model = model.layers[0]
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = len(base_model.layers) - 20

# Freeze all layers before fine_tune_at
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training (fine-tuning)
fine_tune_epochs = 10
total_epochs = EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset,
    callbacks=callbacks_list,
    verbose=1
)

print("\nTraining completed!")

# Plot training history
def plot_training_history(history, history_fine=None):
    plt.figure(figsize=(12, 4))
    
    # Combine histories if fine-tuning was done
    if history_fine:
        acc = history.history['accuracy'] + history_fine.history['accuracy']
        val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
        loss = history.history['loss'] + history_fine.history['loss']
        val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    else:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

# Plot the training history
plot_training_history(history, history_fine)

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(validation_dataset, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Make predictions for confusion matrix
print("\nGenerating predictions...")
y_pred = []
y_true = []

for images, labels in validation_dataset:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save the final model
model.save('garbage_classification_model.h5')
print("\nModel saved as 'garbage_classification_model.h5'")

# Function to predict single image
def predict_image(image_path):
    """
    Predict the class of a single image
    """
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return predicted_class, confidence

# Example usage of prediction function
print("\nTo predict a single image, use:")
print("predicted_class, confidence = predict_image('path/to/your/image.jpg')")
print("print(f'Predicted: {predicted_class} with {confidence:.2%} confidence')")

# Create Gradio interface for easy testing
def gradio_predict(image):
    """
    Gradio interface function for image prediction
    """
    if image is None:
        return "Please upload an image"
    
    # Preprocess image
    img = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, 0)
    
    # Make prediction
    predictions = model.predict(img, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    # Create results string
    result = f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2%}"
    
    # Add all predictions
    result += "\n\nAll Predictions:\n"
    for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
        result += f"{class_name}: {prob:.2%}\n"
    
    return result

# Launch Gradio interface
print("\nLaunching Gradio interface...")
interface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Garbage Classification",
    description="Upload an image to classify the type of garbage"
)

# Uncomment the line below to launch the Gradio interface
# interface.launch()

print("\nScript completed successfully!")
print("Your model is trained and ready to use!")