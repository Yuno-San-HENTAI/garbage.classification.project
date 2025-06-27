from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Training generator
train_generator = datagen.flow_from_directory(
    r"C:\garbage classification aicte",  # <<< Your dataset path
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation generator
val_generator = datagen.flow_from_directory(
    r"C:\garbage classification aicte",  
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

num_classes = train_generator.num_classes
print(f"Number of classes detected: {num_classes}")

from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras import layers, models, optimizers

# Load EfficientNetV2B2 without the top layer (we'll add our own)
base_model = EfficientNetV2B2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)

# Freeze base model layers (so we don't train ImageNet weights initially)
base_model.trainable = False

# Add custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')  # num_classes comes from earlier
])

# Compile the model
model.compile(
    optimizer=optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()  # Show the architecture

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Unfreeze all layers
base_model.trainable = True

# Re-compile with lower learning rate (important!)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Much lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train further
fine_tune_epochs = 5  # Start with 5, increase if you want
total_epochs = 10 + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]  # Continue training from where we stopped
)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

# Predict on validation set
val_generator.reset()
pred_probs = model.predict(val_generator)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_generator.classes

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))

