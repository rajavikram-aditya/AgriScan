import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    val_dir = os.path.join(base_dir, 'dataset', 'val')

    if not os.path.exists(train_dir):
        print(f"Error: Training directory {train_dir} not found.")
        return

    # Hyperparameters
    img_size = (224, 224)
    batch_size = 32
    epochs = 15

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    num_classes = len(train_generator.class_indices)
    print(f"Found {num_classes} classes.")

    # Save class indices
    class_indices_path = os.path.join(base_dir, 'class_indices.json')
    with open(class_indices_path, 'w') as f:
        json.dump(train_generator.class_indices, f)
    print(f"Saved class indices to {class_indices_path}")

    # Build model using MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base layers
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    # Save the model
    model_path = os.path.join(base_dir, 'agriscan_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plot_path = os.path.join(base_dir, 'training_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training plot to {plot_path}")

    # Generate Confusion Matrix
    print("Generating confusion matrix...")
    val_generator.reset()
    Y_pred = model.predict(val_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(train_generator.class_indices.keys()))
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.tight_layout()
    cm_path = os.path.join(base_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

if __name__ == '__main__':
    main()
