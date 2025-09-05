print(" Setting up ASL Hand Sign Recognition System")
print("=" * 60)


# Install required packages
!pip install kaggle tensorflow opencv-python matplotlib seaborn scikit-learn pillow


# Import libraries
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import files
import datetime


print(f" Setup complete! Using TensorFlow {tf.__version__}")



print("\n Step 1: Upload your kaggle.json file")
print("Click 'Choose Files' and select the kaggle.json you downloaded from Kaggle")


# Upload kaggle credentials
uploaded = files.upload()


# Setup Kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json


print(" Kaggle credentials set up!")


# Test Kaggle connection
print("\n Testing Kaggle connection...")
!kaggle --version


print("\n Downloading ASL dataset... (This takes 2-3 minutes)")


# Try multiple datasets until one works
datasets_to_try = [
    ("grassknoted/asl-alphabet", "asl-alphabet.zip"),
    ("debashishsau/aslamerican-sign-language-aplhabet-dataset", "aslamerican-sign-language-aplhabet-dataset.zip"),
    ("datamunge/sign-language-mnist", "sign-language-mnist.zip")
]


dataset_downloaded = False
dataset_info = None


for dataset_name, expected_zip in datasets_to_try:
    try:
        print(f" Trying {dataset_name}...")
        !kaggle datasets download -d {dataset_name}


        # Check if download succeeded
        zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
        if zip_files:
            dataset_file = zip_files[0]
            dataset_info = (dataset_name, dataset_file)
            print(f" Successfully downloaded: {dataset_file}")
            dataset_downloaded = True
            break
    except Exception as e:
        print(f" Failed: {str(e)[:100]}")
        continue


if not dataset_downloaded:
    print(" All Kaggle downloads failed. Using backup method...")
    # Create a minimal test dataset
    os.makedirs('asl_dataset/train/A', exist_ok=True)
    os.makedirs('asl_dataset/train/B', exist_ok=True)
    os.makedirs('asl_dataset/train/C', exist_ok=True)


    # Download placeholder images for testing
    for letter in ['A', 'B', 'C']:
        for i in range(10):
            !wget -q -O asl_dataset/train/{letter}/sample_{i}.jpg "https://via.placeholder.com/224x224/{letter}0{letter}0{letter}0/FFFFFF?text={letter}"


    print(" Created test dataset with A, B, C signs")
    dataset_info = ("test_dataset", "none")
    dataset_downloaded = True


if dataset_info[1] != "none":
    print(f"\n Extracting {dataset_info[1]}...")


    # Clean up any existing dataset
    !rm -rf asl_dataset


    try:
        with zipfile.ZipFile(dataset_info[1], 'r') as zip_ref:
            zip_ref.extractall('asl_dataset')
        print(" Dataset extracted successfully!")
    except Exception as e:
        print(f" Extraction error: {e}")


print("\n Exploring dataset structure...")


# Find the correct training path
def find_training_path():
    """Find where the actual training images are stored"""
    possible_paths = [
        'asl_dataset/asl_alphabet_train/asl_alphabet_train',
        'asl_dataset/asl_alphabet_train',
        'asl_dataset/train',
        'asl_dataset/Train',
        'asl_dataset/ASL_Dataset/Train',
        'asl_dataset'
    ]


    for path in possible_paths:
        if os.path.exists(path):
            # Check if this path has class folders with images
            try:
                contents = os.listdir(path)
                class_folders = [item for item in contents if os.path.isdir(os.path.join(path, item))]


                # Verify folders contain images
                valid_folders = 0
                for folder in class_folders[:5]:  # Check first 5 folders
                    folder_path = os.path.join(path, folder)
                    if os.path.exists(folder_path):
                        images = [f for f in os.listdir(folder_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                        if len(images) > 0:
                            valid_folders += 1


                if valid_folders >= 3:  # Need at least 3 classes
                    return path, class_folders
            except:
                continue


    return None, []


training_path, class_folders = find_training_path()


if training_path:
    print(f" Found training data at: {training_path}")
    print(f" Contains {len(class_folders)} classes: {sorted(class_folders)}")


    # Count images per class
    print("\n Images per class:")
    total_images = 0
    for class_name in sorted(class_folders)[:10]:  # Show first 10 classes
        class_path = os.path.join(training_path, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(f"   {class_name}: {len(images)} images")
            total_images += len(images)


    if len(class_folders) > 10:
        print(f"   ... and {len(class_folders) - 10} more classes")


    print(f"\n Total: {total_images} images across {len(class_folders)} classes")
else:
    print(" Could not find proper training data structure!")
    print("Please check your dataset or try a different one.")


if training_path:
    print("\n Setting up data preprocessing...")


    # Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2


    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )


    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )


    # Create data generators
    print(" Creating data generators...")


    train_generator = train_datagen.flow_from_directory(
        training_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )


    validation_generator = val_datagen.flow_from_directory(
        training_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )


    # Get class information
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)


    print(f" Data generators ready!")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {validation_generator.samples}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes: {class_names}")


    # Display sample images
    print("\n Sample images from your dataset:")
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()


    # Get a batch of images
    sample_batch_x, sample_batch_y = next(train_generator)


    for i in range(min(8, len(sample_batch_x))):
        # Display image
        axes[i].imshow(sample_batch_x[i])


        # Get class name
        class_idx = np.argmax(sample_batch_y[i])
        class_name = class_names[class_idx]


        axes[i].set_title(f'Sign: {class_name}')
        axes[i].axis('off')


    plt.tight_layout()
    plt.show()

if 'num_classes' in locals():
    print("\n Building AI Model...")


    # Use MobileNetV2 as base model (efficient and accurate)
    base_model = MobileNetV2(
        weights='imagenet',  # Pre-trained weights
        include_top=False,   # Exclude final classification layer
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )


    # Freeze base model weights initially
    base_model.trainable = False


    # Build complete model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])


    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )


    print(" Model built successfully!")
    print(f" Model has {model.count_params():,} parameters")


    # Show model architecture
    model.summary()







# ============================================================================
# CELL 6: FIXED MODEL TRAINING - REPLACE YOUR ORIGINAL CELL 6 WITH THIS
# ============================================================================


# Import additional modules needed for the fix
import glob
from PIL import Image


def create_robust_data_generators_fixed(training_path, class_names, img_size=224, batch_size=16, validation_split=0.2):
    """Create robust data generators with proper error handling"""
    print(" Creating robust data generators...")


    # Ensure all classes have enough images
    min_images_per_class = 10
    valid_classes = []


    for class_name in class_names:
        class_path = os.path.join(training_path, class_name)
        if os.path.exists(class_path):
            image_files = glob.glob(os.path.join(class_path, '*.[jJ][pP][gG]')) + \
                         glob.glob(os.path.join(class_path, '*.[pP][nN][gG]')) + \
                         glob.glob(os.path.join(class_path, '*.[jJ][pP][eE][gG]'))


            if len(image_files) >= min_images_per_class:
                valid_classes.append(class_name)
            else:
                print(f"    Skipping {class_name}: only {len(image_files)} images")


    print(f"   Valid classes: {len(valid_classes)}")


    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )


    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )


    # Create generators with error handling
    try:
        train_generator = train_datagen.flow_from_directory(
            training_path,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            classes=valid_classes,  # Only use valid classes
            interpolation='bilinear'
        )


        validation_generator = val_datagen.flow_from_directory(
            training_path,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            classes=valid_classes,  # Only use valid classes
            interpolation='bilinear'
        )


        print(f" Data generators created successfully!")
        print(f"   Training samples: {train_generator.samples}")
        print(f"   Validation samples: {validation_generator.samples}")


        return train_generator, validation_generator, True


    except Exception as e:
        print(f" Error creating generators: {e}")
        return None, None, False


if 'training_path' in locals() and 'class_folders' in locals():
    print("\n Starting FIXED Model Training...")
    print("This will take 20-40 minutes depending on dataset size")
    print("=" * 60)


    # Step 1: Create robust data generators
    train_generator, validation_generator, generators_ok = create_robust_data_generators_fixed(
        training_path, class_folders, img_size=224, batch_size=16  # Smaller batch size
    )


    if generators_ok:
        # Update class information
        class_names = list(train_generator.class_indices.keys())
        num_classes = len(class_names)
        print(f"   Final classes: {class_names}")


        # Rebuild model with correct number of classes
        print("\n Rebuilding model with correct classes...")


        # Use MobileNetV2 as base model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )


        # Freeze base model weights
        base_model.trainable = False


        # Build complete model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])


        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


        print(f" Model rebuilt with {model.count_params():,} parameters")


        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_asl_model_fixed.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=4,
                min_lr=0.00001,
                verbose=1
            )
        ]


        # Calculate steps
        steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
        validation_steps = max(1, validation_generator.samples // validation_generator.batch_size)


        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Validation steps: {validation_steps}")


        # Train the model
        try:
            history = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=25,  # Reduced epochs
                validation_data=validation_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )


            print("\n Training completed successfully!")


            # Plot training history
            print("\n Training Results:")


            fig, axes = plt.subplots(1, 2, figsize=(12, 4))


            # Accuracy
            axes[0].plot(history.history['accuracy'], label='Training Accuracy')
            axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[0].set_title('Model Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True)


            # Loss
            axes[1].plot(history.history['loss'], label='Training Loss')
            axes[1].plot(history.history['val_loss'], label='Validation Loss')
            axes[1].set_title('Model Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True)


            plt.tight_layout()
            plt.show()


            # Final results
            final_accuracy = max(history.history['val_accuracy'])
            print(f" Best Validation Accuracy: {final_accuracy*100:.2f}%")


        except Exception as e:
            print(f" Training failed: {e}")
            print("Try reducing batch_size further or check your dataset")


    else:
        print(" Could not create data generators. Please check your dataset structure.")
        print("\n Debugging your dataset:")
        for class_name in class_folders[:5]:  # Check first 5 classes
            class_path = os.path.join(training_path, class_name)
            if os.path.exists(class_path):
                files = os.listdir(class_path)
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"   {class_name}: {len(image_files)} images")
else:
    print(" No training path found. Please run the previous cells first.")











if 'model' in locals() and 'validation_generator' in locals():
    print("\n Evaluating Model Performance...")


    # Evaluate on validation set
    validation_generator.reset()
    eval_result = model.evaluate(validation_generator, verbose=1)


    print(f"\n Final Evaluation:")
    print(f"   Test Loss: {eval_result[0]:.4f}")
    print(f"   Test Accuracy: {eval_result[1]*100:.2f}%")


    # Get predictions for detailed analysis
    validation_generator.reset()
    predictions = model.predict(validation_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)


    # Calculate Top-3 accuracy manually (since it wasn't included in model compilation)
    top3_predictions = np.argsort(predictions, axis=1)[:, -3:]  # Get top 3 predictions
    true_classes = validation_generator.classes
   
    # Check if true class is in top 3 predictions
    top3_correct = 0
    for i, true_class in enumerate(true_classes):
        if true_class in top3_predictions[i]:
            top3_correct += 1
   
    top3_accuracy = top3_correct / len(true_classes)
    print(f"   Test Top-3 Accuracy: {top3_accuracy*100:.2f}%")


    # Get true labels
    true_classes = validation_generator.classes


    # Classification report
    print(f"\n Detailed Performance Report:")
    report = classification_report(true_classes, predicted_classes,
                                 target_names=class_names,
                                 output_dict=True)


    # Print classification report
    print(classification_report(true_classes, predicted_classes, target_names=class_names))


    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)


    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Model Performance per Class')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


    # Show prediction examples
    print("\n Prediction Examples:")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()


    # Get a batch for visualization
    validation_generator.reset()
    batch_x, batch_y = next(validation_generator)
    batch_predictions = model.predict(batch_x, verbose=0)


    for i in range(min(12, len(batch_x))):
        # Get prediction and true label
        pred_idx = np.argmax(batch_predictions[i])
        true_idx = np.argmax(batch_y[i])


        pred_class = class_names[pred_idx]
        true_class = class_names[true_idx]
        confidence = batch_predictions[i][pred_idx] * 100


        # Display image
        axes[i].imshow(batch_x[i])


        # Color code: green if correct, red if wrong
        color = 'green' if pred_idx == true_idx else 'red'
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\n{confidence:.1f}%',
                         color=color, fontsize=10)
        axes[i].axis('off')


    plt.suptitle('Prediction Examples (Green=Correct, Red=Wrong)', fontsize=16)
    plt.tight_layout()
    plt.show()



if 'model' in locals():
    def predict_sign(image_data):
        """Predict ASL sign from uploaded image"""
        from PIL import Image
        import io


        # Load image
        if isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data))
        else:
            img = Image.open(image_data)


        # Preprocess
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)


        # Predict
        predictions = model.predict(img_array, verbose=0)[0]


        # Get top predictions
        top_indices = np.argsort(predictions)[-5:][::-1]
        results = []


        for idx in top_indices:
            results.append({
                'sign': class_names[idx],
                'confidence': predictions[idx] * 100
            })


        return results, img


    print("\n Test Your Model!")
    print("Upload an image of a hand sign to see if your AI can recognize it!")
    print("Click 'Choose Files' below:")


    uploaded = files.upload()


    for filename in uploaded.keys():
        print(f"\n Analyzing {filename}...")


        # Make prediction
        results, processed_img = predict_sign(uploaded[filename])


        # Display image
        plt.figure(figsize=(8, 6))
        plt.imshow(processed_img)
        plt.title(f'Your Image: {filename}')
        plt.axis('off')
        plt.show()


        # Show predictions
        print(" AI Predictions:")
        print("-" * 50)


        for i, result in enumerate(results, 1):
            confidence_bar = "█" * int(result['confidence'] / 5)
            print(f"{i}. {result['sign']:<15} {result['confidence']:6.2f}% {confidence_bar}")


        # Give feedback
        best_pred = results[0]
        if best_pred['confidence'] > 80:
            print(f"\n I'm very confident this is: {best_pred['sign']}!")
        elif best_pred['confidence'] > 60:
            print(f"\n I think this might be: {best_pred['sign']}")
        else:
            print(f"\n I'm not very sure. Best guess: {best_pred['sign']}")


        print(f" Confidence: {best_pred['confidence']:.1f}%")




if 'model' in locals():
    print("\n Saving Your Trained Model...")


    # Create timestamp for unique naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    # Save model
    model_filename = f"asl_model_{timestamp}.h5"
    model.save(model_filename)


    # Save class names
    import json
    class_names_filename = f"class_names_{timestamp}.json"
    with open(class_names_filename, 'w') as f:
        json.dump(class_names, f)


    # Save training info
    info = {
        "model_name": "ASL Hand Sign Recognition",
        "timestamp": timestamp,
        "num_classes": len(class_names),
        "classes": class_names,
        "image_size": IMG_SIZE,
        "final_accuracy": float(max(history.history['val_accuracy'])),
        "total_parameters": int(model.count_params())
    }


    info_filename = f"model_info_{timestamp}.json"
    with open(info_filename, 'w') as f:
        json.dump(info, f, indent=2)


    print(f" Model saved as: {model_filename}")
    print(f" Class names saved as: {class_names_filename}")
    print(f" Model info saved as: {info_filename}")


    # Download files to your computer
    print("\n Downloading files to your computer...")
    files.download(model_filename)
    files.download(class_names_filename)
    files.download(info_filename)


    print(f"\n SUCCESS! Your ASL recognition model is complete!")
    print(f" Final accuracy: {info['final_accuracy']*100:.1f}%")
    print(f" Can recognize: {len(class_names)} different signs")
    print(f" Total parameters: {info['total_parameters']:,}")


print("\n" + "="*80)
print(" CONGRATULATIONS! You've built an AI that recognizes ASL signs!")
print("="*80)

