This Python script uses TensorFlow to apply five image augmentation techniques to a facial recognition dataset. It loads images from a specified directory, applies random transformations (horizontal flip, vertical flip, brightness adjustment, contrast adjustment, and saturation adjustment), and visualizes the original vs. augmented images side by side. The script leverages tf.keras.utils.image_dataset_from_directory for dataset loading and matplotlib for visualization.

Features:
Loads and preprocesses facial recognition images.
Applies five augmentation techniques to enhance training data.
Randomly selects and displays original vs. augmented images.

Usage:
Set data_directory = "facialrecognition" to point to your dataset.
Run the script to visualize augmentations.
