# CNN-Based Emotion Classification with Image Augmentation

## Overview
This project trains a **Convolutional Neural Network (CNN)** to classify human emotions from facial images. It compares the performance of the model **with and without data augmentation** to analyze the effects on generalization.

## Dataset
The dataset used is **Human Face Emotions** from Kaggle:
- Automatically downloaded using `kagglehub.dataset_download("sanidhyak/human-face-emotions")`.
- Contains labeled images of faces with different emotions.

## Model Architecture
The CNN model consists of:
1. **Conv2D layers** (Feature extraction)
2. **MaxPooling** (Downsampling)
3. **Flatten layer** (Converts features to 1D)
4. **Dense layers** (Classification)
5. **Softmax activation** (Multiclass output)

## Data Augmentation Techniques Used
When augmentation is applied, the following transformations are introduced:
- **Random Flip** (Horizontal & Vertical)
- **Random Rotation**
- **Random Zoom**
- **Random Brightness**
- **Random Contrast**

## Training Process
The models were trained for **10 epochs** with the following steps:
1. Load dataset and split into **Training (80%)** and **Validation (20%)**.
2. Define CNN architecture.
3. Train the model **without augmentation**.
4. Train the model **with augmentation** (Real-time transformations).
5. Compare accuracy and loss between the two models.

## Running the Project
### 1. Install Dependencies
Ensure you have TensorFlow and Kaggle Hub installed:
```bash
pip install tensorflow kagglehub matplotlib
```

### 2. Train the Model Without Augmentation
Run the following Python script:
```python
python train_no_augmentation.py
```
This saves the model as `cnn_emotion_classifier.h5`.

### 3. Train the Model With Augmentation
Run the following script:
```python
python train_with_augmentation.py
```
This saves the model as `cnn_emotion_classifier_augmented.h5`.

### 4. Compare Model Performance
Run the comparison script:
```python
python compare_performance.py
```
This will generate plots comparing accuracy and loss.

## Expected Results
| Model | Final Validation Accuracy |
|--------|-------------------------|
| Without Augmentation | ~Lower (Overfits Faster) |
| With Augmentation | ~Higher (Better Generalization) |

## Future Improvements
- Increase dataset size.
- Use **Transfer Learning** (e.g., MobileNet, VGG16).
- Experiment with **additional augmentation** (Gaussian noise, elastic deformations).

## Author
Ayak Duk
Sala Aline 
Emmanuel Kiplimo 
Jamie 



