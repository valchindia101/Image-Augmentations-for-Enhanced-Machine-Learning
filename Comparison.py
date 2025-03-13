import matplotlib.pyplot as plt
import tensorflow as tf

# Load training history of both models
history_no_aug = tf.keras.models.load_model("cnn_emotion_classifier.h5")
history_aug = tf.keras.models.load_model("cnn_emotion_classifier_augmented.h5")

# Extract training history
def get_history(model):
    return model.history.history['accuracy'], model.history.history['val_accuracy'], \
           model.history.history['loss'], model.history.history['val_loss']

# Get history for both models
train_acc_no_aug, val_acc_no_aug, train_loss_no_aug, val_loss_no_aug = get_history(history_no_aug)
train_acc_aug, val_acc_aug, train_loss_aug, val_loss_aug = get_history(history_aug)

# Plot Accuracy Comparison
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc_no_aug, label='Train No Aug')
plt.plot(val_acc_no_aug, label='Val No Aug')
plt.plot(train_acc_aug, label='Train Aug')
plt.plot(val_acc_aug, label='Val Aug')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")

# Plot Loss Comparison
plt.subplot(1, 2, 2)
plt.plot(train_loss_no_aug, label='Train No Aug')
plt.plot(val_loss_no_aug, label='Val No Aug')
plt.plot(train_loss_aug, label='Train Aug')
plt.plot(val_loss_aug, label='Val Aug')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")

plt.show()

# Print Final Validation Accuracies
print(f"Final Validation Accuracy (No Aug): {val_acc_no_aug[-1]:.4f}")
print(f"Final Validation Accuracy (With Aug): {val_acc_aug[-1]:.4f}")
