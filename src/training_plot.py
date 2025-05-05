import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

class TrainingPlot(Callback):
    def __init__(self, output_path="training_plot.png"):
        super().__init__()
        self.output_path = output_path
        self.history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    def on_epoch_end(self, epoch, logs=None):
        # Append metrics at the end of each epoch
        self.history["accuracy"].append(logs.get("accuracy"))
        self.history["val_accuracy"].append(logs.get("val_accuracy"))
        self.history["loss"].append(logs.get("loss"))
        self.history["val_loss"].append(logs.get("val_loss"))

        # Plot the metrics
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["accuracy"], label="Training Accuracy")
        plt.plot(self.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Over Epochs")
        plt.grid(True, linestyle='--', linewidth=0.5)  # Add grid lines

        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.close()
        print(f"Training plot saved to {self.output_path}")
