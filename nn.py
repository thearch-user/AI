import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.ndimage import zoom, rotate
import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw
import io

# Enhanced Neural Network Class with improvements
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.1, regularization=0.001):
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Initialize weights and biases for multiple hidden layers
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            # He initialization for ReLU
            std_dev = np.sqrt(2.0 / self.layer_sizes[i])
            weight = np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]) * std_dev
            bias = np.zeros((self.layer_sizes[i+1], 1))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current_activation = X
        
        # Forward through all layers except the last
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], current_activation) + self.biases[i]
            self.z_values.append(z)
            current_activation = self.relu(z)
            self.activations.append(current_activation)
        
        # Output layer with softmax
        z_output = np.dot(self.weights[-1], current_activation) + self.biases[-1]
        self.z_values.append(z_output)
        output_activation = self.softmax(z_output)
        self.activations.append(output_activation)
        
        return output_activation
    
    def backward(self, X, Y):
        m = X.shape[1]
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        dz = self.activations[-1] - Y
        gradients_w[-1] = (1 / m) * np.dot(dz, self.activations[-2].T) + (self.regularization / m) * self.weights[-1]
        gradients_b[-1] = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        
        # Backpropagate through hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(self.weights[l+1].T, dz) * self.relu_derivative(self.z_values[l])
            gradients_w[l] = (1 / m) * np.dot(dz, self.activations[l].T) + (self.regularization / m) * self.weights[l]
            gradients_b[l] = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=0)
    
    def evaluate(self, X, Y):
        predictions = self.predict(X)
        true_labels = np.argmax(Y, axis=0)
        return accuracy_score(true_labels, predictions)

# Enhanced training function with validation
def train_model(nn, X_train, Y_train, X_val=None, Y_val=None, epochs=100, batch_size=128):
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Mini-batch training
        permutation = np.random.permutation(X_train.shape[1])
        X_shuffled = X_train[:, permutation]
        Y_shuffled = Y_train[:, permutation]
        
        for i in range(0, X_train.shape[1], batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[:, i:i+batch_size]
            
            # Forward and backward pass
            nn.forward(X_batch)
            nn.backward(X_batch, Y_batch)
        
        # Calculate training loss
        predictions = nn.forward(X_train)
        loss = -np.mean(Y_train * np.log(predictions + 1e-8))
        train_losses.append(loss)
        
        # Validation accuracy
        if X_val is not None and Y_val is not None:
            val_accuracy = nn.evaluate(X_val, Y_val)
            val_accuracies.append(val_accuracy)
        
        if epoch % 10 == 0:
            if X_val is not None:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    
    if val_accuracies:
        ax2.plot(val_accuracies)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation Accuracy")
    
    plt.tight_layout()
    plt.show()
    
    return train_losses, val_accuracies

# Enhanced drawing interface using tkinter
class DigitDrawer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw a Digit (0-9)")
        self.root.geometry("300x400")
        
        self.canvas = Canvas(self.root, width=280, height=280, bg='white', cursor="cross")
        self.canvas.pack(pady=10)
        
        self.label = Label(self.root, text="Draw a digit and click 'Predict'")
        self.label.pack()
        
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)
        
        self.predict_btn = Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.result_label = Label(self.root, text="", font=("Arial", 16))
        self.result_label.pack(pady=10)
        
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        
        self.nn = None
    
    def set_model(self, neural_network):
        self.nn = neural_network
    
    def paint(self, event):
        x, y = event.x, event.y
        r = 8  # Brush radius
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")
    
    def predict_digit(self):
        if self.nn is None:
            self.result_label.config(text="Model not loaded!")
            return
        
        # Convert to numpy array and preprocess
        small_img = self.image.resize((28, 28), Image.LANCZOS)
        digit_array = np.array(small_img) / 255.0
        
        # Center the digit (optional enhancement)
        digit_array = self.center_digit(digit_array)
        
        # Flatten and reshape for prediction
        digit_flat = digit_array.flatten().reshape(-1, 1)
        
        # Get prediction
        prediction = self.nn.predict(digit_flat)[0]
        confidence = np.max(self.nn.forward(digit_flat))
        
        self.result_label.config(text=f"Prediction: {prediction} (Confidence: {confidence:.2f})")
        
        # Show processed image
        self.show_processed_image(digit_array)
    
    def center_digit(self, digit_array):
        """Center the digit in the image"""
        # Find bounding box
        nonzero = np.nonzero(digit_array)
        if len(nonzero[0]) == 0:
            return digit_array
        
        min_y, max_y = np.min(nonzero[0]), np.max(nonzero[0])
        min_x, max_x = np.min(nonzero[1]), np.max(nonzero[1])
        
        # Calculate center of mass
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2
        
        # Shift to center
        shift_y = 14 - center_y
        shift_x = 14 - center_x
        
        # Apply shift
        shifted = np.roll(digit_array, shift_y, axis=0)
        shifted = np.roll(shifted, shift_x, axis=1)
        
        return shifted
    
    def show_processed_image(self, digit_array):
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title("Original Drawing")
        
        plt.subplot(1, 2, 2)
        plt.imshow(digit_array.reshape(28, 28), cmap='gray')
        plt.title("Processed (28x28)")
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        self.root.mainloop()

# Data augmentation function
def augment_data(X, Y, augment_factor=2):
    X_augmented = [X]
    Y_augmented = [Y]
    
    for _ in range(augment_factor - 1):
        X_new = X.copy()
        
        # Random transformations
        for i in range(X.shape[1]):
            img = X[:, i].reshape(28, 28)
            
            # Random rotation
            angle = np.random.uniform(-10, 10)
            img_rotated = rotate(img, angle, reshape=False, mode='constant', cval=0)
            
            # Random scaling
            scale = np.random.uniform(0.9, 1.1)
            if scale != 1.0:
                img_scaled = zoom(img, scale)
                # Center crop back to 28x28
                start = (img_scaled.shape[0] - 28) // 2
                img_scaled = img_scaled[start:start+28, start:start+28]
            else:
                img_scaled = img
            
            # Random shifting
            shift_x, shift_y = np.random.randint(-2, 3, 2)
            img_shifted = np.roll(img_scaled, shift_x, axis=1)
            img_shifted = np.roll(img_shifted, shift_y, axis=0)
            
            X_new[:, i] = img_shifted.flatten()
        
        X_augmented.append(X_new)
        Y_augmented.append(Y)
    
    return np.hstack(X_augmented), np.hstack(Y_augmented)

# Main Program
if __name__ == "__main__":
    print("Loading MNIST dataset...")
    
    # Load MNIST Dataset
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser='auto')
    data = mnist.data / 255.0
    labels = mnist.target.astype("int")
    
    # One-hot encode labels
    labels_one_hot = np.eye(10)[labels]
    
    # Split Data
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        data, labels_one_hot, test_size=0.15, random_state=42
    )
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=0.15, random_state=42
    )
    
    # Transpose for neural network
    X_train, X_val, X_test = X_train.T, X_val.T, X_test.T
    Y_train, Y_val, Y_test = Y_train.T, Y_val.T, Y_test.T
    
    print(f"Training set: {X_train.shape[1]} samples")
    print(f"Validation set: {X_val.shape[1]} samples")
    print(f"Test set: {X_test.shape[1]} samples")
    
    # Data augmentation
    print("Augmenting training data...")
    X_train_aug, Y_train_aug = augment_data(X_train, Y_train, augment_factor=2)
    print(f"After augmentation: {X_train_aug.shape[1]} training samples")
    
    # Initialize Enhanced Neural Network
    nn = NeuralNetwork(
        input_size=784, 
        hidden_sizes=[256, 128],  # Two hidden layers
        output_size=10,
        learning_rate=0.1,
        regularization=0.001
    )
    
    # Train Neural Network
    print("Training the neural network...")
    train_model(nn, X_train_aug, Y_train_aug, X_val, Y_val, epochs=50, batch_size=128)
    
    # Final evaluation
    test_accuracy = nn.evaluate(X_test, Y_test)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    
    # Show classification report
    test_predictions = nn.predict(X_test)
    test_true = np.argmax(Y_test, axis=0)
    print("\nClassification Report:")
    print(classification_report(test_true, test_predictions))
    
    # Launch drawing interface
    print("\nLaunching drawing interface...")
    drawer = DigitDrawer()
    drawer.set_model(nn)
    drawer.run()
