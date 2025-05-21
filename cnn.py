import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
import cv2

from activations import ReLU, Softmax
from convolution import Conv2D
from maxpooling import MaxPool2D
from fullyconnected import Flatten, Dense
from configurations import NUM_CLASSES, IMG_SIZE, CLASSES

class CNN:
    def __init__(self):
        self.layers = [
            # Architecture as specified
            Conv2D(filters=32, kernel_size=3, in_channels=3, padding=1),
            ReLU(),
            MaxPool2D(),
            Flatten(),
            Dense(input_size=16*16*32, output_size=128),  # Adjusted for 32x32 input after pooling
            ReLU(),
            Dense(input_size=128, output_size=NUM_CLASSES),
            Softmax()
        ]
        self.epoch_count = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def forward(self, X):
        # Preprocessing steps
        # X = self._preprocess_images(X)
        for layer in self.layers:
            X = layer.forward(X)
        return X


    def backward(self, y_true, learning_rate):
        d_out = y_true
        for layer in reversed(self.layers):
            if isinstance(layer, (Dense, Conv2D)):
                d_out = layer.backward(d_out, learning_rate)
            else:
                d_out = layer.backward(d_out)
        return d_out

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32,
              learning_rate=0.001, checkpoint_dir=None, checkpoint_freq=1,
              resume_from=None):
        """
        Enhanced training with checkpointing and best model saving
        """
        # Initialize or resume training
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {self.epoch_count}")

        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        # Create checkpoint directory if needed
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        print("\ntraining started ............\n")
        for epoch in range(epochs):
            current_epoch = self.epoch_count + 1
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0
            correct = 0
            total = 0

            # Training loop
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                output = self.forward(X_batch)
                loss = -np.mean(np.log(output[np.arange(len(y_batch)), y_batch] + 1e-10))
                preds = np.argmax(output, axis=1)
                batch_correct = np.sum(preds == y_batch)

                epoch_loss += loss * len(X_batch)
                correct += batch_correct
                total += len(X_batch)

                y_onehot = np.eye(NUM_CLASSES)[y_batch]
                self.backward(y_onehot, learning_rate)

            # Validation
            val_output = self.forward(X_val)
            val_loss = -np.mean(np.log(val_output[np.arange(len(y_val)), y_val] + 1e-10))
            val_preds = np.argmax(val_output, axis=1)
            val_acc = np.mean(val_preds == y_val)

            # Record metrics
            train_loss = epoch_loss / total
            train_acc = correct / total
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            print("-----------------------------------------------------------------------------------")
            print(f"Epoch {current_epoch}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = current_epoch
                if checkpoint_dir:
                    best_model_path = os.path.join(checkpoint_dir, "best_model.ckpt")
                    self.save_checkpoint(best_model_path, current_epoch,
                                       train_loss_history, val_loss_history,
                                       train_acc_history, val_acc_history)
                    print(f"New best model saved with val_acc {val_acc:.4f}")

            # Save checkpoint if needed
            if checkpoint_dir and (current_epoch % checkpoint_freq == 0 or epoch == epochs-1):
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{current_epoch}.ckpt")
                self.save_checkpoint(checkpoint_path, current_epoch,
                                   train_loss_history, val_loss_history,
                                   train_acc_history, val_acc_history)
                print(f"Saved checkpoint to {checkpoint_path}")

            self.epoch_count += 1

        # Final report
        print(f"\nTraining completed. Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        
        # Plot results
        if checkpoint_dir:
            self.plot_training_history_from_dir(checkpoint_dir)
        else:
            self.plot_training_history(train_loss_history, val_loss_history,
                                 train_acc_history, val_acc_history)

    # [Keep all other methods (save_checkpoint, load_checkpoint, evaluate, etc.) the same]
    # Only the architecture and preprocessing steps have changed

    def save_checkpoint(self, filepath, epoch, train_loss_hist, val_loss_hist,
                       train_acc_hist, val_acc_hist):
        """Save complete training state"""
        model_data = {
            'epoch': epoch,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'train_loss_history': train_loss_hist,
            'val_loss_history': val_loss_hist,
            'train_acc_history': train_acc_hist,
            'val_acc_history': val_acc_hist,
            'layers': []
        }

        for layer in self.layers:
            if isinstance(layer, (Conv2D, Dense)):
                layer_data = {
                    'type': type(layer).__name__,
                    'weights': layer.weights,
                    'bias': layer.bias
                }
                if isinstance(layer, Conv2D):
                    layer_data.update({
                        'filters': layer.filters,
                        'kernel_size': layer.kernel_size,
                        'in_channels': layer.in_channels,
                        'stride': layer.stride,
                        'padding': layer.padding
                    })
                else:  # Dense
                    layer_data.update({
                        'input_size': layer.weights.shape[0],
                        'output_size': layer.weights.shape[1]
                    })
                model_data['layers'].append(layer_data)
            else:
                model_data['layers'].append({'type': type(layer).__name__})

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_checkpoint(self, filepath):
        """Load complete training state"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.epoch_count = model_data['epoch']
        self.best_val_acc = model_data.get('best_val_acc', 0.0)
        self.best_epoch = model_data.get('best_epoch', 0)
        new_layers = []

        for layer_data in model_data['layers']:
            if layer_data['type'] == 'Conv2D':
                layer = Conv2D(
                    filters=layer_data['filters'],
                    kernel_size=layer_data['kernel_size'],
                    in_channels=layer_data['in_channels'],
                    stride=layer_data['stride'],
                    padding=layer_data['padding']
                )
                layer.weights = layer_data['weights']
                layer.bias = layer_data['bias']
                new_layers.append(layer)
            elif layer_data['type'] == 'Dense':
                layer = Dense(
                    input_size=layer_data['input_size'],
                    output_size=layer_data['output_size']
                )
                layer.weights = layer_data['weights']
                layer.bias = layer_data['bias']
                new_layers.append(layer)
            elif layer_data['type'] == 'ReLU':
                new_layers.append(ReLU())
            elif layer_data['type'] == 'MaxPool2D':
                new_layers.append(MaxPool2D())
            elif layer_data['type'] == 'Flatten':
                new_layers.append(Flatten())
            elif layer_data['type'] == 'Softmax':
                new_layers.append(Softmax())

        self.layers = new_layers
        print(f"Loaded checkpoint from epoch {self.epoch_count}")

    # [Keep all other methods the same as in your original code]

    def plot_training_history(self, train_loss, val_loss, train_acc, val_acc):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()


    def plot_training_history_from_dir(self, checkpoint_dir):
        """Plot training history by reading checkpoints from a directory"""
        train_loss, val_loss, train_acc, val_acc = [], [], [], []

        # Get all checkpoint files and sort by epoch number
        checkpoint_files = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt') and not f.endswith('best_model.ckpt')],
            key=lambda x: int(x.split('_')[1].split('.')[0])  # Extract epoch number from "epoch_X.ckpt"
        )
        print(checkpoint_files)

        for file in checkpoint_files:
            file_path = os.path.join(checkpoint_dir, file)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    train_loss.append(data['train_loss_history'][-1])  # Get last value for this epoch
                    print(data['train_loss_history'][-1])
                    val_loss.append(data['val_loss_history'][-1])
                    train_acc.append(data['train_acc_history'][-1])
                    val_acc.append(data['val_acc_history'][-1])
                    print(f"Loaded checkpoint from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        if not train_loss:
            print("No valid checkpoint data found to plot!")
            return

        # Plotting
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def evaluate(self, X_test, y_test):
        output = self.forward(X_test)
        preds = np.argmax(output, axis=1)
        acc = np.mean(preds == y_test)
        print(f"Test Accuracy: {acc:.4f}")
        return acc

    # @staticmethod
    def load_model(self, filepath):
        # """Load a model from a file"""
        # with open(filepath, 'rb') as f:
        #     model_data = pickle.load(f)

        # model = CNN()
        # new_layers = []

        # for layer_data in model_data['layers']:
        #     if layer_data['type'] == 'Conv2D':
        #         layer = Conv2D(
        #             filters=layer_data['filters'],
        #             kernel_size=layer_data['kernel_size'],
        #             in_channels=layer_data['in_channels'],
        #             stride=layer_data['stride'],
        #             padding=layer_data['padding']
        #         )
        #         layer.weights = layer_data['weights']
        #         layer.bias = layer_data['bias']
        #         new_layers.append(layer)
        #     elif layer_data['type'] == 'Dense':
        #         layer = Dense(
        #             input_size=layer_data['input_size'],
        #             output_size=layer_data['output_size']
        #         )
        #         layer.weights = layer_data['weights']
        #         layer.bias = layer_data['bias']
        #         new_layers.append(layer)
        #     elif layer_data['type'] == 'ReLU':
        #         new_layers.append(ReLU())
        #     elif layer_data['type'] == 'MaxPool2D':
        #         new_layers.append(MaxPool2D())
        #     elif layer_data['type'] == 'Flatten':
        #         new_layers.append(Flatten())
        #     elif layer_data['type'] == 'Softmax':
        #         new_layers.append(Softmax())

        # model.layers = new_layers
        self.load_checkpoint(filepath)
        print(f"Model loaded from {filepath}")
        # return model


    def predict(self, image_path, show_image=True):
        """Predict class for a single image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image from {image_path}")
                return None

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            if img.shape != (IMG_SIZE, IMG_SIZE, 3):
                print(f"Error: Image shape {img.shape} is not (64, 64, 3)")
                return None

            X = np.array([img], dtype=np.float32) / 255.0
            output = self.forward(X)
            pred_class_idx = np.argmax(output, axis=1)[0]
            pred_class = CLASSES[pred_class_idx]
            confidence = output[0][pred_class_idx]

            if show_image:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(f"Predicted: {pred_class} ({confidence:.2f})")
                plt.axis('off')
                plt.show()

            return pred_class, confidence
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
