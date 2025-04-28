import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import time
from datetime import datetime

# Import from our modules
from data_preprocess import preprocess_dataset
from data_loader import get_data_loaders

# Set up argument parser
parser = argparse.ArgumentParser(description='Train multimodal emotion classification models')
parser.add_argument('--csv_path', type=str, default='MAS/dataset_clean.csv', help='Path to the dataset CSV')
parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'rf', 'both'], help='Model type to train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.006, help='Learning rate')
parser.add_argument('--hidden_size', type=int, default=256, help='Base hidden layer size for MLP')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
parser.add_argument('--early_stop', type=int, default=20, help='Early stopping patience')
parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--model_depth', type=str, default='deep', choices=['shallow', 'medium', 'deep', 'very_deep'], 
                   help='Depth of the MLP model architecture')
parser.add_argument('--use_batch_norm', action='store_true', help='Use batch normalization in the model')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for regularization')


# Enhanced Neural Network model with configurable depth
class MultimodalMLP(nn.Module):
    def __init__(self, input_size, hidden_size, depth='deep', use_batch_norm=True, dropout_rate=0.3, num_classes=2):
        super(MultimodalMLP, self).__init__()
        
        self.depth = depth
        self.use_batch_norm = use_batch_norm
        layers = []
        
        # Input layer
        if use_batch_norm:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        else:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Add hidden layers based on depth parameter
        if depth == 'shallow':
            # Just one hidden layer (already added above)
            pass
        
        elif depth == 'medium':
            # Add one more hidden layer
            if use_batch_norm:
                layers.append(nn.Linear(hidden_size, hidden_size // 2))
                layers.append(nn.BatchNorm1d(hidden_size // 2))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size // 2))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
        
        elif depth == 'deep':
            # Add three more hidden layers with decreasing sizes
            hidden_sizes = [hidden_size, hidden_size // 2, hidden_size // 4]
            
            for i in range(1, len(hidden_sizes)):
                if use_batch_norm:
                    layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                    layers.append(nn.BatchNorm1d(hidden_sizes[i]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                else:
                    layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
        
        elif depth == 'very_deep':
            # Add five more hidden layers with decreasing sizes
            hidden_sizes = [hidden_size, hidden_size // 2, hidden_size // 2, hidden_size // 4, hidden_size // 4, hidden_size // 8]
            
            for i in range(1, len(hidden_sizes)):
                if use_batch_norm:
                    layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                    layers.append(nn.BatchNorm1d(hidden_sizes[i]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                else:
                    layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                    
        # Output layer
        if depth == 'shallow':
            layers.append(nn.Linear(hidden_size, num_classes))
        elif depth == 'medium':
            layers.append(nn.Linear(hidden_size // 2, num_classes))
        elif depth == 'deep':
            layers.append(nn.Linear(hidden_size // 4, num_classes))
        elif depth == 'very_deep':
            layers.append(nn.Linear(hidden_size // 8, num_classes))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)


# Random Forest model (if needed)
def train_random_forest(X_train, y_train, params=None):
    from sklearn.ensemble import RandomForestClassifier
    
    if params is None:
        # Default parameters
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
    else:
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    
    # Train the model
    print("Training Random Forest...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Random Forest training completed in {training_time:.2f} seconds")
    
    return clf


# Training function for MLP
def train_mlp(model, train_loader, val_loader, args):
    # Set device (GPU if available, else CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")
        print("Check NVIDIA drivers and PyTorch CUDA installation.")
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # For tracking training progress
    train_losses = []
    val_losses = []
    val_f1_scores = []
    best_val_f1 = 0
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    patience_counter = 0
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate loss
                val_loss += loss.item() * inputs.size(0)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Store predictions and targets for metrics calculation
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate validation metrics
        val_accuracy = accuracy_score(all_targets, all_preds)
        val_precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
        val_recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
        val_f1_scores.append(val_f1)
        
        # Update learning rate scheduler
        scheduler.step(val_f1)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, "
              f"Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with F1: {val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    # Return training history and best validation F1 score
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1_scores': val_f1_scores,
        'best_val_f1': best_val_f1,
        'training_time': training_time
    }
    
    return model, history


# Evaluate model
def evaluate_model(model, data_loader, device, model_type='mlp'):
    if model_type == 'mlp':
        # Neural network evaluation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # Store predictions and targets
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
    else:
        # Random Forest evaluation
        all_preds = []
        all_targets = []
        
        for inputs, targets in data_loader:
            # Convert to numpy for sklearn model
            inputs_np = inputs.numpy()
            targets_np = targets.numpy()
            
            # Get predictions
            preds = model.predict(inputs_np)
            
            # Store predictions and targets
            all_preds.extend(preds)
            all_targets.extend(targets_np)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(all_targets, all_preds, normalize='true')
    
    # Print evaluation results
    print(f"\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return metrics


# Plot training history
def plot_training_history(history, output_dir):
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot validation F1 score
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1_scores'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, output_dir, model_type):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_type.upper()}')
    plt.colorbar()
    
    # Add text annotations
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, f'{conf_matrix[i, j]:.2f}',
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_type}.png'))
    plt.close()


def main():
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"{args.model_type}_{args.model_depth}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save command line arguments
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Preprocess data
    data_paths = preprocess_dataset(
        args.csv_path,
        output_dir='processed_data',
        test_size=args.test_size,
        random_state=args.seed
    )
    
    # Create data loaders
    train_loader, test_loader = get_data_loaders(
        data_paths['train_path'],
        data_paths['test_path'],
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get input size from the first batch
    for inputs, _ in train_loader:
        input_size = inputs.shape[1]
        break
    
    # Train models based on selected type
    if args.model_type in ['mlp', 'both']:
        # Initialize MLP model
        mlp_model = MultimodalMLP(
            input_size, 
            args.hidden_size, 
            depth=args.model_depth,
            use_batch_norm=args.use_batch_norm,
            dropout_rate=args.dropout_rate
        )
        
        # Print model architecture
        print("\nModel Architecture:")
        print(mlp_model)
        print(f"Total parameters: {sum(p.numel() for p in mlp_model.parameters())}")
        
        # Train MLP model
        print("\n" + "="*50)
        print(f"Training MLP Model (Depth: {args.model_depth})")
        print("="*50)
        
        mlp_model, mlp_history = train_mlp(mlp_model, train_loader, test_loader, args)
        
        # Evaluate MLP model
        mlp_metrics = evaluate_model(mlp_model, test_loader, device, model_type='mlp')
        
        # Save MLP results
        torch.save(mlp_model.state_dict(), os.path.join(args.output_dir, 'mlp_model.pth'))
        torch.save(mlp_model, os.path.join(args.output_dir, 'mlp_model_full.pth'))  # Save full model
        plot_training_history(mlp_history, args.output_dir)
        plot_confusion_matrix(mlp_metrics['confusion_matrix'], args.output_dir, 'mlp')
        
        # Save metrics to file
        with open(os.path.join(args.output_dir, 'mlp_metrics.txt'), 'w') as f:
            f.write(f"Model depth: {args.model_depth}\n")
            f.write(f"Hidden size: {args.hidden_size}\n")
            f.write(f"Batch normalization: {args.use_batch_norm}\n")
            f.write(f"Dropout rate: {args.dropout_rate}\n\n")
            f.write(f"Accuracy: {mlp_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {mlp_metrics['precision']:.4f}\n")
            f.write(f"Recall: {mlp_metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {mlp_metrics['f1']:.4f}\n")
            f.write(f"Training Time: {mlp_history['training_time']:.2f} seconds\n")
    
    if args.model_type in ['rf', 'both']:
        # For Random Forest, we need to convert the DataLoader data to numpy arrays
        print("\n" + "="*50)
        print("Training Random Forest Model")
        print("="*50)
        
        # Convert PyTorch DataLoader to numpy arrays for Random Forest
        X_train, y_train = [], []
        for inputs, targets in train_loader:
            X_train.append(inputs.numpy())
            y_train.append(targets.numpy())
        
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        
        # Train Random Forest
        rf_model = train_random_forest(X_train, y_train)
        
        # Evaluate Random Forest
        rf_metrics = evaluate_model(rf_model, test_loader, device, model_type='rf')
        
        # Save Random Forest model and results
        import joblib
        joblib.dump(rf_model, os.path.join(args.output_dir, 'rf_model.pkl'))
        plot_confusion_matrix(rf_metrics['confusion_matrix'], args.output_dir, 'rf')
        
        # Save metrics to file
        with open(os.path.join(args.output_dir, 'rf_metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {rf_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {rf_metrics['precision']:.4f}\n")
            f.write(f"Recall: {rf_metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {rf_metrics['f1']:.4f}\n")
    
    print(f"\nTraining and evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()