# train_mvp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from structure import CGRPAOptimizer


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for MNIST classification."""
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    """Main training function for MNIST with CGRPAOptimizer."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.005
    num_epochs = 2
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize model
    model = SimpleMLP().to(device)
    print(f"Model architecture:\n{model}")
    
    # Initialize optimizer
    optimizer = CGRPAOptimizer(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        tau_max=10.0,
        weight_decay=0.0
    )
    
    actual_lr = optimizer.param_groups[0].get('lr', None)
    print(f"Optimizer: CGRPAOptimizer (requested lr={learning_rate}, actual lr={actual_lr})")

    # Enforce requested learning rate if discrepancy exists
    if actual_lr is None or abs(actual_lr - learning_rate) > 1e-12:
        print(f"WARNING: optimizer lr ({actual_lr}) differs from requested ({learning_rate}). Enforcing requested lr.")
        for g in optimizer.param_groups:
            g['lr'] = learning_rate
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Define closure for second-order optimization
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Retain graph for Hutchinson curvature estimation
                loss.backward(retain_graph=True)
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                return loss
            
            # Optimizer step with closure
            try:
                loss = optimizer.step(closure)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print("WARNING: Loss is NaN! Stopping training.")
                    return
                    
                running_loss += loss.item()
                
            except Exception as e:
                print(f"Error during optimization step: {e}")
                return
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {avg_loss:.4f}")
                running_loss = 0.0
        
        print(f"Epoch [{epoch+1}/{num_epochs}] completed.\n")
    
    print("Training completed successfully!")
    
    # Evaluate training accuracy
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Training Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    train()
