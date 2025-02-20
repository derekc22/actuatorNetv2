import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

class ActuatorNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(ActuatorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Softsign(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softsign(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softsign(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_concatenated_model(data_dir, output_dir, val_split, resume_training, n_epochs):
    """Train single model for all joints"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load concatenated dataset
    data = torch.load(data_dir / "concatenated_data.pt", weights_only=True)
    inputs, targets = data['inputs'], data['targets']
    
    # Verify dimensions
    assert inputs.shape[1] == 60, "Input features should have 60 dimensions (10 joints × 6 features)"
    assert targets.shape[1] == 10, "Targets should have 10 dimensions (one per joint)"

    # Create datasets
    dataset = TensorDataset(inputs, targets)
    val_size = int(len(dataset) * val_split)
    train_dataset, val_dataset = random_split(dataset, [len(dataset)-val_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512)

    # Model setup
    model = ActuatorNet(input_dim=60, output_dim=10, hidden_size=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Checkpoint setup
    checkpoint_path = output_dir / "concatenated_checkpoint.pth"
    start_epoch = 0
    if resume_training and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # Training metrics
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(start_epoch, n_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)

        # Progress update
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "f"Train Loss: {avg_train_loss:.4e} | "f"Val Loss: {avg_val_loss:.4e}")

    # Save final model
    torch.save(model.state_dict(), output_dir / "concatenated_model.pth")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.title('Concatenated Model Training Progress')
    plt.legend()
    plt.savefig(output_dir / "concatenated_loss_curve.png")
    plt.close()

if __name__ == "__main__":

    # Train concatenated model
    train_concatenated_model(
        data_dir='./collected_data/inverse_kinematics',
        output_dir='./trained_models/concatenated',
        val_split=0.2, 
        resume_training=False, 
        n_epochs=800
    )