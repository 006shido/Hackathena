import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
from pathlib import Path

# Add path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ml.models.face_swap_model import get_face_swap_model
from ml.data.dataset import FaceSwapDataset

def train_face_swap_model(model, dataloader, num_epochs=10, device='cpu'):
    """
    Train the face swap model

    Args:
        model: The face swap model to train
        dataloader: DataLoader with source-target pairs
        num_epochs: Number of training epochs
        device: Device to run training on ('cpu' or 'cuda')
    """

    print(f"Starting training on {device}")
    model.to(device)

    # Define loss functions
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        print(f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, data in enumerate(dataloader):
            source = data['source'].to(device)
            target = data['target'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass - input both source and target
            swapped_output = model(source, target)

            # Compute losses
            # 1. Reconstruction loss (L1 between output and target)
            loss_recon = criterion_l1(swapped_output, target)

            # Total loss (without identity preservation loss since it's not appropriate here)
            total_loss_batch = loss_recon

            # Backward pass
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss = {total_loss_batch.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}")

    return model

def main():
    """
    Main training function
    """

    parser = argparse.ArgumentParser(description='Train face swap model')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')

    args = parser.parse_args()

    print("Face Swap Model Training")
    print("=" * 30)

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    print("Creating face swap model...")
    model = get_face_swap_model('simple')
    print(f"Model created: {type(model).__name__}")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create dataset from actual path instead of dummy_path
    print(f"Loading dataset from: {args.data_dir}")
    dataset = FaceSwapDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Train model
    print("\nStarting training...")
    trained_model = train_face_swap_model(model, dataloader, num_epochs=args.num_epochs, device=device)

    # Save model
    save_path = "face_swap_model_checkpoint.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_type': 'simple',
    }, save_path)
    print(f"Model saved to {save_path}")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
