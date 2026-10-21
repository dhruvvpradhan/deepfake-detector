import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from config import Config
from models.detector import DualBranchDetector
from dataset import DeepfakeVideoDataset

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)

    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            preds = model(sequences)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    auc = roc_auc_score(all_labels, all_preds)
    return auc

def run_training(train_paths, train_labels, val_paths, val_labels):
    device = torch.device(Config.DEVICE)
    model = DualBranchDetector(
        sequence_length=Config.SEQUENCE_LENGTH,
        hidden_dim=Config.LSTM_HIDDEN_DIM,
        fft_dim=Config.FFT_FEAT_DIM,
        dropout=Config.DROPOUT_RATE
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    train_ds = DeepfakeVideoDataset(train_paths, train_labels, Config.SEQUENCE_LENGTH, Config.FRAME_SIZE)
    val_ds = DeepfakeVideoDataset(val_paths, val_labels, Config.SEQUENCE_LENGTH, Config.FRAME_SIZE)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    best_auc = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_auc = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] | Loss: {loss:.4f} | Val ROC-AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_deepfake_detector.pth")
            print("--> Model checkpoint saved.")

if __name__ == "__main__":
    import os
    import glob
    import random

    print("Scanning data directories...")
    
    # 1. Grab all .mp4 files from the folders
    real_videos = glob.glob("data/real/*.mp4")
    fake_videos = glob.glob("data/fake/*.mp4")
    
    if not real_videos and not fake_videos:
        print("Error: No videos found. Please place .mp4 files in data/real and data/fake.")
        exit()

    # 2. Assign labels: 0.0 for Real, 1.0 for Fake
    all_paths = real_videos + fake_videos
    all_labels = [0.0] * len(real_videos) + [1.0] * len(fake_videos)

    # 3. Shuffle the dataset so the model doesn't just memorize the order
    dataset = list(zip(all_paths, all_labels))
    random.shuffle(dataset)
    all_paths, all_labels = zip(*dataset)

    # 4. Split into 80% Training and 20% Validation
    split_idx = int(0.8 * len(all_paths))
    
    train_paths = list(all_paths[:split_idx])
    train_labels = list(all_labels[:split_idx])
    
    val_paths = list(all_paths[split_idx:])
    val_labels = list(all_labels[split_idx:])

    print(f"Found {len(all_paths)} total videos.")
    print(f"Training on {len(train_paths)} videos | Validating on {len(val_paths)} videos")
    print("Starting training loop...")
    
    # 5. Execute the training function
    run_training(train_paths, train_labels, val_paths, val_labels)
    