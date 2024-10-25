import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from model.multi_input_vision_mamba import Multi_input_MambaClassifier
from data_process.data_preprocess import create_folder
from data_process.data_loader import NEO_voxel_dataset

def compute_metrics(preds, targets):
     
    accuracy = (preds == targets).float().mean().item() * 100
    sensitivity = (preds * targets).sum().item() / (targets.sum().item() + 1e-5)
    specificity = ((1 - preds) * (1 - targets)).sum().item() / ((1 - targets).sum().item() + 1e-5)
    f1 = f1_score(targets.numpy(), preds.numpy(), zero_division=1)
    auc = roc_auc_score(targets.numpy(), preds.numpy())
    return accuracy, sensitivity, specificity, f1, auc


def train_loop(model, optimizer, criterion, train_loader, device, epoch):
  
    model.train()
    total_loss = 0
    preds, targets = [], []

    for data, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()

        output = model(data).squeeze()
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = (torch.sigmoid(output) > 0.5).int()
        preds.extend(pred.cpu().numpy())
        targets.extend(target.cpu().numpy())

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    metrics = compute_metrics(preds, targets)
    print(f'Average loss: {total_loss / len(train_loader):.4f}, Metrics: {metrics}')

    return total_loss / len(train_loader), metrics


def val_loop(model, criterion, val_loader, device, epoch, model_save_path=None, save_auc_threshold=0.75):
    
    model.eval()
    total_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data, target = data.to(device), target.to(device).float()
            output = model(data).squeeze()
            loss = criterion(output, target)
            total_loss += loss.item()

            pred = (torch.sigmoid(output) > 0.5).int()
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    metrics = compute_metrics(preds, targets)
    print(f'Validation Metrics: Average loss: {total_loss / len(val_loader):.4f}, Metrics: {metrics}')

    if metrics[-1] > save_auc_threshold and model_save_path is not None:
        model_save_filename = f"model_epoch{epoch}_auc_{metrics[-1]:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(model_save_path, model_save_filename))
        print(f"Model saved with AUC: {metrics[-1]:.2f} at epoch {epoch}")

    return total_loss / len(val_loader), metrics


def train(model, optimizer, criterion, train_loader, val_loader, epochs, device, model_save_path):
     
    create_folder(model_save_path)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6, verbose=True)
    best_auc = 0

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_avg_loss, train_metrics = train_loop(model, optimizer, criterion, train_loader, device, epoch)
        val_avg_loss, val_metrics = val_loop(model, criterion, val_loader, device, epoch, model_save_path)

        if val_metrics[-1] > best_auc:
            best_auc = val_metrics[-1]
        scheduler.step(val_avg_loss)

    print(f"Training complete. Best AUC: {best_auc:.2f}")


def main(args):
 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Using device:", device)
    train_set = NEO_voxel_dataset(args.data_path, args.train_txt, args.selected_indices)
    val_set = NEO_voxel_dataset(args.data_path, args.test_txt, args.selected_indices)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = Multi_input_MambaClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, optimizer, criterion, train_loader, val_loader, args.epochs, device, args.model_save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=218)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_path', type=str, default='voxel_and_baseline_h5py_dataset')
    parser.add_argument('--train_txt', type=str, default='dataset_files/training_index.txt')
    parser.add_argument('--test_txt', type=str, default='dataset_files/testing_index.txt')
    parser.add_argument('--model_save_path', type=str, default='check_points/mamba_classifier')
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
