import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from models.video_classifier import VideoClassifier
from datasets.accident_dataset import AccidentDataset
from utils.transforms import get_transforms
from utils.logger import Logger
import torch.optim as optim # 옵티마이저 필요
from torch.optim import lr_scheduler # 스케줄러 필요


def evaluate(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Eval]"):
            videos = batch['video'].to(device)       # [B, T, C, H, W]
            labels = batch['label'].float().to(device)

            outputs = model(videos).squeeze(1)        # [B]
            preds = (torch.sigmoid(outputs) > 0.5).long()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"[Eval] Acc: {acc:.4f}, F1: {f1:.4f}")
    return f1


def train(model, train_loader, val_loader, criterion, optimizer, device, cfg):
    model.to(device)
    best_f1 = 0
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    for epoch in range(1, cfg['num_epochs'] + 1):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
            videos = batch['video'].to(device)
            labels = batch['label'].float().to(device)

            outputs = model(videos).squeeze(1)    # [B]
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        scheduler.step()  # 스케줄러 업데이트
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"[Train set] - [Epoch {epoch}] Loss: {total_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

        # 평가 + best 모델 저장
        if epoch % cfg['eval_interval'] == 0:
            val_f1 = evaluate(model, val_loader, device)
            if val_f1 > best_f1:
                best_f1 = val_f1
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.module.state_dict(), "checkpoints/best_model.pth")
                print(f"✅ Best model saved at epoch {epoch} with F1: {val_f1:.4f}")


def main():
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['log_dir'], exist_ok=True)
    sys.stdout = Logger(cfg['log_dir'], filename='train.log')

    print("Loading datasets...")
    train_dataset = AccidentDataset(cfg['data_root'], mode='train',
                                     transform=get_transforms('train', cfg['image_size']))
    val_dataset = AccidentDataset(cfg['data_root'], mode='val',
                                   transform=get_transforms('val', cfg['image_size']))

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=True)

    print("Initializing model...")
    model = VideoClassifier(
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPU 병렬 처리 (1개만 있어도 동작)
    if torch.cuda.device_count() > 1 and len(cfg['gpu_ids']) > 1:
        print(f"[INFO] Using {len(cfg['gpu_ids'])} GPUs: {cfg['gpu_ids']}")
        model = nn.DataParallel(model, device_ids=cfg['gpu_ids'])

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    print("📚 Starting training...")
    train(model, train_loader, val_loader, criterion, optimizer, device, cfg)


if __name__ == "__main__":
    main()