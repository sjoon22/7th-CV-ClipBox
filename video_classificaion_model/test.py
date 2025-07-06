import os
import torch
import yaml
from torch.utils.data import DataLoader
from utils.transforms import get_transforms
from datasets.accident_dataset import AccidentDataset
import tqdm

with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

model = torch.load(r"C:\sichan_programming\car_sc\checkpoints\best_model.pth")

test_dataset = AccidentDataset(r"C:\sichan_programming\sequence_1304", mode='val',
                                     transform=get_transforms('val', cfg['image_size']))

test_loader = DataLoader(test_dataset, shuffle=True,
                              num_workers=0, pin_memory=True)

def test(model, test_loader, device):
    model.test()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            videos = batch['video'].to(device)       # [B, T, C, H, W]
            labels = batch['label'].float().to(device)

            outputs = model(videos).squeeze(1)        # [B]
            preds = (torch.sigmoid(outputs) > 0.5).long()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    print(f"[label]: {all_preds:.4f}")


test(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')