import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

class LatentClassifierTrainer:
    """Trains classifier with proper class weighting for imbalanced data"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

    def train(self, train_tensors, train_labels, val_tensors, val_labels,
              num_epochs=50, batch_size=32, learning_rate=5e-4):
        
        num_positive = (train_labels == 1).sum().item()
        num_negative = (train_labels == 0).sum().item()
        total = num_positive + num_negative
        pos_weight = torch.tensor([num_negative / max(num_positive, 1)], device=self.device)

        print(f"\n{'_'*60}\nTRAINING SETUP\n{'_'*60}")
        print(f"Class balance: Positive: {num_positive} | Negative: {num_negative}")
        print(f"Epochs: {num_epochs} | Batch: {batch_size} | LR: {learning_rate}\n{'='*60}\n")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_val_auc = 0
        best_model_state = None
        patience_counter = 0

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []

            indices = torch.randperm(len(train_labels))
            for i in range(0, len(train_labels), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_traj = train_tensors[batch_idx].to(self.device)
                batch_labels = train_labels[batch_idx].to(self.device).float()

                logits = self.model(batch_traj)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_preds.append(logits.detach().cpu().numpy())
                train_targets.append(batch_labels.detach().cpu().numpy())

            train_preds = np.concatenate(train_preds)
            train_targets = np.concatenate(train_targets)
            train_probs = 1.0 / (1.0 + np.exp(-train_preds))
            train_acc = accuracy_score(train_targets, (train_probs > 0.5).astype(int))
            train_auc = roc_auc_score(train_targets, train_probs)
            train_loss /= max(1, len(range(0, len(train_labels), batch_size)))

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_tensors.to(self.device))
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_targets = val_labels.numpy()
                val_auc = roc_auc_score(val_targets, val_probs)
                val_acc = accuracy_score(val_targets, (val_probs > 0.5).astype(int))

            print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"\n→ Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(best_model_state)
                    break

            scheduler.step()
        return self.model

    def evaluate(self, test_tensors, test_labels):
        self.model.eval()
        with torch.no_grad():
            test_logits = self.model(test_tensors.to(self.device))
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            targets = test_labels.numpy()

        accuracy = accuracy_score(targets, (test_probs > 0.5).astype(int))
        auc = roc_auc_score(targets, test_probs)
        f1 = f1_score(targets, (test_probs > 0.5).astype(int))

        print(f"\n{'_'*60}\nTEST RESULTS\n{'_'*60}")
        print(f"Accuracy: {accuracy:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}\n{'='*60}\n")
        return {'accuracy': accuracy, 'auc': auc, 'f1': f1, 'probs': test_probs, 'targets': targets}
