import csv
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CostSensitiveFocalLoss(nn.Module):
    def __init__(self, cost_matrix, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super().__init__()
        self.cost_matrix = cost_matrix
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        with torch.no_grad():
            # 创建 label smoothing 的 soft target
            smooth_labels = torch.full(size=(targets.size(0), num_classes),
                                       fill_value=self.smoothing / (num_classes - 1)).to(inputs.device)
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1 - pt) ** self.gamma
        ce_loss = -torch.sum(smooth_labels * log_probs, dim=1)

        cost_factor = self.cost_matrix[targets, 1 - targets]
        loss = focal_term * ce_loss * cost_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



COST_MATRIX = torch.tensor([[0, 1.0], [3.7, 0]], dtype=torch.float32).cuda()


def calculate_metrics(cm):
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
    specificity = TN / (TN + FP) if TN + FP != 0 else 0
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = sensitivity
    return sensitivity, specificity, precision, recall


def plot_confusion_matrix(cm, classes, filename):
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                     xticklabels=classes, yticklabels=classes, cbar=False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            percent = cm_percentage[i, j]
            color = "white" if value > cm.max() / 2 else "black"
            ax.text(j + 0.5, i + 0.4, f"{value}", ha='center', va='center', fontsize=14, fontweight='bold', color=color)
            ax.text(j + 0.5, i + 0.5, f"{percent:.1f}%", ha='center', va='center', fontsize=10, color=color)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def validate(model, dataloader, fold,phase='val',save_dir = './results'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data in dataloader
            x_img_s2 = data['s2'].cuda()
            x_img_s3 = data['s3'].cuda()
            x_feat = data['feature'].cuda()
            labels = data['label'].cuda()
            logits = model(x_img_s2,x_img_s3, x_feat)
            probs = torch.softmax(logits, dim=1)[:, 1]  
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) 


    cm = confusion_matrix(all_labels, all_preds)
    sensitivity, specificity, precision, recall = calculate_metrics(cm)
    f1 = f1_score(all_labels, all_preds, average='macro')
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    if phase.startswith('test_fold'):
        os.makedirs(save_dir, exist_ok=True)
        roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        roc_data.to_csv(f'{save_dir}/roc_test_fold11{fold}.csv', index=False)

    plot_confusion_matrix(cm, ['Class 0', 'Class 1'], f'./results/{phase}_confusion_matrix.png')
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend()
    plt.savefig(f'./results/{phase}_roc_curve.png')
    plt.close()

    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))
    print(f'{phase} AUC: {roc_auc:.4f}')
    return f1, np.mean(np.array(all_preds) == np.array(all_labels)), sensitivity, specificity, precision, recall, roc_auc

def train(model, train_loader, val_loader, optimizer, scheduler, epochs,fold):
    cls_criterion = CostSensitiveFocalLoss(cost_matrix=COST_MATRIX, gamma=2.0, reduction='mean',label_smoothing=0.1)
    best_auc = 0
    log = open('./results/log.txt', 'a')
    for epoch in range(epochs):
        iters = 0
        model.train()
        for data in train_loader:
            x_img_s2 = data['s2'].cuda()
            x_img_s3 = data['s3'].cuda()
            x_feat = data['feature'].cuda()
            labels = data['label'].cuda()
            logits = model(x_img_s2,x_img_s3, x_feat)
            loss = cls_criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iters += 1
            if iters % 100 == 0:
                print('epoches:{}, iter:{}, loss:{:.4f}'.format(epoch, iters, loss.item()))
        val_f1, val_acc, val_sens, val_spec, val_prec, val_recall, val_auc = validate(model, val_loader, fold, f'val_epoch_{epoch}')
        if val_auc > best_auc:
            torch.save(model.state_dict(), f'./models/best_models_fold{fold}.pth')
            best_auc = val_auc
        log.write(f"Epoch {epoch}:best auc: {best_auc:.4f} , Recall: {val_recall:.4f},F1: {val_f1:.4f}, Acc: {val_acc:.4f}, Sens: {val_sens:.4f}, Spec: {val_spec:.4f}, AUC: {val_auc:.4f}\n")
        print('epoch {}, best auc {:.4f}, val f1 {:.4f} , val AUC {:.4f}, Accuracy {:4f}'.format(epoch, best_auc,  val_f1,  val_auc , val_acc))
        scheduler.step()
    log.close()
def analyze_user_predictions_verbose(model, dataloader, fold, save_path='./results/user_prediction_stats_fold11{}.csv'):
    model.eval()
    user_window_preds = defaultdict(list)
    user_window_probs = defaultdict(list)

    user_window_week = defaultdict(list)
    user_labels = {}

    with torch.no_grad():
        for data in dataloader:
            x_s2 = data['s2'].cuda()
            x_s3 = data['s3'].cuda()
            x_feat = data['feature'].cuda()
            
            user_ids = data['user_id'].cpu().numpy()
            labels = data['label'].cpu().numpy()

            logits = model(x_s2, x_s3, x_feat)
            probs = torch.softmax(logits, dim=1)[:, 1] 
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            

            for uid, pred, prob, label in zip(user_ids, preds, probs.cpu().numpy(), labels):
                user_window_preds[uid].append(pred)
                user_window_probs[uid].append(prob)
               
                user_labels[uid] = label

    
    save_path = save_path.format(fold)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', newline='') as csvfile:
        fieldnames = ['user_id', 'true_label', 'num_windows', 'pred_0_count', 'pred_1_count',
                      'prob_mean', 'prob_max', 'prob_min', 'prob_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for uid in sorted(user_window_preds.keys()):
            preds = np.array(user_window_preds[uid])
            probs = np.array(user_window_probs[uid])
            
            label = user_labels[uid]

            unique, counts = np.unique(preds, return_counts=True)
            pred_dist = dict(zip(unique, counts))
            writer.writerow({
                'user_id': uid,
                'true_label': label,
                'num_windows': len(preds),
                'pred_0_count': pred_dist.get(0, 0),
                'pred_1_count': pred_dist.get(1, 0),
                'prob_mean': round(probs.mean(), 4),
                'prob_max': round(probs.max(), 4),
                'prob_min': round(probs.min(), 4),
                'prob_std': round(probs.std(), 4)
                })
                
def train_user_level_decision_tree(train_csv_path, max_depth=2, random_state=42):
    df = pd.read_csv(train_csv_path)

    df['pred_1_ratio'] = df['pred_1_count'] / df['num_windows']

    X_train = df[['pred_1_ratio', 'prob_mean']].values
    y_train = df['true_label'].values

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_user_level_decision_tree(clf, test_csv_path, fold, save_dir='./results'):
    df = pd.read_csv(test_csv_path)

    df['pred_1_ratio'] = df['pred_1_count'] / df['num_windows']

    X_test = df[['pred_1_ratio', 'prob_mean']].values
    y_true = df['true_label'].values

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sensitivity, specificity, precision, recall = calculate_metrics(cm)
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), 0.0

    os.makedirs(save_dir, exist_ok=True)

    df['user_pred'] = y_pred
    df['user_prob'] = y_prob
    df.to_csv(f'{save_dir}/user_level_predictions_fold{fold}.csv', index=False)

    pd.DataFrame({'FPR': fpr, 'TPR': tpr}).to_csv(
        f'{save_dir}/roc_user_fold{fold}.csv', index=False
    )

    plot_confusion_matrix(
        cm,
        ['User Class 0', 'User Class 1'],
        f'{save_dir}/user_fold{fold}_confusion_matrix.png'
    )

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend()
    plt.savefig(f'{save_dir}/user_fold{fold}_roc_curve.png')
    plt.close()

    print(f'\n[Fold {fold}] User-level results')
    print(classification_report(y_true, y_pred, target_names=['User Class 0', 'User Class 1']))
    print(f'[Fold {fold}] User-level AUC: {roc_auc:.4f}')

    return f1, acc, sensitivity, specificity, precision, recall, roc_auc   
    
def stratified_group_k_fold(X, y, groups, k, seed=None):
    label_per_group = {}
    for idx, g in enumerate(groups):
        label_per_group[g] = y[idx]

    unique_groups = np.unique(groups)
    group_labels = np.array([label_per_group[g] for g in unique_groups])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(unique_groups, group_labels):
        train_groups = unique_groups[train_idx]
        val_groups = unique_groups[val_idx]
        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        val_indices = [i for i, g in enumerate(groups) if g in val_groups]
        yield train_indices, val_indices
