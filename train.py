import os
import argparse
import numpy as np
import pandas as pd
import torch


from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import auc

from dataset import MyDataset
from model import CNNWaveletDualClassifier
from training_utils import (
    setup_seed, train, validate,
    analyze_user_predictions_verbose,
    train_user_level_decision_tree,
    evaluate_user_level_decision_tree,
    stratified_group_k_fold
)
from scipy import interpolate
from scipy import interpolate
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--k_folds', type=int, default=5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seed(args.seed)

    os.makedirs('./results', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    all_metrics = []
    all_fpr_tpr_pairs = []
    all_user_metrics = []
    all_user_fpr_tpr_pairs = []
    
    full_dataset = MyDataset(subset='train')
    all_labels = np.array([data['label'] for data in full_dataset])
    all_users = np.array([data['user_id'] for data in full_dataset])
 
    group_kfold = stratified_group_k_fold(np.zeros(len(all_labels)),all_labels,all_users,k=args.k_folds,seed = args.seed) 
    for fold, (train_idx, val_idx) in enumerate(group_kfold):
        print(f"\n===== Fold {fold+1} / {args.k_folds} =====")
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        labels = np.array([full_dataset[i]['label'] for i in train_idx])
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels)
        weights = np.array([class_weights[label] for label in labels])
        weights = torch.tensor(weights, dtype=torch.float32).cuda()
        sampler = WeightedRandomSampler(weights, len(weights))
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        
        train_eval_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False)

        model = CNNWaveletDualClassifier(num_classes=args.class_num).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100])

        train(model, train_loader, val_loader, optimizer, scheduler, args.epochs,fold=fold+1)

        model.load_state_dict(torch.load(f'./models/best_models_fold{fold+1}.pth'))
        print(f"Evaluating on Fold {fold+1}...")
        f1, acc, sens, spec, prec, recall, auc_score = validate(model, val_loader, fold=fold+1 ,phase=f'test_fold_{fold+1}')
        roc_csv = pd.read_csv(f'./results/roc_test_fold11{fold+1}.csv')
        all_fpr_tpr_pairs.append(roc_csv)
        train_user_csv = f'./results/user_prediction_stats_train_fold11{fold+1}.csv'
        val_user_csv = f'./results/user_prediction_stats_test_fold11{fold+1}.csv'
        analyze_user_predictions_verbose(model, train_eval_loader,fold=fold+1,save_path='./results/user_prediction_stats_train_fold11{}.csv')
        analyze_user_predictions_verbose(model,val_loader,fold=fold+1,save_path='./results/user_prediction_stats_test_fold11{}.csv')
        
        user_tree = train_user_level_decision_tree(
            train_csv_path=train_user_csv,
            max_depth=2,
            random_state=args.seed + fold
        )

        
        user_f1, user_acc, user_sens, user_spec, user_prec, user_recall, user_auc = \
            evaluate_user_level_decision_tree(
                user_tree,
                test_csv_path=val_user_csv,
                fold=fold+1,
                save_dir='./results'
            )

        all_user_metrics.append([
            user_f1, user_acc, user_sens, user_spec, user_prec, user_recall, user_auc
        ])

        user_roc_csv = pd.read_csv(f'./results/roc_user_fold{fold+1}.csv')
        all_user_fpr_tpr_pairs.append(user_roc_csv)
        all_metrics.append([f1, acc, sens, spec, prec, recall, auc_score])
    
    all_metrics = np.array(all_metrics)
    mean_metrics = np.mean(all_metrics, axis=0)
    print("\n=== Average Cross-Validation Performance ===")
    print(f"F1: {mean_metrics[0]:.4f}, Acc: {mean_metrics[1]:.4f}, Sens: {mean_metrics[2]:.4f}, "
      f"Spec: {mean_metrics[3]:.4f}, Precision: {mean_metrics[4]:.4f}, Recall: {mean_metrics[5]:.4f}, AUC: {mean_metrics[6]:.4f}")
    

    mean_fpr = np.linspace(0, 1, 100)
    mean_tprs = []
    
    for df in all_fpr_tpr_pairs:
        interp_func = interpolate.interp1d(df['FPR'], df['TPR'], bounds_error=False, fill_value=(0, 1))
        mean_tprs.append(interp_func(mean_fpr))
    
    mean_tpr = np.mean(mean_tprs, axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)
    
    pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv('./results/roc_mean_windows11.csv', index=False)
    print(f"Mean ROC AUC: {mean_auc:.4f}")
    
    all_user_metrics = np.array(all_user_metrics)
    mean_user_metrics = np.mean(all_user_metrics, axis=0)

    print("\n=== Average Cross-Validation Performance (User Level) ===")
    print(
        f"F1: {mean_user_metrics[0]:.4f}, Acc: {mean_user_metrics[1]:.4f}, "
        f"Sens: {mean_user_metrics[2]:.4f}, Spec: {mean_user_metrics[3]:.4f}, "
        f"Precision: {mean_user_metrics[4]:.4f}, Recall: {mean_user_metrics[5]:.4f}, "
        f"AUC: {mean_user_metrics[6]:.4f}"
    )

    mean_user_fpr = np.linspace(0, 1, 100)
    mean_user_tprs = []
    for df in all_user_fpr_tpr_pairs:
        df = df.sort_values('FPR').drop_duplicates('FPR')
        interp_func = interpolate.interp1d(
            df['FPR'], df['TPR'],
            bounds_error=False,
            fill_value=(0, 1)
        )
        mean_user_tprs.append(interp_func(mean_user_fpr))

    mean_user_tpr = np.mean(mean_user_tprs, axis=0)
    mean_user_auc = auc(mean_user_fpr, mean_user_tpr)

    pd.DataFrame({'FPR': mean_user_fpr, 'TPR': mean_user_tpr}).to_csv(
        './results/roc_mean_users11.csv', index=False
    )
    print(f"Mean User-level ROC AUC: {mean_user_auc:.4f}")
