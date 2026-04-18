"""
CoralVisionNet IO - Full Evaluation Pipeline
==============================================
Calculates Accuracy, Precision, Recall, F1-Score, Confusion Matrix,
and exports all results to JSON and CSV formats.
"""

import os
import json
import csv
import logging
import torch
import torch.nn as nn
from torch.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def plot_reliability_diagram(confidences, accuracies, num_bins=10, save_path="reliability.png"):
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    bins = np.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_confs = []
    
    ece = 0.0
    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.bar(bins[:-1], bin_accs, width=1/num_bins, align='edge', color='blue', alpha=0.5, edgecolor='black', label='Outputs')
    plt.plot(bin_confs, bin_accs, marker='o', color='red', label='Calibration Curve')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.title(f'Reliability Diagram (ECE: {ece:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return ece

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics_csv(report_dict, save_path):
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for cls_name, metrics in report_dict.items():
            if cls_name not in ['accuracy', 'macro avg', 'weighted avg']:
                writer.writerow([cls_name, 
                                 f"{metrics['precision']:.4f}", 
                                 f"{metrics['recall']:.4f}", 
                                 f"{metrics['f1-score']:.4f}", 
                                 metrics['support']])
        
        writer.writerow([])
        writer.writerow(['accuracy', '', '', f"{report_dict.get('accuracy', 0.0):.4f}", ''])
        writer.writerow(['macro avg', 
                         f"{report_dict['macro avg']['precision']:.4f}", 
                         f"{report_dict['macro avg']['recall']:.4f}", 
                         f"{report_dict['macro avg']['f1-score']:.4f}", 
                         report_dict['macro avg']['support']])
        writer.writerow(['weighted avg', 
                         f"{report_dict['weighted avg']['precision']:.4f}", 
                         f"{report_dict['weighted avg']['recall']:.4f}", 
                         f"{report_dict['weighted avg']['f1-score']:.4f}", 
                         report_dict['weighted avg']['support']])

@torch.no_grad()
def run_evaluation(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_confs = []

    for rgb, lab, labels in loader:
        rgb = rgb.to(device)
        lab = lab.to(device)
        labels = labels.to(device)

        with autocast(device_type=device.type if device.type in ('cuda','cpu') else 'cpu', enabled=False):
            logits = model(rgb, lab)
            loss = criterion(logits, labels)

        running_loss += loss.item() * rgb.size(0)
        probs = F.softmax(logits, dim=1)
        confs, predicted = probs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return epoch_loss, epoch_acc, f1, all_preds, all_labels, all_confs

def evaluate_and_save(model, test_loader, device, class_names, save_dir):
    """Run complete evaluation and save metrics to files."""
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1, preds, labels, confs = run_evaluation(
        model, test_loader, criterion, device)

    logger.info(f"  Test Accuracy : {test_acc:.2f}%")
    logger.info(f"  Test F1-Score : {test_f1:.4f}")
    logger.info(f"  Test Loss     : {test_loss:.4f}")

    # Classification report
    report_str = classification_report(labels, preds, target_names=class_names, digits=4)
    report_dict = classification_report(labels, preds, target_names=class_names, output_dict=True)
    logger.info(f"\nClassification Report:\n{report_str}")

    # Save CSV
    csv_path = os.path.join(save_dir, "classification_report.csv")
    save_metrics_csv(report_dict, csv_path)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    logger.info(f"Confusion Matrix:\n{cm}")
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # ECE & Reliability Diagram
    rel_path = os.path.join(save_dir, "reliability_diagram.png")
    accs = (np.array(preds) == np.array(labels)).astype(float)
    ece = plot_reliability_diagram(confs, accs, save_path=rel_path)
    logger.info(f"  Expected Calibration Error (ECE) : {ece:.4f}")

    # Gating weights (if model supports it)
    gw = None
    if hasattr(model, 'fusion') and hasattr(model.fusion, 'get_weights'):
        gw = model.fusion.get_weights()
        logger.info(f"\nLearned Gating Weights:")
        for k, v in gw.items():
            logger.info(f"  {k} : {v:.4f}")

    results = {
        "accuracy": round(test_acc, 2),
        "f1_score": round(test_f1, 4),
        "loss": round(test_loss, 4),
        "ece": round(ece, 4),
        "gating_weights": gw,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    
    json_path = os.path.join(save_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    try:
        from utils.explainability import UnifiedXAI
        import cv2
        logger.info("Generating XAI visual overlays for a small batch...")
        xai_dir = os.path.join(save_dir, "xai_visualizations")
        os.makedirs(xai_dir, exist_ok=True)
        
        xai = UnifiedXAI(model)
        rgb_batch, lab_batch, labels_batch = next(iter(test_loader))
        rgb_batch = rgb_batch.to(device)
        lab_batch = lab_batch.to(device)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3, 1, 1)
        
        num_imgs = min(5, rgb_batch.size(0))
        for i in range(num_imgs):
            c_idx = labels_batch[i].item()
            
            img_unnorm = rgb_batch[i] * std + mean
            img_bgr = img_unnorm.permute(1, 2, 0).cpu().numpy() * 255.0
            img_bgr = img_bgr.clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            
            overlay, maps = xai.explain(rgb_batch[i:i+1], lab_batch[i:i+1], c_idx, original_image=img_bgr)
            
            if overlay is not None:
                c_name = class_names[c_idx] if c_idx < len(class_names) else str(c_idx)
                out_path = os.path.join(xai_dir, f"test_img_{i}_class_{c_name}.jpg")
                cv2.imwrite(out_path, overlay)
                
        xai.release()
        logger.info(f"XAI visualizations saved to {xai_dir}")
    except Exception as e:
        logger.error(f"Failed to generate XAI visualizations: {e}")

    logger.info(f"Evaluation artifacts saved to: {save_dir}")
    return results
