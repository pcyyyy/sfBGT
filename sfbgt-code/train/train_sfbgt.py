import numpy as np
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def move_batch_to_device(batch, device):
    return {
        "node_features": batch["node_features"].to(device),
        "pair_features": batch["pair_features"].to(device),
        "labels": batch["labels"].to(device),
    }


def _specificity_score(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = cm.sum()
    specificities = []

    for class_index in range(num_classes):
        tp = cm[class_index, class_index]
        fn = cm[class_index, :].sum() - tp
        fp = cm[:, class_index].sum() - tp
        tn = total - tp - fn - fp
        denominator = tn + fp
        specificities.append(tn / denominator if denominator > 0 else 0.0)

    return float(np.mean(specificities))


def compute_metrics(logits, labels):
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    y_pred = probabilities.argmax(axis=1)
    num_classes = probabilities.shape[1]

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "sen": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "spe": _specificity_score(y_true, y_pred, num_classes),
    }

    try:
        if num_classes == 2:
            metrics["auc"] = roc_auc_score(y_true, probabilities[:, 1])
        else:
            one_hot_targets = np.eye(num_classes)[y_true]
            metrics["auc"] = roc_auc_score(
                one_hot_targets,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
    except ValueError:
        metrics["auc"] = float("nan")

    return metrics


def train_epoch(model, optimizer, device, data_loader):
    model.train()

    total_loss = 0.0
    collected_logits = []
    collected_labels = []

    for iteration, batch in enumerate(data_loader):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        logits = model(batch["node_features"], batch["pair_features"])
        loss = model.loss(logits, batch["labels"])

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        collected_logits.append(logits.detach())
        collected_labels.append(batch["labels"].detach())

    epoch_logits = torch.cat(collected_logits, dim=0)
    epoch_labels = torch.cat(collected_labels, dim=0)
    metrics = compute_metrics(epoch_logits, epoch_labels)
    metrics["loss"] = total_loss / max(iteration + 1, 1)
    return metrics


def evaluate_network(model, device, data_loader):
    model.eval()

    total_loss = 0.0
    collected_logits = []
    collected_labels = []

    with torch.no_grad():
        for iteration, batch in enumerate(data_loader):
            batch = move_batch_to_device(batch, device)

            logits = model(batch["node_features"], batch["pair_features"])
            loss = model.loss(logits, batch["labels"])

            total_loss += loss.detach().item()
            collected_logits.append(logits.detach())
            collected_labels.append(batch["labels"].detach())

    epoch_logits = torch.cat(collected_logits, dim=0)
    epoch_labels = torch.cat(collected_labels, dim=0)
    metrics = compute_metrics(epoch_logits, epoch_labels)
    metrics["loss"] = total_loss / max(iteration + 1, 1)
    return metrics
