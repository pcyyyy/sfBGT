"""
    Training entry for the paper-aligned sfBGT implementation.
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from data.data import LoadData
from nets.sfbgt_graph_classification.load_net import build_model
from train.train_sfbgt import evaluate_network
from train.train_sfbgt import train_epoch


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print("cuda available with GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")

    print("cuda not available")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def view_model_param(net_params):
    model = build_model(net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))

    print("Model/Total parameters:", "sfBGT", total_param)
    return total_param


def metric_to_string(metrics):
    ordered_keys = ["loss", "acc", "f1", "auc", "sen", "spe"]
    parts = []
    for key in ordered_keys:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            parts.append(f"{key}=nan")
        else:
            parts.append(f"{key}={value:.4f}")
    return ", ".join(parts)


def aggregate_fold_metrics(all_metrics):
    summary = {}
    keys = all_metrics[0].keys()

    for key in keys:
        values = np.array([metrics[key] for metrics in all_metrics], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }

    return summary


def select_model_score(metrics):
    auc = metrics.get("auc", float("nan"))
    if not np.isnan(auc):
        return auc
    return metrics["acc"]


def create_data_loaders(dataset, train_indices, val_indices, test_indices, batch_size):
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
    )
    return train_loader, val_loader, test_loader


def train_val_pipeline(model_name, dataset, params, net_params, dirs):
    start_time = time.time()
    labels = dataset.labels.numpy()

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params["device"]
    num_folds = params["num_folds"]

    net_params["total_param"] = view_model_param(net_params)

    config_payload = {
        "dataset": dataset.name,
        "model": model_name,
        "params": params,
        "net_params": {
            key: str(value) if isinstance(value, torch.device) else value
            for key, value in net_params.items()
        },
    }
    with open(write_config_file + ".txt", "w", encoding="utf-8") as file:
        file.write(json.dumps(config_payload, indent=2))

    print("Samples:", len(dataset))
    print("Classes:", dataset.num_classes)
    print("Nodes per graph:", dataset.num_nodes)
    print("Node feature dim:", dataset.node_feat_dim)

    fold_metrics = []
    stratified_kfold = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=params["seed"],
    )

    for fold_index, (train_val_indices, test_indices) in enumerate(
        stratified_kfold.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        fold_seed = params["seed"] + fold_index
        set_seed(fold_seed)

        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=params["val_ratio"],
            random_state=fold_seed,
            stratify=labels[train_val_indices],
        )

        train_loader, val_loader, test_loader = create_data_loaders(
            dataset=dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            batch_size=params["batch_size"],
        )

        model = build_model(net_params).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=params["init_lr"],
            weight_decay=params["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=params["lr_reduce_factor"],
            patience=params["lr_schedule_patience"],
            verbose=True,
        )

        writer = SummaryWriter(log_dir=os.path.join(root_log_dir, f"fold_{fold_index}"))
        best_state = copy.deepcopy(model.state_dict())
        best_val_metrics = None
        best_epoch = 0

        epoch_iterator = tqdm(
            range(params["epochs"]),
            desc=f"Fold {fold_index}/{num_folds}",
            leave=False,
        )
        fold_start_time = time.time()

        for epoch in epoch_iterator:
            train_metrics = train_epoch(model, optimizer, device, train_loader)
            val_metrics = evaluate_network(model, device, val_loader)

            writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            writer.add_scalar("train/acc", train_metrics["acc"], epoch)
            writer.add_scalar("train/f1", train_metrics["f1"], epoch)
            writer.add_scalar("train/auc", train_metrics["auc"], epoch)
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/acc", val_metrics["acc"], epoch)
            writer.add_scalar("val/f1", val_metrics["f1"], epoch)
            writer.add_scalar("val/auc", val_metrics["auc"], epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

            epoch_iterator.set_postfix(
                train_acc=f"{train_metrics['acc']:.3f}",
                val_acc=f"{val_metrics['acc']:.3f}",
                val_auc="nan" if np.isnan(val_metrics["auc"]) else f"{val_metrics['auc']:.3f}",
            )

            if best_val_metrics is None or select_model_score(val_metrics) > select_model_score(best_val_metrics):
                best_val_metrics = val_metrics
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            scheduler.step(val_metrics["loss"])

            if optimizer.param_groups[0]["lr"] < params["min_lr"]:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours > params["max_time"]:
                print("-" * 89)
                print("Max_time for training elapsed {:.2f} hours, so stopping".format(params["max_time"]))
                break

        writer.close()

        model.load_state_dict(best_state)
        test_metrics = evaluate_network(model, device, test_loader)
        fold_metrics.append(test_metrics)

        checkpoint_path = os.path.join(root_ckpt_dir, f"fold_{fold_index}_best.pkl")
        torch.save(best_state, checkpoint_path)

        print(
            f"Fold {fold_index} finished in {time.time() - fold_start_time:.2f}s | "
            f"best_epoch={best_epoch} | "
            f"val: {metric_to_string(best_val_metrics)} | "
            f"test: {metric_to_string(test_metrics)}"
        )

    summary = aggregate_fold_metrics(fold_metrics)
    total_time = time.time() - start_time

    print("\nFINAL CROSS-VALIDATION RESULTS")
    for metric_name, values in summary.items():
        print(f"{metric_name.upper()}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print("TOTAL TIME TAKEN: {:.4f}s".format(total_time))

    with open(write_file_name + ".txt", "w", encoding="utf-8") as file:
        file.write("FINAL CROSS-VALIDATION RESULTS\n")
        for metric_name, values in summary.items():
            file.write(f"{metric_name.upper()}: {values['mean']:.4f} +/- {values['std']:.4f}\n")
        file.write(f"\nTotal Time Taken: {total_time / 3600:.4f} hrs\n")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the config JSON file.")
    parser.add_argument("--gpu_id", help="Override gpu id.")
    parser.add_argument("--model", help="Override model name.")
    parser.add_argument("--dataset", help="Override dataset name.")
    parser.add_argument("--out_dir", help="Override output directory.")
    parser.add_argument("--data_root", help="Override custom data root directory.")

    parser.add_argument("--seed", help="Override random seed.")
    parser.add_argument("--epochs", help="Override epochs.")
    parser.add_argument("--batch_size", help="Override batch size.")
    parser.add_argument("--init_lr", help="Override initial learning rate.")
    parser.add_argument("--lr_reduce_factor", help="Override LR reduce factor.")
    parser.add_argument("--lr_schedule_patience", help="Override LR scheduler patience.")
    parser.add_argument("--min_lr", help="Override minimum learning rate.")
    parser.add_argument("--weight_decay", help="Override weight decay.")
    parser.add_argument("--num_folds", help="Override number of CV folds.")
    parser.add_argument("--val_ratio", help="Override validation ratio inside each fold.")
    parser.add_argument("--max_time", help="Override max training time in hours.")

    parser.add_argument("--branch_hidden_dim", help="Override branch hidden dimension.")
    parser.add_argument("--branch_layers", help="Override number of branch GNN layers.")
    parser.add_argument("--edge_relation_dim", help="Override edge relation dimension.")
    parser.add_argument("--context_hidden_dim", help="Override context hidden dimension.")
    parser.add_argument("--context_layers", help="Override number of context layers.")
    parser.add_argument("--context_heads", help="Override context heads.")
    parser.add_argument("--interaction_hidden_dim", help="Override interaction hidden dimension.")
    parser.add_argument("--coupling_threshold", help="Override common-edge threshold.")
    parser.add_argument("--dropout", help="Override dropout.")
    parser.add_argument("--readout", help="Override readout type.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    with open(args.config, encoding="utf-8") as file:
        config = json.load(file)

    if args.gpu_id is not None:
        config["gpu"]["id"] = int(args.gpu_id)
        config["gpu"]["use"] = True

    device = gpu_setup(config["gpu"]["use"], config["gpu"]["id"])

    model_name = args.model if args.model is not None else config["model"]
    dataset_name = args.dataset if args.dataset is not None else config["dataset"]
    out_dir = args.out_dir if args.out_dir is not None else config["out_dir"]
    data_root = args.data_root if args.data_root is not None else config.get("data_root")

    dataset = LoadData(dataset_name, root_dir=data_root)

    params = config["params"]
    net_params = config["net_params"]

    scalar_param_overrides = {
        "seed": int,
        "epochs": int,
        "batch_size": int,
        "init_lr": float,
        "lr_reduce_factor": float,
        "lr_schedule_patience": int,
        "min_lr": float,
        "weight_decay": float,
        "num_folds": int,
        "val_ratio": float,
        "max_time": float,
    }
    for key, caster in scalar_param_overrides.items():
        value = getattr(args, key)
        if value is not None:
            params[key] = caster(value)

    model_param_overrides = {
        "branch_hidden_dim": int,
        "branch_layers": int,
        "edge_relation_dim": int,
        "context_hidden_dim": int,
        "context_layers": int,
        "context_heads": int,
        "interaction_hidden_dim": int,
        "coupling_threshold": float,
        "dropout": float,
        "readout": str,
    }
    for key, caster in model_param_overrides.items():
        value = getattr(args, key)
        if value is not None:
            net_params[key] = caster(value)

    net_params["device"] = device
    net_params["gpu_id"] = config["gpu"]["id"]
    net_params["node_feat_dim"] = dataset.node_feat_dim
    net_params["pair_feat_dim"] = dataset.pair_feat_dim
    net_params["num_classes"] = dataset.num_classes

    timestamp = time.strftime("%Hh%Mm%Ss_on_%b_%d_%Y")
    root_log_dir = os.path.join(out_dir, "logs", f"{model_name}_{dataset_name}_GPU{config['gpu']['id']}_{timestamp}")
    root_ckpt_dir = os.path.join(out_dir, "checkpoints", f"{model_name}_{dataset_name}_GPU{config['gpu']['id']}_{timestamp}")
    write_file_name = os.path.join(out_dir, "results", f"result_{model_name}_{dataset_name}_GPU{config['gpu']['id']}_{timestamp}")
    write_config_file = os.path.join(out_dir, "configs", f"config_{model_name}_{dataset_name}_GPU{config['gpu']['id']}_{timestamp}")

    os.makedirs(root_log_dir, exist_ok=True)
    os.makedirs(root_ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "configs"), exist_ok=True)

    train_val_pipeline(
        model_name=model_name,
        dataset=dataset,
        params=params,
        net_params=net_params,
        dirs=(root_log_dir, root_ckpt_dir, write_file_name, write_config_file),
    )


if __name__ == "__main__":
    main()
