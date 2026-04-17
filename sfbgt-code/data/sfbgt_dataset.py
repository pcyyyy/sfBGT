from pathlib import Path

import torch
from torch.utils.data import Dataset


class SfBGTDataset(Dataset):
    def __init__(self, name, root_dir=None):
        self.name = name
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parent / "sfbgt"

        node_feature_path = self.root_dir / "final_concat_with_labels.pt"
        pair_feature_path = self.root_dir / "combined_edge_input.pt"

        if not node_feature_path.exists():
            raise FileNotFoundError(f"Missing sfBGT node feature file: {node_feature_path}")
        if not pair_feature_path.exists():
            raise FileNotFoundError(f"Missing sfBGT pair feature file: {pair_feature_path}")

        raw_node_samples = torch.load(node_feature_path, map_location="cpu")
        self.node_features = []
        labels = []

        for node_features, label in raw_node_samples:
            self.node_features.append(node_features.float())
            labels.append(int(label.item()) if torch.is_tensor(label) else int(label))

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.pair_features = torch.load(pair_feature_path, map_location="cpu").float()

        if len(self.node_features) != self.pair_features.size(0):
            raise ValueError(
                "The number of node feature samples does not match the pair feature samples: "
                f"{len(self.node_features)} vs {self.pair_features.size(0)}."
            )

        self.num_samples = len(self.node_features)
        self.num_nodes = self.node_features[0].size(0)
        self.node_feat_dim = self.node_features[0].size(1)
        self.pair_feat_dim = self.pair_features.size(-1)
        self.num_classes = int(self.labels.max().item()) + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {
            "node_features": self.node_features[index],
            "pair_features": self.pair_features[index],
            "label": self.labels[index],
        }

    def collate(self, samples):
        return {
            "node_features": torch.stack([sample["node_features"] for sample in samples], dim=0),
            "pair_features": torch.stack([sample["pair_features"] for sample in samples], dim=0),
            "labels": torch.stack([sample["label"] for sample in samples], dim=0).long(),
        }
