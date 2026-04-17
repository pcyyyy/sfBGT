"""
    Dataset loader for the sfBGT project.
"""


def LoadData(DATASET_NAME, root_dir=None):
    if DATASET_NAME in {"CUSTOM", "SFBGT_CUSTOM"}:
        from .sfbgt_dataset import SfBGTDataset

        return SfBGTDataset(DATASET_NAME, root_dir=root_dir)

    raise ValueError(f"Unsupported dataset: {DATASET_NAME}")
