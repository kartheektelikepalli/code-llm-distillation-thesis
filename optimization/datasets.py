from datasets import load_dataset


def load_dataset_split(dataset_name: str, split: str):
    return load_dataset(dataset_name)[split]