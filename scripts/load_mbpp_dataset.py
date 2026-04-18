import warnings
warnings.filterwarnings("ignore")
"""
Load and explore MBPP dataset from HuggingFace.
Combines train, validation, and test splits.
"""

from datasets import load_dataset, concatenate_datasets


def main():
    """Load MBPP dataset, merge splits, and display information."""
    
    # Load dataset
    dataset = load_dataset("mbpp")

    # Merge splits
    full_dataset = concatenate_datasets([
        dataset["train"],
        dataset["validation"],
        dataset["test"]
    ])

    # Print summary
    print("=" * 60)
    print("MBPP Dataset Information")
    print("=" * 60)
    print(f"Total problems: {len(full_dataset)}")
    print(f"Column names: {full_dataset.column_names}")

    # Print first sample
    print("\n" + "=" * 60)
    print("First Sample")
    print("=" * 60)
    first_sample = full_dataset[0]
    print(f"task_id: {first_sample['task_id']}")
    print(f"text: {first_sample['text']}")
    print(f"test_list: {first_sample['test_list']}")
    print(f"code: {first_sample['code']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
