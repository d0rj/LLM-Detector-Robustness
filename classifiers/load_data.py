import datasets
from pathlib import Path


def semeval(files_root_path: str) -> datasets.DatasetDict:
    ds = datasets.load_dataset(
        "json",
        data_files={
            "train": (Path(files_root_path) / "subtaskA_train_monolingual.jsonl").as_posix(),
            "dev": (Path(files_root_path) / "subtaskA_dev_monolingual.jsonl").as_posix(),
        },
    )
    ds = ds.remove_columns(["model", "source"])
    ds["test"] = datasets.load_dataset(
        "json",
        data_files=(Path(files_root_path) / "subtaskA_monolingual.jsonl").as_posix(),
    )["train"]
    return ds
