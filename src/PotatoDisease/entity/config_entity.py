from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_dir: Path
    data_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass
class DataValidationConfig:
    root_dir: Path
    unzip_dir: Path
    status: str


@dataclass
class ModelTrainingConfig:
    root_dir: Path
    dataset_path: Path
    model_save: Path
    BATCH_SIZE: int
    IMAGE_SIZE: int
    CHANNELS: int
    EPOCHS: int
