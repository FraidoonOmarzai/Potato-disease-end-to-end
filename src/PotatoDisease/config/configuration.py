from PotatoDisease.constants import *
from PotatoDisease.utils.common import read_yaml, create_directories
from PotatoDisease.entity.config_entity import (DataIngestionConfig,
                                                DataValidationConfig)


class ConfigurationManager:
    def __init__(self, config=CONFIG_PATH):
        self.config = read_yaml(config)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            data_url=config.data_url,
            unzip_dir=config.unzip_dir,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config

    def get_data_validation(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            unzip_dir=config.unzip_dir,
            status=config.status
        )
        return data_validation_config
