from PotatoDisease.config.configuration import ConfigurationManager
from PotatoDisease.components.data_validation import DataValidation
from PotatoDisease.logging import logger


STAGE_NAME = 'Data Validation'


class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation()
        data_validation = DataValidation(data_validation_config)
        data_validation.validate_all_files_exist()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
