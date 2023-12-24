from src.PotatoDisease.config.configuration import ConfigurationManager
from src.PotatoDisease.components.data_ingestion import DataIngestion
from src.PotatoDisease.logging import logger


STAGE_NAME = 'Data Ingestion'


class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion()
        data_ingestinion = DataIngestion(config=data_ingestion_config)
        data_ingestinion.download_file()
        data_ingestinion.extract_zip_file()


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
