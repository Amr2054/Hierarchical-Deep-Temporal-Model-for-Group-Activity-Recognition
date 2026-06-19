from pydantic_settings import BaseSettings, SettingsConfigDict
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_PATH = os.path.join(BASE_DIR, ".env")

class Settings(BaseSettings):
    KAGGLE_DATASET_ROOT: str
    KAGGLE_NUM_WORKERS : int
    KAGGLE_ANNOT_DIR : str
    KAGGLE_B3_PHASE_A_MODEL : str
    KAGGLE_B5_PHASE_A_MODEL : str
    LOCAL_DATASET_ROOT : str
    LOCAL_NUM_WORKERS : int
    LOCAL_B3_PHASE_A_MODEL : str
    LOCAL_B5_PHASE_A_MODEL : str

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        extra="ignore"
    )

def get_settings():
    return Settings()