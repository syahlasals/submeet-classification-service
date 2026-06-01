from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App
    APP_NAME: str
    PORT: int

    # Model paths
    CNN_MODEL_PATH: str
    SIGMAS_PATH: str
    VOCAB_PATH: str

    # Hyperparameter Model
    ALPHA: int
    MAX_SEQ_LEN: int
    MAX_VOCAB: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra='ignore'
    )

settings = Settings()