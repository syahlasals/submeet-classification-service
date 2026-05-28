from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App
    APP_NAME: str
    PORT: int

    # Model paths (wajib ada)
    CNN_MODEL_PATH: str
    SIGMAS_PATH: str
    VOCAB_PATH: str

    # Integrasi Laravel
    # LARAVEL_URL: str
    # LARAVEL_WEBHOOK_SECRET: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

settings = Settings()