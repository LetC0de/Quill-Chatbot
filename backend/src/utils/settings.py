from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENROUTER_API_KEY: str
    MISTRAL_API_KEY: str
    MISTRAL_MODEL: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    DB_CONNECTION: str


    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    FRONTEND_URL: str = "http://localhost:5173"

    SECRET_KEY: str
    ALGORITHM: str 
    EXP_TIME: int 


settings = Settings()