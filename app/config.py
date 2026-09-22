from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://ollama:11434"
    ollama_model: str = "gemma3:4b"

    agent_api_key: str = ""
    max_input_chars: int = 8000
    default_temperature: float = 0.2
    request_timeout: int = 300
    log_level: str = "INFO"

    num_predict: int = 1024
    num_ctx: int = 4096
    warmup_on_start: bool = True


settings = Settings()
