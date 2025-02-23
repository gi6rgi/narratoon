from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    flux_model_name: str = "black-forest-labs/FLUX.1-dev"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str
    huggingface_api_key: str
    elevenlabs_api_key: str
    elevenlabs_male_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    elevenlabs_female_voice_id: str = "2OEeJcYw2f3bWMzzjVMU"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
