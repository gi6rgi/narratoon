from src import prompts
from src.audio_generation import ElevenLabsAudioGenerator
from src.image_generation import FluxImageGenerator
from src.settings import AppSettings
from src.text_generation import TextGenerator
from src.video_generation import generate_video

VOICEOVER_PATH = "voiceover.mp3"
CHARACTER_IMAGE_PATH = "character.jpg"


def main():
    app_settings = AppSettings()

    image_generator = FluxImageGenerator(
        model_id=app_settings.flux_model_name,
        api_key=app_settings.huggingface_api_key,
    )
    text_generator = TextGenerator(
        openai_api_key=app_settings.openai_api_key,
        model=app_settings.openai_model,
    )
    audio_generator = ElevenLabsAudioGenerator(
        api_key=app_settings.elevenlabs_api_key,
    )

    # Voiceover generation.
    voiceover_text = text_generator.generate_text(
        prompt="Generate an interesting fact about cars for 4 year old boy",
        system_prompt=prompts.TEXT_GENERATOR_SYSTEM_PROMPT,
    )
    voiceover_audio_bytes = audio_generator.generate_audio(
        text=voiceover_text, voice_id=app_settings.elevenlabs_male_voice_id
    )
    with open(VOICEOVER_PATH, "wb") as file:
        file.write(voiceover_audio_bytes)

    # Character image generation.
    character_prompt = text_generator.generate_text(
        prompt="An auto mechanic",
        system_prompt=prompts.SYSTEM_PROMPT_FLUX_PROMPTER,
    )
    character_image_path = image_generator.generate_image(prompt=character_prompt)

    # Generate final video.
    result_path = generate_video(
        image_path=character_image_path, audio_path=VOICEOVER_PATH
    )
    print(f"Video is generated! Result path: {result_path}")


if __name__ == "__main__":
    main()
