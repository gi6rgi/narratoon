import requests


class ElevenLabsAudioGenerator:
    def __init__(
        self,
        api_key: str,
        stability: float = 0.4,
        similarity_boost: float = 0.75,
    ):
        self.api_key = api_key
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.base_endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/"

    def generate_audio(self, text: str, voice_id: str) -> bytes:
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        payload = {
            "text": text,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
            },
        }

        url = self.base_endpoint + voice_id
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.content
