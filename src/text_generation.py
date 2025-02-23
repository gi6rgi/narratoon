import openai


class TextGenerator:
    def __init__(
        self,
        openai_api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 350,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=openai_api_key)

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return completion.choices[0].message.content
