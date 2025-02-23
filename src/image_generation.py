from gradio_client import Client


class FluxImageGenerator:
    def __init__(
        self,
        model_id: str,
        api_key: str,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 28,
    ) -> None:
        self.client = Client(src=model_id, hf_token=api_key)
        self.width = width
        self.height = height
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def generate_image(self, prompt: str, seed: int | None = None) -> str:
        result = self.client.predict(
            prompt=prompt,
            seed=seed if seed is not None else 0,
            randomize_seed=True,
            width=self.width,
            height=self.height,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            api_name="/infer",
        )
        return result[0]
