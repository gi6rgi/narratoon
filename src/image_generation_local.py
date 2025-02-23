import torch
from diffusers import FluxPipeline
from PIL import Image


class FluxImageGenerator:
    def __init__(
        self,
        model_id: str,
        huggingface_api_key: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        max_sequence_length: int = 512,
        offload: bool = True,
    ):
        self.model_id = model_id
        self.huggingface_api_key = huggingface_api_key
        self.device = device
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_sequence_length = max_sequence_length
        self.torch_dtype = torch_dtype
        self.offload = offload
        self.pipeline = self._init_pipeline()

    def _init_pipeline(self) -> FluxPipeline:
        pipeline = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            use_auth_token=self.huggingface_api_key,
        )
        if self.offload:
            pipeline.enable_model_cpu_offload()
        return pipeline

    def generate_image(self, prompt: str, seed: int | None = None) -> Image.Image:
        generator = torch.Generator(self.device)
        if seed is not None:
            generator.manual_seed(seed)

        generation_args = {
            "prompt": prompt,
            "height": self.height,
            "width": self.width,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "max_sequence_length": self.max_sequence_length,
            "generator": generator,
        }

        result = self.pipeline(**generation_args)
        return result.images[0]
