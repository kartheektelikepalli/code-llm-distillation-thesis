from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


class MLXBackend:

    def __init__(self, model_path, adapter_path=None):
        self.model, self.tokenizer = load(
            model_path,
            adapter_path=adapter_path
        )

    def generate(
        self,
        prompt,
        max_tokens,
        temperature=0.8,
    ):

        sampler = make_sampler(
            temp=temperature
        )

        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )