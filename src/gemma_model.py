from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GemmaModel:
    """Load a local Gemma model and generate answers."""

    def __init__(self, model_path, max_new_tokens, temperature):
        self.model_path = Path(model_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._check_model_path()
        self._load_tokenizer()
        self._load_model()

    def _check_model_path(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Gemma model folder was not found:\n{self.model_path}\n\n"
                "Open config.py and set GEMMA_MODEL to your real model folder."
            )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

    def _load_model(self):
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("cpu")

        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(self, prompt):
        """Generate one answer from one prompt."""
        start_time = time.perf_counter()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        )

        inputs = self._move_inputs_to_device(inputs)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        answer = self._decode_answer(output, inputs["input_ids"].shape[1])
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return answer, elapsed_ms

    def _move_inputs_to_device(self, inputs):
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _decode_answer(self, output, input_length):
        generated_tokens = output[0][input_length:]
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
