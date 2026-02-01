from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class TinyLlamaLLM:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

        self.device = self.model.device

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )

        # 🔑 REMOVE PROMPT TOKENS
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return answer.strip()

