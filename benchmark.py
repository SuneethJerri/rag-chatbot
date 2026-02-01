import time
import torch
import psutil
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_PATH = "models/tinyllama"
PROMPT = "Explain retrieval augmented generation in simple terms."


def get_gpu_mem():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def benchmark(quantized: bool):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\n==============================")
    print("Quantized:", quantized)
    print("==============================")

    start_load = time.time()

    if quantized:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16 if not quantized else None,
        quantization_config=qconfig
    )

    load_time = (time.time() - start_load) * 1000

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    start_gen = time.time()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=64
        )
    gen_time = (time.time() - start_gen) * 1000

    peak_mem = get_gpu_mem()

    print(f"Load time: {load_time:.2f} ms")
    print(f"Generation time: {gen_time:.2f} ms")
    print(f"Peak GPU memory: {peak_mem:.2f} MB")

    return load_time, gen_time, peak_mem


if __name__ == "__main__":
    print("Running FP16 baseline...")
    baseline = benchmark(quantized=False)

    print("\nRunning 4-bit optimized...")
    optimized = benchmark(quantized=True)

    print("\n=== Summary ===")
    print("FP16     :", baseline)
    print("4-bit    :", optimized)

