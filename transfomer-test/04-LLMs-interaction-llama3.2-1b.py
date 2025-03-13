from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig
import torch
import os
import json
import time
import numpy as np


def check_tokenizer_files(model_path):
    """
    Check and validate tokenizer files
    """
    try:
        with open(os.path.join(model_path, "tokenizer_config.json"), "r") as f:
            tokenizer_config = json.load(f)
        print("Tokenizer config:", tokenizer_config)

        with open(os.path.join(model_path, "special_tokens_map.json"), "r") as f:
            special_tokens = json.load(f)
        print("Special tokens map:", special_tokens)

        return tokenizer_config, special_tokens
    except Exception as e:
        print(f"Error reading tokenizer files: {str(e)}")
        return None, None


def load_model(model_path):
    """
    Load a local LLaMA model and tokenizer with enhanced error handling
    """
    try:
        print(f"Checking model path: {model_path}")
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        tokenizer_config, special_tokens = check_tokenizer_files(model_path)

        print("Loading config...")
        config = AutoConfig.from_pretrained(model_path)
        print(f"Model config loaded: {config.__class__.__name__}")

        print("Attempting to load tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
                local_files_only=True,
                revision="main"
            )
        except Exception as e:
            print(f"First tokenizer loading attempt failed: {str(e)}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
                local_files_only=True
            )

        print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading model...")
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True
        )
        print(f"Model loaded: {model.__class__.__name__}")

        return model, tokenizer

    except Exception as e:
        print(f"\nError during model loading: {str(e)}")
        raise


def generate_text(model, tokenizer, prompt, max_length=100):
    """
    Generate text using the loaded model and measure response time.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        end_time = time.time()

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_time = end_time - start_time
        return generated_text, response_time
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        raise


if __name__ == "__main__":
    model_path = "/root/autodl-tmp/llama-1b-hf-32768-fpf"
    print(f"Attempting to load model from: {model_path}")
    model, tokenizer = load_model(model_path)

    prompt = "Write a short story about a robot learning to paint:"
    response_times = []

    for i in range(10):
        print(f"\nGenerating text, iteration {i + 1}...")
        _, response_time = generate_text(model, tokenizer, prompt)
        response_times.append(response_time)
        print(f"Response time: {response_time:.4f} seconds")

    avg_response_time = np.mean(response_times)
    print(f"\nAverage LLM response time over 10 runs: {avg_response_time:.4f} seconds")
