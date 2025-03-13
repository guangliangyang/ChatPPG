from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# ✅ Set model path for Qwen
model_path = "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-1.5B"

# ✅ Load Qwen Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# ✅ Load Qwen Model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use float16 to reduce memory usage
    device_map="auto"  # Auto-assign GPU
)

print("✅ Qwen model loaded successfully!\n")

# ✅ Input text
input_text = "告诉我几个可以投资的股票，什么时候买卖。"

# ✅ Run 10 inference loops and measure execution time
total_time = 0.0
num_iterations = 10

for i in range(num_iterations):
    print(f"🔄 Running LLM inference {i + 1}...\n")

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # Measure inference time
    start_time = time.time()

    # Perform inference (text generation)
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_length=200,  # Set max generation length
            temperature=0.7,  # Sampling temperature
            top_k=50,  # Top-k sampling
            top_p=0.9,  # Nucleus sampling
        )

    end_time = time.time()
    inference_time = end_time - start_time
    total_time += inference_time

    # Decode generated text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Print the generated output
    print(f"📝 LLM Output {i + 1}:")
    print(output_text)
    print(f"⏱️ Inference Time: {inference_time:.4f} seconds")
    print("=" * 80)

# Calculate and print the average inference time
average_time = total_time / num_iterations
print(f"🎯 Average LLM Inference Time: {average_time:.4f} seconds")
print("🎉 All 10 inferences completed!")
