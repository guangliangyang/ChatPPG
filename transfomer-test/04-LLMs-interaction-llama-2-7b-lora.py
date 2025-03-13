import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 【1】设置模型路径
# 如果本地已经下载或转换好的 Llama 2-7B 模型权重位于 "/root/autodl-tmp/llama2-7b" 文件夹中
model_path = "/root/autodl-tmp/llama-7b/"

# 【2】加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True  # 如果从本地加载，设为 True
)

# 【3】加载 Llama 2-7B 模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,  # 一般用 float16；若显存不足可尝试 bfloat16 或 8bit
    device_map="auto",
    load_in_8bit=True  # 8bit 量化，进一步降低显存占用
)

print("✅ Llama-2-7B 加载成功！\n")

# 【4】示例对话或问答
user_input = "请介绍一下 Llama 2-7B 模型的特点。"

# 将输入转为张量并移动到 GPU
inputs = tokenizer(user_input, return_tensors="pt").to("cuda")

# 生成文本
with torch.no_grad():
    output_ids = model.generate(
        inputs.input_ids,
        max_length=2048,   # 适当调大输出长度
        temperature=0.7,  # 采样温度
        top_k=50,         # top-k 采样
        top_p=0.9,        # nucleus sampling
    )

# 解码生成结果
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("📝 Llama-2-7B 生成的回答：")
print(output_text)
