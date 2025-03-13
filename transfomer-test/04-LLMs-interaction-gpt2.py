import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset

# ✅ 设置 Llama-7B 模型路径
model_path = "/root/autodl-tmp/llama-7b"

# ✅ 加载 Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path, local_files_only=True)

# ✅ 加载 Llama 语言模型
model = LlamaForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"  # 自动分配到 GPU
)

print("✅ Llama-7B 模型加载成功！")

# ✅ 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 语言模型任务
    r=8,  # LoRA 低秩维度（控制可训练参数的大小）
    lora_alpha=16,  # LoRA 缩放因子
    lora_dropout=0.05,  # Dropout 防止过拟合
    target_modules=["q_proj", "v_proj"]  # 仅调整 Q 和 V 权重，减少显存占用
)

# ✅ 将 Llama 模型转换为 LoRA 模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 显示可训练的参数量

# ✅ 加载数据集（这里使用 Hugging Face 数据集）
dataset = load_dataset("Abirate/english_quotes", split="train[:1000]")  # 仅取 1000 条数据
def tokenize_function(examples):
    return tokenizer(examples["quote"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ✅ 训练参数
training_args = TrainingArguments(
    output_dir="./lora_llama7b",
    per_device_train_batch_size=1,  # LoRA 允许小 batch 训练
    per_device_eval_batch_size=1,
    num_train_epochs=1,  # 训练 1 轮（可调整）
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # 16-bit 训练减少显存
    gradient_accumulation_steps=4,  # 适用于小 batch
)

# ✅ 使用 Hugging Face `Trainer` 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)
trainer.train()

# ✅ 保存微调后的 LoRA 权重
model.save_pretrained("./lora_llama7b")

print("🎉 LoRA 训练完成，已保存微调后的权重！")

# ✅ 使用训练好的 LoRA 模型进行推理
print("\n🔄 使用 LoRA 训练后的模型进行推理...\n")

# 重新加载 LoRA 训练好的模型
model = PeftModel.from_pretrained(model, "./lora_llama7b")

input_text = "请用中文介绍 Llama 语言模型。"

# 运行 10 次推理
for i in range(10):
    print(f"🔄 第 {i+1} 次调用 LLM...\n")

    # Tokenize 输入文本
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # 执行推理（生成文本）
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_length=200,  # 生成最大长度
            temperature=0.7,  # 采样温度
            top_k=50,  # 限制 top-k 采样
            top_p=0.9,  # nucleus sampling
        )

    # 解码生成的文本
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 打印结果
    print(f"📝 第 {i+1} 次 LLM 生成的文本：")
    print(output_text)
    print("=" * 80)

print("🎉 所有 10 次推理完成！")
