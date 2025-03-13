import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset

# ✅ 设置 Llama-7B 模型路径
model_path = "/root/autodl-tmp/llama-7b"

# ✅ 加载 Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path, local_files_only=True)

# ✅ 修复 `ValueError: Asking to pad but the tokenizer does not have a padding token`
tokenizer.pad_token = tokenizer.eos_token  # 使用 `eos_token` 作为 `pad_token`

# ✅ 加载 Llama 语言模型
model = LlamaForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"  # 自动分配到 GPU
)

print("✅ Llama-7B 模型加载成功！")

# ✅ 配置 LoRA（低秩适配）
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 语言模型任务
    r=8,  # LoRA 低秩维度（控制可训练参数的大小）
    lora_alpha=16,  # LoRA 缩放因子
    lora_dropout=0.05,  # Dropout 防止过拟合
    target_modules=["q_proj", "v_proj"]  # 仅调整 Q 和 V 权重，减少显存占用
)

# ✅ 将 Llama 模型转换为 LoRA 训练模式
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 显示可训练的参数量

# ✅ 加载数据集（这里使用 Hugging Face 数据集）
dataset = load_dataset("Abirate/english_quotes", split="train[:1000]")  # 仅取 1000 条数据

# ✅ Tokenize 数据集
def tokenize_function(examples):
    return tokenizer(examples["quote"], padding="max_length", truncation=True, max_length=256)

# ✅ 修复 `dataset.map()` 报错，保证 `num_proc` 仅在 `datasets` 支持时使用
try:
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4)
except:
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

print("✅ 数据 Tokenization 完成！")

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
    gradient_accumulation_steps=8,  # 适用于小 batch
    optim="adamw_torch",  # 使用更稳定的 AdamW
    evaluation_strategy="steps",  # 评估策略
    eval_steps=100,  # 每 100 步评估一次
)

# ✅ 使用 `Trainer` 进行 LoRA 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)
trainer.train()

# ✅ 保存微调后的 LoRA 权重
model.save_pretrained("./lora_llama7b")
tokenizer.save_pretrained("./lora_llama7b")

print("🎉 LoRA 训练完成，已保存微调后的权重！")

# ✅ 释放显存，防止 OOM
del model
torch.cuda.empty_cache()

# ✅ 重新加载 Llama-7B 并应用 LoRA 训练好的参数
base_model = LlamaForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ✅ 加载微调后的 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "./lora_llama7b")
model.to("cuda")  # 确保 LoRA 模型在 GPU 上

print("\n🔄 使用 LoRA 训练后的模型进行推理...\n")

input_text = "请用中文介绍 Llama 语言模型。"

# ✅ 运行 10 次推理
for i in range(10):
    print(f"🔄 第 {i+1} 次调用 LLM...\n")

    # Tokenize 输入文本
