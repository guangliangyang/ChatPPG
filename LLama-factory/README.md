# 【大模型微调】使用 LLaMA-Factory 微调 LLaMA3

 
## 1. environment
### 1.1  Google Colab, A100 GPU 
 
###  1.2 base model
 Meta-Llama-3.1-8B
 
## 2. LLaMA-Factory  
 
### 2.2 准备训练数据
  chatPPG.json 
 
###  2.6 model merge

将 base model 与训练好的 LoRA Adapter 合并成一个新的模型。  
 
### 2.7 model quantization

模型量化（Model Quantization）是一种将模型的参数和计算从高精度（通常是 32 位浮点数，FP32）转换为低精度（如 16 位浮点数，FP16，或者 8 位整数，INT8）的过程。