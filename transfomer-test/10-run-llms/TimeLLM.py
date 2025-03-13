from math import sqrt
from models.Attention import MultiHeadAttention
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class Encoder_TRSF(nn.Module):
    def __init__(self, input_dim=0, hidden_dim=768, num_heads=8, num_encoder_layers=1):
        super(Encoder_TRSF, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        return x


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class SimpleLinr(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        # self.llama_model_path = '/root/autodl-tmp/llama-7b/'
        self.llm_path = configs.llm_path
        self.model_id = configs.model_id
        self.dropout = nn.Dropout(configs.dropout)
        self.down_proj = configs.down_proj

        if 'ori' in self.model_id:
            if configs.llm_model == 'LLAMA':
                # self.llama_model_path = '/root/autodl-tmp/llama-1b-hf-32768-fpf/'
                # print("come here,",self.llama_model_path)
                # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
                # self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
                self.llama_config = LlamaConfig.from_pretrained(self.llm_path)
                self.llama_config.num_hidden_layers = configs.llm_layers
                self.llama_config.output_attentions = True
                self.llama_config.output_hidden_states = True
                try:
                    # self.llm_model = LlamaModel.from_pretrained(self.llm_path , local_files_only=True)
                    self.llm_model = LlamaModel.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                        # 'huggyllama/llama-7b',
                        self.llm_path,
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    print(self.llama_model_path)
                    print('no we do not download....')
                    exit()
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                        # 'huggyllama/llama-7b',
                        self.llm_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    exit()

            elif configs.llm_model == 'QWEN':
                self.llama_config = LlamaConfig.from_pretrained(self.llm_path)
                self.llama_config.num_hidden_layers = configs.llm_layers
                self.llama_config.output_attentions = True
                self.llama_config.output_hidden_states = True
                try:
                    self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_path, local_files_only=True)
                    print("begin to add lora")
                    # ✅ 配置 LoRA
                    lora_config = LoraConfig(
                        r=8,  # 低秩维度
                        lora_alpha=32,  # LoRA 影响因子
                        lora_dropout=0.1,  # Dropout 防止过拟合
                        bias="all",
                        #bias="none",
                        #bias="lora_only", # set to only lora layers to train
                        #target_modules=["q_proj", "k_proj", "v_proj"],  # 让 LoRA 适用于 W_Q, W_K, W_V
                        target_modules=[ "q_proj", "k_proj"],  # 让 LoRA 适用于  W_K, W_V
                        #task_type="CAUSAL_LM"  # 自回归语言建模任务
                    )

                    # ✅ 让模型支持 LoRA
                    self.llm_model  = get_peft_model(self.llm_model , lora_config)

                    # ✅ 打印可训练参数
                    self.llm_model .print_trainable_parameters()
                    print("lora added")

                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    print(self.llama_model_path)
                    print('no we do not download....')
                    exit()
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.llm_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    exit()

            elif configs.llm_model == 'GPT2':

                # self.gpt_model_path = '/root/autodl-tmp/gpt2/'
                self.gpt2_config = GPT2Config.from_pretrained(self.llm_path)

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        self.llm_path,
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    print('no we do not download....')
                    exit()

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        self.llm_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    print('no we do not download....')
                    exit()
            elif configs.llm_model == 'BERT':
                self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

                self.bert_config.num_hidden_layers = configs.llm_layers
                self.bert_config.output_attentions = True
                self.bert_config.output_hidden_states = True
                try:
                    self.llm_model = BertModel.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.bert_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = BertModel.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.bert_config,
                    )

                try:
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            else:
                raise Exception('LLM model is not defined')
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token
            for param in self.llm_model.parameters():
                param.requires_grad = False
            if configs.prompt_domain:
                self.description = configs.content
            else:
                self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

            # print('description:', self.description)
            # 获取 LLM（大型语言模型）的词嵌入矩阵（Word Embedding Matrix）
            self.word_embeddings = self.llm_model.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
            self.num_tokens = 1000
            # 创建了一个全连接层（Linear Layer），其作用是将一个大小为 self.vocab_size 的输入映射到 self.num_tokens 维度的输出
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
            self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
            self.down_proj = nn.Linear(self.d_llm, self.d_ff)

        # d_model: 目标嵌入维度（Transformer 模型的隐藏维度）。
        # patch_len: 每个 Patch（小块）的长度（滑动窗口大小）。
        # stride: Patch 滑动窗口的步长（决定相邻 Patch 的重叠程度）。
        # dropout: 训练时的 dropout 比例。
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        # ori
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        # configs.enc_in	Number of input features
        # affine=False	    No trainable scale/shift
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        if "removeLLM" in self.model_id:
            # del self.tokenizer , self.llm_model, self.word_embeddings, self.mapping_layer, self.reprogramming_layer
            del self.output_projection

        if 'llm_to_trsf' in configs.model_id:
            # del self.tokenizer , self.llm_model, self.word_embeddings, self.mapping_layer, self.reprogramming_layer
            del self.output_projection
            self.basic_trsf = Encoder_TRSF(hidden_dim=configs.d_model)

        if 'llm_to_attn' in configs.model_id:
            # del self.tokenizer , self.llm_model, self.word_embeddings, self.mapping_layer, self.reprogramming_layer
            del self.output_projection
            self.basic_attn = MultiHeadAttention(d_model=configs.d_model)

        # except ori
        if 'ori' not in configs.model_id:
            self.output_projection = SimpleLinr(self.patch_nums * configs.d_model, self.pred_len,
                                                head_dropout=configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    # x_enc 编码后的时间序列输入
    # x_mark_enc：时间戳编码（encoder 部分），可能包含时间信息，如日期、小时、星期等辅助信息。
    # x_dec：解码器的输入数据，通常是目标序列的初始部分。
    # x_mark_dec：时间戳编码（decoder 部分），解码器使用的时间信息。
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # print("x_enc, x_mark_enc, x_dec, x_mark_dec", x_enc, x_mark_enc, x_dec, x_mark_dec)
        # 归一化输入数据 x_enc
        x_enc = self.normalize_layers(x_enc, 'norm')

        # batch_size,time_steps,num_features
        B, T, N = x_enc.size()
        # 交换,调整维度
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        if "removeLLM" in self.model_id:
            x_enc = x_enc.reshape(B, N, T)
            # torch.Size([14, 64, 32])
            dec_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
            dec_out = torch.reshape(
                dec_out, (B, n_vars, -1)).contiguous()

            dec_out = self.output_projection(dec_out)
            # dec_out.shape [14, 1, 96]
            dec_out = dec_out.permute(0, 2, 1).contiguous()

            dec_out = self.normalize_layers(dec_out, 'denorm')

            return dec_out

        elif 'llm_to_trsf' in self.model_id:
            x_enc = x_enc.reshape(B, N, T)
            # print('1', x_enc.shape) torch.Size([14, 512, 1])
            enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
            # torch.Size([14, 64, 32])
            dec_out = self.basic_trsf(enc_out)
            # dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
            # print('5', dec_out.shape)  torch.Size([14, 240, 4096])

            dec_out = torch.reshape(
                dec_out, (B, n_vars, -1)).contiguous()

            dec_out = self.output_projection(dec_out)
            # dec_out.shape [14, 1, 96]
            dec_out = dec_out.permute(0, 2, 1).contiguous()

            dec_out = self.normalize_layers(dec_out, 'denorm')
            return dec_out

        elif 'llm_to_attn' in self.model_id:
            x_enc = x_enc.reshape(B, N, T)
            # print('1', x_enc.shape) torch.Size([14, 512, 1])
            enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
            # prompt_embeddings.shape              enc_out.shape
            dec_out, _ = self.basic_attn(enc_out, enc_out, enc_out)
            # dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
            # print('5', dec_out.shape)  torch.Size([14, 240, 4096])

            dec_out = torch.reshape(
                dec_out, (B, n_vars, -1)).contiguous()

            dec_out = self.output_projection(dec_out)
            # dec_out.shape [14, 1, 96]
            dec_out = dec_out.permute(0, 2, 1).contiguous()

            dec_out = self.normalize_layers(dec_out, 'denorm')

            return dec_out

        elif 'ori' in self.model_id:
            min_values = torch.min(x_enc, dim=1)[0]
            max_values = torch.max(x_enc, dim=1)[0]
            medians = torch.median(x_enc, dim=1).values
            # 这些过去的时间点与当前时间点的模式最相似
            lags = self.calcute_lags(x_enc)
            trends = x_enc.diff(dim=1).sum(dim=1)

            prompt = []
            for b in range(x_enc.shape[0]):
                min_values_str = str(min_values[b].tolist()[0])
                max_values_str = str(max_values[b].tolist()[0])
                median_values_str = str(medians[b].tolist()[0])
                lags_values_str = str(lags[b].tolist())
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description}"
                    f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                    "Input statistics: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                    f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
                )
                prompt.append(prompt_)

            # print('prompt:', prompt)

            x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                    max_length=2048).input_ids
            # 将 prompt 转换成 LLM（大语言模型）的输入嵌入表示（embeddings）
            prompt_embeddings = self.llm_model.get_input_embeddings()(
                prompt.to(x_enc.device))  # (batch, prompt_token, dim)

            # 对 LLM 词嵌入矩阵 (word_embeddings) 进行线性映射（降维或变换），然后恢复原来的维度顺序。
            # self.word_embeddings 词嵌入矩阵 word_embeddings∈R V×D V = 词汇表大小 (vocab_size) D = 词嵌入维度 (embedding_dim)
            #       从  (V,D) 变成  (D,V)
            # mapping_layer 是一个全连接层（Linear Layer），其作用是将一个大小为 self.vocab_size 的输入映射到 self.num_tokens 维度的输出
            #  mapping_layer:RD→RM，降维后的嵌入维度 (num_tokens)
            # 计算完映射后，再次交换维度，把数据恢复回(𝑉,𝑀)。
            # reprogramming_layer 会使用.  可以优化到全局，不用每次计算
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

            # 时间步 T 被移动到最后，这种格式更适用于 CNN 或 Transformer 的 patch_embedding 处理
            # .contiguous() 让数据存储在连续内存中，以提高计算效率
            x_enc = x_enc.permute(0, 2, 1).contiguous()
            # bfloat16（Brain Floating Point）是一种 低精度浮点格式，
            # 将时间序列数据转换为可用于 Transformer 的 token 表示。
            #  将时间序列数据切分成 Patch（小块）并进行嵌入处理，类似于 ViT（Vision Transformer） 处理图像的方式
            ts_patches, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))

            patch_embeddings = self.reprogramming_layer(ts_patches, source_embeddings, source_embeddings)
            llama_enc_out = torch.cat([prompt_embeddings, patch_embeddings], dim=1)
            # print("begin to call LLMs")
            # 传入已经嵌入（embedding）后的数据。  获取 LLM 处理后的隐藏状态 形状：(𝐵,𝑇,𝑑llm)  --LLM 的隐藏维度
            # dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
            outputs = self.llm_model(inputs_embeds=llama_enc_out, output_hidden_states=True)
            dec_out = outputs.hidden_states[-1]  # 取最后一层隐藏状态

            # print("end calling LLMs,", dec_out)
            # 只保留 d_ff 维度的前 self.d_ff 个通道。
            # 降维，使 dec_out 适配后续的 d_ff 维度计算
            #       （d_ff 是 Transformer 的 feed-forward 层维度）
            # (B,T,dllm)→(B,T,dff)
            # !!!! 如果 self.d_ff 远小于 D，且没有精挑细选特征，可能会丢失关键信息。用 Linear 层代替截取!!!!!
            # print("原始 dec_out 形状:", dec_out.shape)  # 打印原始形状
            # 原始 dec_out 形状: torch.Size([2, 354, 4096]) 截取后 dec_out 形状: torch.Size([2, 354, 128])

            # print("without down_proj 2")
            dec_out = dec_out[:, :, :self.d_ff]
            # if self.down_proj:
            #    dec_out = self.down_proj(dec_out)
            # else:
            #    print("without down_proj")
            #    dec_out = dec_out[:, :, :self.d_ff]

            # print("down_proj后 dec_out 形状:", dec_out.shape)  # 检查截取后的形状

            # to (−1,nvars,T′,dff) ,n_vars 代表变量的数量,T'是 Transformer 输出的 seq_len
            dec_out = torch.reshape(
                dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
            # 交换维度顺序 (B,nvars,dff,T′)
            dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

            # 1. 只保留最后 patch_nums 个时间步的数据。
            # 2. 一个 nn.Linear 层，将 d_ff 映射到目标 target_window 维度。
            dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
            # LLM 适配最终的输出格式 (B,dff,nvars)
            dec_out = dec_out.permute(0, 2, 1).contiguous()

            # 去标准化（De-Normalization），恢复到原始数据分布。
            dec_out = self.normalize_layers(dec_out, 'denorm')

            return dec_out

    # 傅里叶变换（FFT） 计算时间序列的滞后相关性（Lag Correlation）
    # 通过频域运算找到最重要的滞后时间步
    def calcute_lags(self, x_enc):
        # 快速傅里叶变换（FFT）
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = q_fft
        # 共轭复数（Complex Conjugate），频域中的自相关功率谱
        res = q_fft * torch.conj(k_fft)
        # 逆快速傅里叶变换，将频域的自相关信号转换回时域,计算自相关（Auto-correlation），
        # dim=-1 代表沿着**时间轴（seq_len 维度）**进行转换。
        # corr 是一个时间序列，表示不同滞后时间步的相关性。
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


# 自定义的注意力机制（Attention Layer）
class ReprogrammingLayer(nn.Module):
    # d_model：目标嵌入的维度（通常来自 Transformer）。
    # n_heads：多头注意力机制的头数（Multi-Head Attention）。
    # d_keys（可选）：注意力计算时，Key 维度的大小（默认 d_model // n_heads）。
    # d_llm：LLM 模型的嵌入维度（与 d_model 可能不同）。
    # attention_dropout：注意力分数的 Dropout，防止过拟合。
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        # 目标嵌入（Target Embedding） 投影到 Query 空间，形状 (B, L, d_model) → (B, L, d_keys * n_heads)。
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        # 将 源嵌入（Source Embedding） 投影到 Key 空间，形状 (S, d_llm) → (S, d_keys * n_heads)。
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        # 将 值嵌入（Value Embedding） 投影到 Value 空间，形状 (S, d_llm) → (S, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        # 最终输出时再映射回 d_llm 维度。
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    # 将目标（Target）嵌入重新编程（Reprogram）为源（Source）嵌入，并通过注意力机制进行加权计算。
    # 类似于 Cross-Attention，但用于 time series patch 和LLM（大语言模型）嵌入对齐。
    def forward(self, target_embedding, source_embedding, value_embedding):
        # 批量大小 B，序列长度 L
        B, L, _ = target_embedding.shape
        # 源嵌入序列长度 S
        S, _ = source_embedding.shape
        H = self.n_heads

        # target_embedding 变成 (B, L, H, d_keys)，供 Query 使用
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        # source_embedding 变成 (S, H, d_keys)，供 Key 使用
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        # value_embedding 变成 (S, H, d_keys)，供 Value 使用。
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # 跨模态（Cross-Modal）注意力
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        # 对注意力分数进行缩放，防止梯度消失（与 Transformer 相同）
        scale = 1. / sqrt(E)

        # 使用 einsum 计算 Query @ Key^T。 每个 Query 位置 l 与所有 Key 位置 s 计算相似度
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        # 在 S 维度（Source 维度）上进行 Softmax，得到注意力权重
        # 防止过拟合，减少依赖于某些特定 Key。
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # 加权求和，计算 Reprogramming 结果
        #   计算 Attention @ Value，得到新的 Target 表示：
        #       A 形状 (B, H, L, S)
        #       value_embedding 形状 (S, H, d_keys)
        #       计算结果 reprogramming_embedding 形状 (B, L, H, d_keys)
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
