export CUDA_VISIBLE_DEVICES="0"
model_name=TimeLLM
learning_rate=0.05
llama_layers=1
master_port=8097
num_process=1
batch_size=1
d_model=32
d_ff=128
count_hyper=0

comment='TimeLLM-ETTh1'
itts='0'
train_epochs=25
methods_h="ori"
#methods_h="ori"
for method in $methods_h;
do
for itt in $itts;
do
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96_$method \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --llm_dim 4096 \
  --llm_path /root/autodl-tmp/llama-1b-hf-32768-fpf \
  --des 'Exp' \
  --itr $itt \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate 0.05 \
  --llm_layers $llama_layers \
  --count_hyper $count_hyper \
  --train_epochs $train_epochs \
  --model_comment $comment
 
done 
done