export PYTHONPATH=$(pwd)

python main.py \
--model qwen2-7b \
--model_path /leonardo_work/EUHPC_D12_071/projects/mm-exams/models/qwen2-vl-7b-instruct/ \
--dataset /leonardo_work/EUHPC_D12_071/projects/mm-exams/dataset/test.hf \
--num_samples 3
#--selected_langs ['slovak'] \
#--dataset dokato/multimodal-SK-exams \
#--api_key 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 \