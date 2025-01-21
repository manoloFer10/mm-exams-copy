export PYTHONPATH=$(pwd)

python main.py \
--num_samples '3' \
--setting zero-shot \
--model llama \
--selected_langs ['slovak'] \
--dataset dokato/multimodal-SK-exams \
--api_key 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 \
