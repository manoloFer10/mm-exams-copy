import argparse
import ast
import pandas as pd
import numpy as np
from datasets import load_dataset
import os
import random
from typing import List, Dict
from predict_answers import run_answer_prediction

from model_utils import initialize_model, query_model, SUPPORTED_MODELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=str,
        default="3",
        help="number of samples to test",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="zero-shot",
        help="[few-shot, zero-shot]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--selected_langs",
        type=list,
        default=None,
        help="list of strings of languages",
    )
    parser.add_argument(
        "--api_key", 
        type=str, 
        default=None, 
        help="explicitly give an api key"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset name or path"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Specify the model to use (must be one from the following: {', '.join(SUPPORTED_MODELS)})",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint or name",
    )
    args = parser.parse_args()
    return args


def test_lang(args, lang: str): # Manu: already done by run_answer_prediction
    # TODO: needs refactor to fit model_predict calls
    setting = args.setting

    output_folder = f"outputs/{setting}/mode_{args.model}/{lang}"
    os.makdirs(output_folder, exist_ok=True)

    if setting == "few-shot":
        fewshot_samples = generate_fewshot_samples(lang)
    else:
        fewshot_samples = {}

    # load dataset
    dataset = load_dataset(args.dataset)
    dataset = dataset.filter(lambda sample: sample["language"] == lang)

    if args.num_samples != "all":
        max_test_samples = int(args.num_samples)
        dataset = dataset.select(range(max_test_samples))

    # generate prompts
    all_prompts = [
        generate_prompt(lang, args.setting, args.model, question, fewshot_samples)
        for question in dataset
    ]

    # initialize and query
    if args.model in ["qwen", "llava"]:
        if args.model == "qwen":
            model, tokenizer = initialize_model("qwen", args.model_path)
            predictions = query_model("qwen", model, tokenizer, all_prompts)
        elif args.model == "llava":
            model, tokenizer, image_processor = initialize_model(
                "llava", args.model_path
            )
    elif args.model == "chatgpt":
        model, tokenizer, image_processor = None, None, None
    else:
        raise ValueError(f"Unsuported model: {args.model}")

    # Save Results
    save_results(output_folder, dataset, predictions, all_prompts)
    print(f"Evaluation completed for {lang}. Results saved to {output_folder}")



def save_results(
    output_folder: str,
    results: Dict
):
    output_path = os.path.join(output_folder, "results.json")
    df = pd.read_json(results)
    df.to_csv(output_path, index=False)


def main():
    #TODO: pending finish
    # Manu: levantar el dataset desde args y correr run_evaluation con las config de args.

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset.
    if args.num_samples == 'all':
        dataset = load_dataset(args.dataset)['train']
    else: 
        dataset = load_dataset(args.dataset)['train'].select(range(int(args.num_samples)))

    # Run evaluation.
    results = run_answer_prediction(dataset, args)

    # Save csv results in 'results' folder.
    output_folder = 'results'
    save_results(output_folder, results)
    print(f"Evaluation completed. Results saved to {output_folder}")


if __name__ == '__main__':
    main()