# Run evaluations, experiments and plots here

import argparse
import pandas as pd
import os
from datasets import load_dataset, Dataset
from huggingface_hub import login
from eval_utils import (
    EVALUATION_STYLES,
    perform_complete_evaluation,
    perform_metrics,
    perform_descriptive_statistics,
    perform_plots
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=str,
        default="3",
        help="number of samples to evaluate",
    )
    parser.add_argument(
        "--results_dataset", 
        type=str, 
        required=True, 
        help="dataset name or path"
    )
    parser.add_argument(
        "--is_hf_dataset",
        type=str, 
        required=True,
        help= "boolean to determine if it is a name/path from a dataset in HF." 
    )
    parser.add_argument(
        "--hf_token",
        type=str, 
        help= "HF token to access private datasets (if needed)" 
    )
    parser.add_argument(
        "--evaluation_style", 
        type=str, 
        required=True, 
        help=f"type of evaluation to perform. Should be one of: {', '.join(EVALUATION_STYLES)}"
    )
    parser.add_argument(
        "--output_folder",
        type=str, 
        required=True,
        help= "where to save output results" 
    )
    parser.add_argument(
        "--filter_data_by",
        type=str,
        default= None, 
        help= "how to filter the entry dataset" 
    )
    args = parser.parse_args()
    return args

def run_evaluation(results, style, output_folder):

    if style not in EVALUATION_STYLES: raise NameError(f'{style} is not a supported evaluation style. Evaluation styles: {EVALUATION_STYLES}')

    os.makedirs(output_folder, exist_ok=True)

    if style == 'complete':
        perform_complete_evaluation(results, output_folder)
    if style == 'metrics':
        perform_metrics(results, output_folder)
    if style == 'statistics':
        perform_descriptive_statistics(results, output_folder)
    if style == 'plotting':
        perform_plots(results, output_folder)

def load_dataset_from_entry(args):

    if args.hf_token:
        print('Logging in...')
        login(args.hf_token)

    if isinstance(args.results_dataset, str):
        if args.is_hf_dataset == 'True':
            results = load_dataset(args.results_dataset)['train']
            df_dataset = results.to_pandas()
        else:
            df_dataset = pd.read_json(args.results_dataset)
    else:
        raise TypeError(f'Unexpected dataset format: {args.results_dataset}')       

    if args.num_samples != 'all':
        df_dataset = df_dataset.head(int(args.num_samples))

    if args.filter_data_by:
        filter_by = args.filter_data_by
        if filter_by == 'only_image_png':
            df_dataset = df_dataset[df_dataset['image_png'] != 'None']
        if filter_by == 'exclude_image_png':
            df_dataset = df_dataset[df_dataset['image_png'] == 'None']

    print(f'Filtered dataset length: {len(df_dataset)}')
    return df_dataset

def main():
    args = parse_args()

    # Load dataset.
    results = load_dataset_from_entry(args)

    run_evaluation(results, args.evaluation_style, args.output_folder)
    
if __name__ == '__main__':
    main()