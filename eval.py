# Run evaluations, experiments and plots here

import argparse
import pandas as pd
from datasets import load_dataset, Dataset
from eval_utils import (
    EVALUATION_STYLES,
    perform_complete_evaluation,
    perform_accuracy_evaluation,
    perform_descriptive_statistics,
    perform_experiments,
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
    args = parser.parse_args()
    return args

def run_evaluation(results, style, output_folder):
    if style not in EVALUATION_STYLES: raise NameError(f'{style} is not a supported evaluation style. Evaluation styles: {EVALUATION_STYLES}')
    if style == 'complete':
        perform_complete_evaluation(results, output_folder)
    if style == 'accuracy':
        perform_accuracy_evaluation(results, output_folder)
    if style == 'statistics':
        perform_descriptive_statistics(results, output_folder)
    if style == 'experiments':
        perform_experiments(results, output_folder)
    if style == 'plotting':
        perform_plots(output_folder)

def load_dataset_from_entry(args):

    if isinstance(args.results_dataset, dict):
        df_dataset = pd.DataFrame(args.results_dataset)
    elif isinstance(args.results_dataset, str):
        df_dataset = pd.read_json(args.results_dataset)
    elif isinstance(args.results_dataset, Dataset): 
        results = load_dataset(args.results_dataset)['train']
        df_dataset = results.to_pandas()
    else:
        raise TypeError(f'Unexpected dataset format: {args.results_dataset}')       

    if args.num_samples != 'all':
        df_dataset = df_dataset.head(int(args.num_samples))
    
    return df_dataset

def main():
    args = parse_args()

    # Load dataset.
    results = load_dataset_from_entry(args)

    run_evaluation(results, args.evaluation_style, args.output_folder)
    
if __name__ == '__main__':
    main()