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
    args = parser.parse_args()
    return args

def run_evaluation(results, style):

    if style not in EVALUATION_STYLES: raise NameError(f'{style} is not a supported evaluation style. Evaluation styles: {EVALUATION_STYLES}')

    if style == 'complete':
        perform_complete_evaluation(results)
    if style == 'accuracy':
        perform_accuracy_evaluation(results, 'eval_results/results_accuracy')
    if style == 'statistics':
        perform_descriptive_statistics(results)
    if style == 'experiments':
        perform_experiments(results)
    if style == 'plotting':
        perform_plots()

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

    run_evaluation(results, args.evaluation_style)
    
if __name__ == '__main__':
    main()