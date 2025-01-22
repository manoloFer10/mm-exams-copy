# Run evaluations, experiments and plots here

import argparse
from model_utils import SUPPORTED_MODELS
from datasets import load_dataset
from eval_utils import (
    EVALUATION_STYLES,
    perform_complete_evaluation,
    perform_accuracy_evaluation,
    perform_overall_statistics
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

def run_evaluation(results, args):
    style = args.evaluation_style

    if style not in EVALUATION_STYLES: raise NameError(f'{style} is not a supported evaluation style. Evaluation styles: {EVALUATION_STYLES}')

    if style == 'complete':
        perform_complete_evaluation(results)
    if style == 'accuracy':
        perform_accuracy_evaluation(results)
    if style == 'statistics':
        perform_overall_statistics(results)


def main():
    args = parse_args()

    # Load dataset.
    if args.num_samples == 'all':
        results = load_dataset(args.results_dataset)['train']
    else: 
        results = load_dataset(args.results_dataset)['train'].select(range(int(args.num_samples)))
    
    run_evaluation(results, args)
    
if __name__ == '__main__':
    main()