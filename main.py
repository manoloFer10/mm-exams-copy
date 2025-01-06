import argparse
import numpy as np
from datasets import load_dataset
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=str,
        default="5",
        help="number of samples to test",
    )
    parser.add_argument(
        "--setting",
        type=int,
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
        type=str,
        default=None,
        help="list of strings of languages",
    )
    parser.add_argument(
        "api_key", type=str, default=None, help="explicitly give an api key"
    )

    # add other testing parameters

    args = parser.parse_args()
    return args


def test_lang(args, lang, api_key):

    model = args.model
    setting = args.setting

    output_folder = f"outputs/{setting}/mode_{model}/{lang}"
    os.makdirs(output_folder, exist_ok=True)

    if setting == "few-shot":

        fewshot_samples = generate_fewshot_samples(lang)
    else:
        fewshot_samples = {}

    # load dataset
    dataset = load_dataset(args.dataset)
    dataset = dataset.filter(lambda sample: sample["language"] == lang)

    if args.max_test_samples != "all":
        max_test_samples = int(args.max_test_samples)
        dataset = dataset.select(range(max_test_samples))

    # generate prompts
    all_prompts = []
    for question in dataset:
        prompt = generate_prompt(lang, setting, model, question, fewshot_samples)
        all_prompts.append(prompt)


def generate_one_example(question, lang):
    # TODO: add other languages and methods - check images as answers
    answer_word = {"english": "Answer:"}
    prompt = (
        question["question"]
        + "\n"
        + "\n".join(question["options"])
        + f"\n{answer_word}"
    )
    return prompt


def generate_fewshot_samples():
    pass


def generate_prompt(lang, setting, model, question, fewshot_samples):
    # TODO: add all the languages
    if lang == "english":
        hint = f"The following is a multiple choice question about {question['category_original_lang']}."
    else:
        raise NotImplemented

    # TODO: add chat models and languages
    if setting == "zero-shot":
        if lang == "english":
            hint += "Please only give the correct option, without nay other details or explanations."
        else:
            raise NotImplemented

    if setting == "zero-shot":
        prompt = hint + "\n\n" + generate_one_example(question, lang)
    elif setting == "few-shot":
        dev_questions_list = fewshot_samples[question["level"]][
            question["subject_category"]
        ]
        prompt = (
            hint
            + "\n\n"
            + "\n\n".join(dev_questions_list)
            + "\n\n"
            + generate_one_example(question, lang)
        )
    else:
        raise NotImplementedError

    return prompt


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # select test parameters
    all_langs = []
    selected_langs = eval(args.selected_langs) if args.selected_langs else all_langs

    # read api key
    api_key = args.api_key

    for lang in selected_langs:
        test_lang(args, lang, api_key)
