import argparse
import json
import re


def format_other_answers(sample):
    answer = sample["reasoning_by_pangea"]
    pattern = r"<ANSWER>\s*([A-Za-z])\s*</ANSWER>"
    match = re.search(pattern, answer, re.IGNORECASE)
    if match:  # Extract and convert answer letter
        letter = match.group(1).upper()
        election = ord(letter) - ord("A")
        sample["prediction_by_pangea"] = election
        return sample
    match = re.search(r"Answer: assistant (\w)", answer)
    if match:
        letter = match.group(1).upper()
        election = ord(letter) - ord("A")
        sample["prediction_by_pangea"] = election
        return sample
    return sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="path to raw inference",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="path to formatted inference",
    )
    args = parser.parse_args()
    return args


def recategorize(dataset, category_map, category):
    with open(category_map, "r") as file:
        mapping = json.load(file)

    level_mapping = mapping.get(category, {})
    reverse_mapping = {}
    for new_category, old_categories in level_mapping.items():
        for old_category in old_categories:
            reverse_mapping[old_category] = new_category

    recategorized_dataset = []
    for item in dataset:
        original_category = item.get(category, "").strip()
        new_category = reverse_mapping.get(original_category, original_category)
        recategorized_item = item.copy()
        recategorized_item[category] = new_category
        recategorized_dataset.append(recategorized_item)

    return recategorized_dataset


def main():
    args = parse_args()
    with open(args.results_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"Number of inference output: {len(data)}")
    prediction_field = next(
        (key for key in data[0].keys() if key.startswith("prediction_by_")),
        "prediction",
    )

    def get_missing_answers(data):
        missing_answers = []
        for sample in data:
            if sample[prediction_field] not in [0, 1, 2, 3]:
                missing_answers.append(sample)
        return missing_answers

    missing = get_missing_answers(data)
    print(f"Number of missing answers: {len(missing)}")

    def format_answers_molmo(data):
        formatted_data = []
        for sample in data:
            original = sample[prediction_field]
            answer = sample["reasoning"]
            if original not in [0, 1, 2, 3]:
                if answer:
                    pattern = r"([ABCD])|([0123])"
                    match = re.search(pattern, answer, re.IGNORECASE)
                    if match:
                        if match.group(1):
                            letter = match.group(1).upper()
                            sample["prediction"] = ord(letter) - ord("A")
                        else:
                            sample["prediction"] = int(match.group(2))
            formatted_data.append(sample)
        return formatted_data

    def format_answers_pangea(data, pangea=False):
        formatted_data = []
        for sample in data:
            original = sample[prediction_field]
            answer = sample["reasoning"]
            if original not in [0, 1, 2, 3]:
                if answer:
                    if pangea:
                        answer = answer.split("Answer: assistant ")[-1]
                    pattern = r"assistant <ANSWER>\s*([ABCD])\s*</ANSWER>"
                    match = re.search(pattern, answer)
                    if not match:
                        match = re.search(
                            r"assistant\s+([ABCD])", answer, re.IGNORECASE
                        )
                    if not match:
                        pattern = r"correct answer is <ANSWER>\s*([ABCD])\s*</ANSWER>"
                        match = re.search(pattern, answer)
                    if not match:
                        pattern = r"correct answer is\s*([ABCD])"
                        match = re.search(pattern, answer)
                    if not match:
                        pattern = r"correct answer is option\s*([ABCD])"
                        match = re.search(pattern, answer)
                    if not match:
                        match = re.search(r"assistant\s*<\s*([ABCD])\s*", answer)
                    if not match:
                        match = re.search(r"\n\n\s*([ABCD])", answer)
                    if not match:
                        match = re.search(r"\n\n<ANSWER>\s*([ABCD])", answer)
                    if not match:
                        match = re.search(r"is option\s*([ABCD])", answer)
                    if not match:
                        match = re.search(r"answer is:\s*([ABCD])", answer)
                    if not match:
                        match = re.search(r"<ANSWER>\s*([ABCD])\s*</ANSWER>", answer)
                    if not match:
                        match = re.search(r"[Oo]ption\s*\(\s*([ABCD])\s*", answer)
                    if not match:
                        match = re.search(r"([ABCD])[\.\s]?", answer)
                    if match:
                        try:
                            letter = match.group(1).upper()
                            sample[prediction_field] = ord(letter) - ord("A")
                        except:
                            print(match)
                    if not match:
                        matches = re.findall(r"<ANSWER>\s*([ABCD])\s*</ANSWER>", answer)
                        if len(matches) >= 4:
                            letter = matches[3].upper()
                            sample[prediction_field] = ord(letter) - ord("A")

            formatted_data.append(sample)
        return formatted_data

    formatted_data = format_answers_pangea(data)
    assert len(formatted_data) == len(data)
    missing = get_missing_answers(formatted_data)
    print(f"Number of missing after answers formatting: {len(missing)}")
    import code

    code.interact(local=locals())
    if args.save_path is not None:
        with open(args.save_path, "w") as f:
            json.dump(formatted_data, f, indent=2)

        print(f"Formatted data saved to: {args.save_path}")


if __name__ == "__main__":
    main()
