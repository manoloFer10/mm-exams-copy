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


def main():
    with open("./results/few-shot/qwen25-raw.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"Number of inference output: {len(data)}")

    def get_missing_answers(data):
        missing_answers = []
        for sample in data:
            if sample["prediction_by_qwen2.5-7b"] not in [0, 1, 2, 3]:
                missing_answers.append(sample)
        return missing_answers

    missing = get_missing_answers(data)
    print(f"Number of missing answers: {len(missing)}")
    import code

    code.interact(local=locals())

    def format_answers(data):
        formatted_data = []
        for sample in data:
            original = sample["prediction_by_molmo"]
            answer = sample["reasoning_by_molmo"]
            if original not in [0, 1, 2, 3]:
                if answer:
                    # pattern = f"([ABCD])"
                    pattern = r"([ABCD])|([0123])"
                    match = re.search(pattern, answer, re.IGNORECASE)
                    if match:
                        if match.group(1):
                            letter = match.group(1).upper()
                            sample["prediction_by_molmo"] = ord(letter) - ord("A")
                        else:
                            sample["prediction_by_molmo"] = int(match.group(2))
            formatted_data.append(sample)
        return formatted_data

    formatted_data = format_answers(data)
    assert len(formatted_data) == len(data)
    missing = get_missing_answers(formatted_data)
    print(f"Number of missing after answers formatting: {len(missing)}")
    # save_path = "./results/zero-shot/molmo_formatted.json"
    # with open(save_path, "w") as f:
    #    json.dump(formatted_data, f, indent=2)

    # print(f"Formatted data saved to: {save_path}")


if __name__ == "__main__":
    main()
