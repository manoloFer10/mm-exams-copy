import json
import pandas as pd


def recategorize(dataset, mapping, category):

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
    data_file_paths = [
        'Final(zero-shot)all_model_answers_clean.json'
    ]

    category_map_path = 'recategorization\categorization.json'
    categories = ['level', 'category_en', 'general_category_en']

    with open(category_map_path, "r") as file:
        mapping = json.load(file)

    for data_file in data_file_paths:
        recategorized = None
        with open(data_file, "r") as file:
            df = json.load(file)

        for category in categories:
            if recategorized is None:
                recategorized = recategorize(df, mapping, category)
            else: 
                recategorized = recategorize(recategorized, mapping, category)

        with open('recategorization/r_' + data_file, 'w') as f:
            json.dump(recategorized, f, indent=2)

if __name__ == '__main__':
    main()