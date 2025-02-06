from datasets import load_from_disk
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pickle


def main():
    dataset = load_from_disk("dataset/full.hf")
    print(dataset)
    initial_questions = len(dataset)
    stats = {
        "multimodal_questions": 0,
        "questions_by_language": {},
        "multimodal_by_language": {},
        "questions_by_category": {},
        "multimodal_by_category": {},
        "questions_by_language_and_category": {},
        "multimodal_by_language_and_category": {},
    }

    # check only >=4 options (cut to 4 afterwards)
    dataset = dataset.filter(lambda example: len(example["options"]) >= 4)
    only_four_or_more_options = len(dataset)

    # Count questions with exactly 4 options
    only_four_options = sum(1 for example in dataset if len(example["options"]) == 4)

    # filtering images in options
    dataset = dataset.filter(lambda example: ".png" not in example["options"][0])

    # Define category mappings
    normalized_category_mapping = {
        # STEM
        "Mathematics": {
            "Mathematical Reasoning",
            "Mathematical Abilities",
            "Numerical Ability",
            "Maths",
            "maths",
            "math",
            "Math",
            "Algebra",
            "Geometry",
            "Statistics",
        },
        "Chemistry": {"Chemistry", "chemistry"},
        "Biology": {"Biology", "Medicine/Biology"},
        "Physics": {"Physics", "physics"},
        "Informatics": {
            "Computer Science",
            "computer science",
            "Computer Science and Information Technology",
            "Computer Knowledge",
            "computer concepts",
        },
        "Engineering": {
            "Electronics and Communications Engineering",
            "Mechanical Engineering",
            "Electrical Engineering",
            "Civil Engineering",
            "Aerospace Engineering",
            "Chemical Engineering",
            "Production and Industrial Engineering",
            "Textile Engineering and Fibre Science",
            "Biomedical Engineering",
            "Mining Engineering",
            "Geology Geophysics",
            "Geophysics",
            "Environmental Science & Engineering",
            "Engineering Sciences",
            "Electronics and Communication Engineering",
            "Metallurgical Engineering",
            "Naval Architecture and Marine Engineering",
            "Architecture and Planning",
            "Petroleum Engineering",
        },
        "Olympiads": {"olympiads"},
        "STEM (General)": {"STEM", "Science"},
        "Natural Sciences": {
            "Natural Science",
            "Life Sciences",
            "Biophysics",
            "Ecology and Evolution",
        },
        # Health Sciences
        "Medicine": {"Medicine", "Medicine/Physiology", "Medicine/Health"},
        "Nursing": {"Nursing"},
        "Health Sciences": {"Health"},
        # Language and Communication
        "Ukrainian Language": {"Ukrainian Language and Literature"},
        "Portuguese Language": {
            "Portuguese Language",
            "Portuguese Language, Literature",
            "Portuguese Language, Literature, History",
        },
        "English Language": {
            "English",
            "English Language",
            "Reading_comprehension",
            "reading_comprehension",
        },
        "Arabic Language": {"Arabic Language"},
        "Literature and Linguistics": {
            "Literature",
            "Humanities & Social Sciences - Linguistics",
        },
        # Social Sciences
        "History": {
            "Indian History",
            "Ukrainian History",
            "History",
            "History and Civics",
        },
        "Geography": {"Geography"},
        "Philosophy": {"Philosophy", "Philosophy and Logic"},
        "Social Sciences": {
            "Economics",
            "Social Studies 2",
            "Civics",
            "Indian Politics",
            "Political Studies",
            "Sociology",
            "Psychology",
            "Psychology and Sociology",
        },
        # Cognitive Skills / Aptitude
        "Reasoning": {
            "Reasoning",
            "General Intelligence and Reasoning",
            "Visual Reasoning",
            "Reasoning and General Intelligence",
        },
        "Mental Ability": {"Mental Ability", "mental ability", "IQ"},
        "Skills": {"drawing aptitude", "Design"},
        "Driving License": {
            "Driving License",
            "driving licence",
            "Driving_test",
            "Driver's License Exam",
            "Driving",
        },
        # General Knowledge, Culture and Arts
        "General Knowledge": {
            "general knowledge",
            "General Awareness",
            "Current Affairs",
        },
        "Culture": {"Culture", "Art History"},
        "Religious Studies": {"Islamic Studies", "Christian Studies"},
    }

    general_category_mapping = {
        "STEM": {
            "Mathematics",
            "Chemistry",
            "Biology",
            "Physics",
            "Informatics",
            "Engineering",
            "Natural Sciences",
            "STEM (General)",
            "Olympiads",
        },
        "Health Sciences": {"Medicine", "Nursing", "Health Sciences"},
        "Language and Communication": {
            "Ukrainian Language and Literature",
            "Portuguese Language",
            "English Language",
            "Arabic Language",
            "Literature & Linguistics",
        },
        "Social Sciences": {"History", "Geography", "Philosophy", "Social Sciences"},
        "Cognitive Skills / Aptitude": {
            "Reasoning",
            "Mental Ability",
            "Skills",
            "Driving License",
        },
        "General Knowledge, Culture and Arts": {
            "General Knowledge",
            "Culture",
            "Religious Studies",
        },
    }

    def normalize_categories(example, category_mapping, general_category_mapping):
        original_category = example["category_en"]
        normalized_category = next(
            (
                norm_cat
                for norm_cat, raw_cats in category_mapping.items()
                if original_category in raw_cats
            ),
            "Other / Uncategorized",  # Default if no match
        )
        general_category = next(
            (
                gen_cat
                for gen_cat, norm_cats in general_category_mapping.items()
                if normalized_category in norm_cats
            ),
            "Other / Uncategorized",  # Default if no match
        )
        example["category_en"] = normalized_category
        example["general_category_en"] = general_category
        return example

    dataset = dataset.map(
        partial(
            normalize_categories,
            category_mapping=normalized_category_mapping,
            general_category_mapping=general_category_mapping,
        ),
        load_from_cache_file=False,
    )

    # Collect statistics
    def collect_stats(example, stats):
        lang = example["language"]
        category = example["category_en"]
        general_category = example["general_category_en"]
        is_multimodal = example["image_png"] is not None

        # Update multimodal count
        if is_multimodal:
            stats["multimodal_questions"] += 1

        # Update language statistics
        if lang not in stats["questions_by_language"]:
            stats["questions_by_language"][lang] = 0
            stats["multimodal_by_language"][lang] = 0
        stats["questions_by_language"][lang] += 1
        if is_multimodal:
            stats["multimodal_by_language"][lang] += 1

        # Update category statistics
        if category not in stats["questions_by_category"]:
            stats["questions_by_category"][category] = 0
            stats["multimodal_by_category"][category] = 0
        stats["questions_by_category"][category] += 1
        if is_multimodal:
            stats["multimodal_by_category"][category] += 1

        # Update language and category statistics
        key = (lang, category)
        if key not in stats["questions_by_language_and_category"]:
            stats["questions_by_language_and_category"][key] = 0
            stats["multimodal_by_language_and_category"][key] = 0
        stats["questions_by_language_and_category"][key] += 1
        if is_multimodal:
            stats["multimodal_by_language_and_category"][key] += 1

        return example

    dataset.map(partial(collect_stats, stats=stats), load_from_cache_file=False)

    # Print statistics
    print(f"{initial_questions=}")
    print(f"{only_four_or_more_options=}")
    print(f"{only_four_options=}")
    print(f"{stats['multimodal_questions']=}")
    print(f"{stats['questions_by_language']=}")
    print(f"{stats['multimodal_by_language']=}")
    print(f"{stats['questions_by_category']=}")
    print(f"{stats['multimodal_by_category']=}")
    print(f"{stats['questions_by_language_and_category']=}")
    print(f"{stats['multimodal_by_language_and_category']=}")

    # minimal number of mm questions
    questions_by_language = stats["multimodal_by_language"]
    languages_to_keep = [
        lang for lang, count in questions_by_language.items() if count >= 100
    ]
    dataset = dataset.filter(lambda example: example["language"] in languages_to_keep)

    original_num_languages = len(questions_by_language)
    filtered_num_languages = len(languages_to_keep)
    print(f"Original number of languages: {original_num_languages}")
    print(f"Number of languages after filtering (>=100): {filtered_num_languages}")
    print(f"Number of questions after filtering (>=100): {len(dataset)}")

    # maximum number of questions per language to 1000

    def cap_questions_per_language(dataset, stats, max_questions_per_language=1000):
        capped_dataset = []

        # Iterate over each language
        for lang, lang_count in tqdm(stats["multimodal_by_language"].items()):
            if lang_count <= max_questions_per_language:
                # If the language has fewer than max_questions_per_language, keep all questions
                capped_dataset.extend([ex for ex in dataset if ex["language"] == lang])
            else:
                categories_in_lang = {
                    cat: count
                    for (l, cat), count in stats[
                        "multimodal_by_language_and_category"
                    ].items()
                    if l == lang
                }
                # Calculate the target number of questions per category
                total_categories = sum(
                    1 for value in categories_in_lang.values() if value != 0
                )
                target_per_category = max_questions_per_language // total_categories
                sorted_categories = dict(
                    sorted(categories_in_lang.items(), key=lambda item: item[1])
                )
                num_questions = list(sorted_categories.values())
                added = 0
                extra = 0

                for t, (category, count) in enumerate(sorted_categories.items()):
                    if count == 0:
                        continue
                    category_questions = [
                        ex
                        for ex in dataset
                        if ex["language"] == lang and ex["category_en"] == category
                    ]
                    total_categories -= 1
                    if count < target_per_category:
                        capped_dataset.extend(category_questions)
                        extra += target_per_category - count
                        added += count
                    else:
                        #     if t == len(sorted_categories) - 1:
                        #         new_cap == max_questions_per_language - added
                        #     else:
                        #         new_cap = remaining_slots // (total_categories + 2)
                        # if t < len(sorted_categories) - 1:
                        #     questions_left_cap = sum(num_questions[t + 1 :]) // (
                        #         total_categories + 2
                        #     )
                        #     if questions_left_cap < new_cap:
                        #         new_cap = questions_left_cap
                        multimodal_questions = [
                            ex
                            for ex in category_questions
                            if ex["image_png"] is not None
                        ]

                        cap = min(
                            target_per_category + (extra // (total_categories + 1)),
                            max_questions_per_language - added,
                        )
                        extra = len(multimodal_questions) - target_per_category
                        if len(multimodal_questions) < cap:
                            capped_dataset.extend(multimodal_questions)
                            added += len(multimodal_questions)
                        else:
                            sampled_questions = random.sample(
                                multimodal_questions,
                                cap,
                            )
                            capped_dataset.extend(sampled_questions)
                            added += len(sampled_questions)

        return capped_dataset

    # Apply the capping function
    stratified_dataset = cap_questions_per_language(dataset, stats)

    # Convert back to a dataset (if needed)
    stratified_dataset = dataset.from_list(stratified_dataset)

    # Print final statistics after capping
    print(f"Number of questions after capping: {len(stratified_dataset)}")

    # compute new statistics
    stratified_stats = {
        "multimodal_questions": 0,
        "questions_by_language": {},
        "multimodal_by_language": {},
        "questions_by_category": {},
        "multimodal_by_category": {},
        "questions_by_language_and_category": {},
        "multimodal_by_language_and_category": {},
    }

    stratified_dataset.map(
        partial(collect_stats, stats=stratified_stats), load_from_cache_file=False
    )

    # Print statistics
    print(f"{stratified_stats['multimodal_questions']=}")
    print(f"{stratified_stats['questions_by_language']=}")
    print(f"{stratified_stats['multimodal_by_language']=}")
    print(f"{stratified_stats['questions_by_category']=}")
    print(f"{stratified_stats['multimodal_by_category']=}")
    print(f"{stratified_stats['questions_by_language_and_category']=}")
    print(f"{stratified_stats['multimodal_by_language_and_category']=}")

    with open("dataset/stast.pkl", "wb") as f:
        pickle.dump(stats, f)

    with open("dataset/stratified_stats.pkl", "wb") as f:
        pickle.dump(stratified_stats, f)

    stratified_dataset.save_to_disk("dataset/stratified_dataset.hf")


if __name__ == "__main__":
    main()
