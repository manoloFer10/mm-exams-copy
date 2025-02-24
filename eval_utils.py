import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

EVALUATION_STYLES = ["complete", "accuracy", "statistics", "experiments", "plotting"]

LANGUAGES = {
    "aa": "Afar",
    "ab": "Abkhazian",
    "ae": "Avestan",
    "af": "Afrikaans",
    "ak": "Akan",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "av": "Avaric",
    "ay": "Aymara",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bh": "Bihari languages",
    "bi": "Bislama",
    "bm": "Bambara",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan; Valencian",
    "ce": "Chechen",
    "ch": "Chamorro",
    "co": "Corsican",
    "cr": "Cree",
    "cs": "Czech",
    "cu": "Church Slavic; Old Slavonic; Church Slavonic; Old Bulgarian; Old Church Slavonic",
    "cv": "Chuvash",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dv": "Divehi; Dhivehi; Maldivian",
    "dz": "Dzongkha",
    "ee": "Ewe",
    "el": "Greek, Modern (1453-)",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "ff": "Fulah",
    "fi": "Finnish",
    "fj": "Fijian",
    "fo": "Faroese",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Gaelic; Scomttish Gaelic",
    "gl": "Galician",
    "gn": "Guarani",
    "gu": "Gujarati",
    "gv": "Manx",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "ho": "Hiri Motu",
    "hr": "Croatian",
    "ht": "Haitian; Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "hz": "Herero",
    "ia": "Interlingua (International Auxiliary Language Association)",
    "id": "Indonesian",
    "ie": "Interlingue; Occidental",
    "ig": "Igbo",
    "ii": "Sichuan Yi; Nuosu",
    "ik": "Inupiaq",
    "io": "Ido",
    "is": "Icelandic",
    "it": "Italian",
    "iu": "Inuktitut",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kg": "Kongo",
    "ki": "Kikuyu; Gikuyu",
    "kj": "Kuanyama; Kwanyama",
    "kk": "Kazakh",
    "kl": "Kalaallisut; Greenlandic",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "kr": "Kanuri",
    "ks": "Kashmiri",
    "ku": "Kurdish",
    "kv": "Komi",
    "kw": "Cornish",
    "ky": "Kirghiz; Kyrgyz",
    "la": "Latin",
    "lb": "Luxembourgish; Letzeburgesch",
    "lg": "Ganda",
    "li": "Limburgan; Limburger; Limburgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lu": "Luba-Katanga",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mh": "Marshallese",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "na": "Nauru",
    "nb": "Bokmål, Norwegian; Norwegian Bokmål",
    "nd": "Ndebele, North; North Ndebele",
    "ne": "Nepali",
    "ng": "Ndonga",
    "nl": "Dutch; Flemish",
    "nn": "Norwegian Nynorsk; Nynorsk, Norwegian",
    "no": "Norwegian",
    "nr": "Ndebele, South; South Ndebele",
    "nv": "Navajo; Navaho",
    "ny": "Chichewa; Chewa; Nyanja",
    "oc": "Occitan (post 1500)",
    "oj": "Ojibwa",
    "om": "Oromo",
    "or": "Oriya",
    "os": "Ossetian; Ossetic",
    "pa": "Panjabi; Punjabi",
    "pi": "Pali",
    "pl": "Polish",
    "ps": "Pushto; Pashto",
    "pt": "Portuguese",
    "qu": "Quechua",
    "rm": "Romansh",
    "rn": "Rundi",
    "ro": "Romanian; Moldavian; Moldovan",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "sa": "Sanskrit",
    "sc": "Sardinian",
    "sd": "Sindhi",
    "se": "Northern Sami",
    "sg": "Sango",
    "si": "Sinhala; Sinhalese",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sm": "Samoan",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "ss": "Swati",
    "st": "Sotho, Southern",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "ti": "Tigrinya",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tn": "Tswana",
    "to": "Tonga (Tonga Islands)",
    "tr": "Turkish",
    "ts": "Tsonga",
    "tt": "Tatar",
    "tw": "Twi",
    "ty": "Tahitian",
    "ug": "Uighur; Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "ve": "Venda",
    "vi": "Vietnamese",
    "vo": "Volapük",
    "wa": "Walloon",
    "wo": "Wolof",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "za": "Zhuang; Chuang",
    "zh": "Chinese",
    "zu": "Zulu",
}


def compute_accuracy(results, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    keys_to_keep = {
        "language",
        "lang",
        "accuracy",
        "country",
        "level",
        "category_en",
        "category_original_lang",
        "image_information",
        "image_type",
        "general_category_en",
        "is_multimodal",
    }

    with open(results, "r") as f:
        results = json.load(f)

    # check change of prediction name
    prediction_field = next(
        (key for key in results[0].keys() if key.startswith("prediction_by_")),
        "prediction",
    )
    for sample in results:
        if sample[prediction_field] not in [0, 1, 2, 3]:
            sample["accuracy"] = None
        else:
            sample["accuracy"] = int(
                sample.get(prediction_field) == sample.get("answer")
            )
        sample["lang_code"] = sample.pop("language")
        sample["language"] = LANGUAGES[sample["lang_code"]]
        sample["is_multimodal"] = sample["image"] is not None

    filtered_data = [
        {key: sample[key] for key in keys_to_keep if key in sample}
        for sample in results
    ]

    output_file = os.path.join(output_folder, "full_accuracy.json")
    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=2)
    print(f"Accuracy saved in {output_file}")
    return output_file


def get_results(results, output_dir, filter_multimodal=""):
    data = pd.read_json(results)
    if filter_multimodal == "multimodal":
        data = data[data["is_multimodal"] == True]
    elif filter_multimodal == "text-only":
        data = data[data["is_multimodal"] == False]
    group_columns = ["language", "category_en", "general_category_en"]
    results = []
    grouped = data.groupby(group_columns)
    results = []
    for name, group in grouped:
        none_count = group["accuracy"].isna().sum()
        valid_count = group["accuracy"].notna().sum()
        avg_accuracy = group["accuracy"].mean()
        results.append(
            {
                "language": name[0],
                "category_en": name[1],
                "general_category_en": name[2],
                "average_accuracy": avg_accuracy,
                "valid_samples": valid_count,
                "none_samples": none_count,
            }
        )
    results_df = pd.DataFrame(results)
    output_dir = os.path.join(
        output_dir,
        f"accuracy_results{'_'+filter_multimodal if filter_multimodal else ''}.csv",
    )
    results_df.to_csv(output_dir, index=False)
    print(f"Analysis saved in {output_dir}")
    return results_df

    category = "language"
    summary = (
        results_df.groupby(category)
        .apply(
            lambda x: pd.Series(
                {
                    "weighted_avg_accuracy": (
                        (x["average_accuracy"] * x["valid_samples"]).sum()
                        / x["valid_samples"].sum()
                    ),
                    "error_rate": 1
                    - (
                        (x["average_accuracy"] * x["valid_samples"]).sum()
                        / x["valid_samples"].sum()
                    ),
                    "valid_percentage": (
                        x["valid_samples"].sum()
                        / (x["valid_samples"].sum() + x["none_samples"].sum())
                    )
                    * 100,
                    "total_none_samples": x["none_samples"].sum(),
                    "total_valid_samples": x["valid_samples"].sum(),
                }
            )
        )
        .reset_index()
    )


def perform_complete_evaluation(df_dataset, output_folder):

    perform_accuracy_evaluation(df_dataset, output_folder)
    perform_descriptive_statistics(df_dataset, output_folder)
    perform_plots(df_dataset, output_folder)
    print("not implemented yet: perform_experiments(df_dataset)")


def perform_accuracy_evaluation(df_dataset, output_folder):
    code2lang = LANGUAGES

    output_folder = output_folder + "/results_accuracy"
    os.makedirs(output_folder, exist_ok=True)

    if "language" in df_dataset.columns:
        df_dataset["language"] = df_dataset["language"].map(code2lang)

    model_columns = [
        col for col in df_dataset.columns if col.startswith("prediction_by_")
    ]
    model_names = [col.replace("prediction_by_", "") for col in model_columns]
    df_dataset.rename(columns=dict(zip(model_columns, model_names)), inplace=True)

    # Group by language and calculate accuracies
    group_by_and_score(df_dataset, "language", model_names, output_folder)
    # Group by country and calculate accuracies
    group_by_and_score(df_dataset, "country", model_names, output_folder)
    # Group by level and calculate accuracies
    group_by_and_score(df_dataset, "level", model_names, output_folder)
    # Group by category_en and calculate accuracies
    group_by_and_score(df_dataset, "category_en", model_names, output_folder)
    group_by_and_score(df_dataset, "general_category_en", model_names, output_folder)

    # Group by level and calculate accuracies
    group_by_and_score(df_dataset, "level", model_names, output_folder)

    # Group by category_en and calculate accuracies
    group_by_and_score(df_dataset, "category_en", model_names, output_folder)

    # Calculate accuracies by language and category for each model.
    for model in model_names:
        accuracy_df = df_dataset[model] == df_dataset["answer"]

        accuracies = (
            accuracy_df.groupby([df_dataset["language"], df_dataset["category_en"]])
            .mean()
            .unstack(fill_value=0)
        )
        file_name = model + "_accuracy_language&category.csv"
        output_file = output_folder + "/" + file_name
        accuracies.to_csv(output_file)

    print(f"Accuracy results saved to: {output_file}")


def group_by_and_score(df_dataset, group, model_names, output_folder):

    # Group and calculate accuracy
    accuracy_df = df_dataset[model_names].eq(df_dataset["answer"], axis=0)
    accuracies_by_lang = accuracy_df.groupby(df_dataset[group]).mean()
    overall_accuracies = accuracy_df.mean()
    accuracies_by_lang.loc["Overall"] = overall_accuracies

    # Save
    file_name = "/accuracy_across_" + group + ".csv"
    output_file = output_folder + file_name
    accuracies_by_lang.to_csv(output_file)


def perform_descriptive_statistics(df_dataset, output_folder):
    code2lang = LANGUAGES

    output_folder = output_folder + "/statistics"
    os.makedirs(output_folder, exist_ok=True)

    # Frequency Tables
    categorical_fields = [
        "language",
        "country",
        "level",
        "category_en",
        "image_type",
        "image_information",
    ]  # Manu: excluded 'category_original_lang' because it will be endless.
    for field in categorical_fields:
        if field in df_dataset.columns:
            freq_table = df_dataset[field].value_counts().reset_index()
            freq_table.columns = [field, "counts"]
            freq_table["proportion"] = freq_table["counts"] / freq_table["counts"].sum()
            # Map language codes to names if the column is 'language'
            # if field == 'language':
            #     freq_table['full_lang'] = freq_table[field].map(code2lang)
            freq_table.to_csv(output_folder + f"/{field}_frequency.csv", index=False)

    #  Length Statistics. Manu: do we really need these??
    # text_fields = ['question', 'options']
    # for field in text_fields:
    #     if field in df_dataset.columns:
    #         length_stats = df_dataset[field].dropna().apply(len).describe()
    #         length_stats.to_csv(os.path.join(output_folder, f"{field}_length_statistics.csv"), header=True)

    # Answer distribution
    if "answer" in df_dataset.columns:
        answer_stats = calculate_distribution(df_dataset, "answer")
        answer_stats.columns = ["correct answer counts", "proportion correct answer"]

        model_columns = [
            col for col in df_dataset.columns if col.startswith("prediction_by_")
        ]
        distributions = [
            calculate_distribution(df_dataset, col) for col in model_columns
        ]
        answer_stats = pd.concat([answer_stats] + distributions, axis=1)
        answer_stats.to_csv(output_folder + "/answer_balance.csv", index=True)

    # image_type, image_information, level and category_en distributions per language
    get_distribution_table(df_dataset, "category_en", code2lang, output_folder)
    get_distribution_table(df_dataset, "image_type", code2lang, output_folder)
    get_distribution_table(df_dataset, "level", code2lang, output_folder)
    get_distribution_table(df_dataset, "image_information", code2lang, output_folder)

    print(f"Overall statistics saved to folder: {output_folder}")


def calculate_distribution(df, column_name):
    """Calculate the distribution and proportion of answers in a given column."""
    distribution = df[column_name].value_counts().reset_index()
    distribution.columns = ["answer", f"counts {column_name}"]
    distribution = distribution.set_index("answer")
    distribution[f"proportion {column_name}"] = (
        distribution[f"counts {column_name}"]
        / distribution[f"counts {column_name}"].sum()
    )
    distribution = distribution.round(2).astype(str)
    return distribution


def get_distribution_table(
    df: pd.DataFrame, field: str, code2lang: dict, output_folder: str
):

    # useful for image fields
    df = df[df[field].notna() & (df[field] != "")]

    pivot_table = df.pivot_table(
        index="language", columns=field, aggfunc="size", fill_value=0
    )
    overall_counts = pivot_table.sum()
    pivot_table.loc["Overall"] = overall_counts

    pivot_table.index = pivot_table.index.map(lambda x: code2lang.get(x, x))
    pivot_table.to_csv(output_folder + f"/{field}_per_language.csv", index=True)


def perform_experiments(df_dataset):

    image_blindess_experiment(df_dataset)
    # image_captioning_experiment


def image_blindess_experiment(df_dataset):
    # Just filter data by 'useful' and run accuracy eval
    image_blindness_dataset = df_dataset[df_dataset["image_information"] == "useful"]
    perform_accuracy_evaluation(
        image_blindness_dataset,
        output_folder="eval_results/experiments/image_blidness",
        file_name="image_blidness_results.csv",
    )


def perform_plots(df_dataset, output_folder):
    origin_folder = output_folder
    output_folder = output_folder + "/plots"
    os.makedirs(output_folder, exist_ok=True)

    # Spider graph; model accuracy by lang
    if os.path.exists(f"{origin_folder}/results_accuracy"):
        generate_spidergraph(
            f"{origin_folder}/results_accuracy/accuracy_across_language.csv",
            "language",
            output_folder,
        )
        generate_spidergraph(
            f"{origin_folder}/results_accuracy/accuracy_across_level.csv",
            "level",
            output_folder,
        )
    else:
        print("No accuracy results folder detected... passing to statistics plots.")

    # Multimodality distribution across lang grouped barplot.
    plot_multimodality_distribution(df_dataset, output_folder)

    # Category distribution across lang stacked barplot.
    if os.path.exists(f"{origin_folder}/statistics"):
        plot_stacked_bar(
            f"{origin_folder}/statistics/category_en_per_language.csv",
            "Categories",
            output_folder,
        )
        plot_stacked_bar(
            f"{origin_folder}/statistics/level_per_language.csv",
            "Levels",
            output_folder,
        )
        plot_stacked_bar(
            f"{origin_folder}/statistics/image_type_per_language.csv",
            "Image Types",
            output_folder,
        )
        plot_stacked_bar(
            f"{origin_folder}/statistics/image_type_per_language.csv",
            "Image Types",
            output_folder,
        )
    else:
        print("No statistics results folder detected...")

    print(f"All plots saved to {output_folder}")


def generate_spidergraph(data_path: str, group: str, output_folder: str):
    # Read and prepare data
    df = pd.read_csv(data_path)
    df = df[df[group] != "Overall"]  # Remove overall score

    # Extract values and models
    group_values = df[group].tolist()
    models = [col for col in df.columns if col != group]
    num_vars = len(group_values)

    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Create figure and polar axis
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_rlim(0, 1)
    ax.set_rticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10, color="grey"
    )
    ax.grid(color="grey", linestyle="--", linewidth=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(group_values, fontsize=14, color="black")
    colors = plt.cm.tab10.colors

    # Plot each model's data
    for i, model in enumerate(models):
        values = df[model].tolist()
        values += values[:1]
        color = colors[i % len(colors)]

        ax.plot(
            angles,
            values,
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=model,
        )
        ax.fill(angles, values, color=color, alpha=0.5)

    # Configure legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(models),
        fontsize=14,
        frameon=False,
    )

    # Save and close figure
    plt.tight_layout()
    output_path = f"{output_folder}/accuracy_{group}_spider.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Spider chart of models' accuracy across {group} saved to: {output_path}")


def plot_multimodality_distribution(df: pd.DataFrame, output_folder: str):
    """
    pd DataFrame as input.

    Should change this function to pick values from a csv generated by perform_descriptive_statistics
    """
    # Ensure required columns exist
    if not {"language", "image_png"}.issubset(df.columns):
        raise ValueError(
            "The JSON file must contain 'language' and 'image_png' columns."
        )

    df["language"] = df["language"].apply(lambda code: LANGUAGES.get(code, code))
    df["has_image"] = (
        df["image_png"].notnull().map({True: "Multimodal", False: "Text Only"})
    )

    grouped = df.groupby(["language", "has_image"]).size().reset_index(name="count")

    pivot_df = grouped.pivot(index="language", columns="has_image", values="count")
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = range(len(pivot_df.index))

    # Plot bars for each category
    for i, (category, counts) in enumerate(pivot_df.items()):
        ax.bar(
            [pos + i * bar_width for pos in x], counts, width=bar_width, label=category
        )

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Multimodality distribution per Language", fontsize=14)
    ax.set_xticks([pos + bar_width / 2 for pos in x])
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right", fontsize=10)
    ax.legend(title="Question Multimodality", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    output_path = f"{output_folder}/question_multimodality_dist.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Grouped bar plots of question multimodality saved to: {output_path}")


def plot_stacked_bar(file_path: str, group_name: str, output_folder: str):
    df = pd.read_csv(file_path)

    if "Overall" in df["language"].values:
        df = df[df["language"] != "Overall"]

    exam_subjects = [col for col in df.columns if col != "language"]
    df_pivot = df.set_index("language")[exam_subjects]

    ax = df_pivot.plot(kind="bar", stacked=True, figsize=(10, 6))

    ax.set_xlabel("Language")
    ax.set_ylabel("Count")
    ax.set_title(f"{group_name} distribution by Language")
    plt.legend(title=f"{group_name}", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save
    plt.tight_layout()
    output_path = f"{output_folder}/stacked_bar_{group_name.lower()}PerLang.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Stacked bar plots of {group_name} by language saved to: {output_path}")
