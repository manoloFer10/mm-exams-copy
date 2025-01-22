import os
import pandas as pd

EVALUATION_STYLES = ['complete', 'accuracy', 'statistics', 'experiments']

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
  "es": "Spanish; Castilian",
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
  "zu": "Zulu"
}

def perform_complete_evaluation(dataset):

    perform_accuracy_evaluation(dataset)
    perform_descriptive_statistics(dataset)
    perform_experiments(dataset)

def perform_accuracy_evaluation(dataset, output_folder = None, file_name = None):
    code2lang = LANGUAGES
    
    #Prepare data in pandas 
    if isinstance(dataset, dict):
        df_dataset = pd.DataFrame(dataset)
    elif isinstance(dataset, str):
        df_dataset = pd.read_json(dataset)
    else: 
        df_dataset = dataset.to_pandas()

    if 'language' in df_dataset.columns:
        df_dataset['language'] = df_dataset['language'].map(code2lang)

    model_columns = [col for col in df_dataset.columns if col.startswith('prediction_by_')]
    model_names = [col.replace('prediction_by_', '') for col in model_columns]
    df_dataset.rename(columns=dict(zip(model_columns, model_names)), inplace=True)


    # Group by language and calculate accuracies
    accuracy_df = df_dataset[model_names].eq(df_dataset['answer'], axis=0)
    accuracies_by_lang  = accuracy_df.groupby(df_dataset['language']).mean()
    overall_accuracies = accuracy_df.mean()
    accuracies_by_lang.loc['Overall'] = overall_accuracies

    # Save
    if not output_folder:
        output_folder = "eval_results/results_accuracy_all_langs"
    os.makedirs(output_folder, exist_ok=True)
    if not file_name:
        file_name = "accuracy_results.csv"
    output_file = os.path.join(output_folder, file_name)
    accuracies_by_lang.to_csv(output_file)

    print(f"Accuracy results saved to: {output_file}")


def perform_descriptive_statistics(dataset):
    code2lang = LANGUAGES

    #Prepare data in pandas 
    if isinstance(dataset, dict):
        df_dataset = pd.DataFrame(dataset)
    elif isinstance(dataset, str):
        df_dataset = pd.read_json(dataset)
    else: 
        df_dataset = dataset.to_pandas()

    output_folder = "eval_results/statistics"
    os.makedirs(output_folder, exist_ok=True)

    # Frequency Tables
    categorical_fields = ['language_full', 'country', 'level', 'category_en', 'image_type'] # Manu: excluded 'category_original_lang' because it will be endless.
    for field in categorical_fields:
        if field in df_dataset.columns:
            freq_table = df_dataset[field].value_counts().reset_index()
            freq_table.columns = [field, 'counts']
            freq_table['proportion'] = freq_table['counts'] / freq_table['counts'].sum()
            if 'language' in freq_table.columns:
                freq_table['language'] = freq_table['language'].map(code2lang)
            freq_table.to_csv(os.path.join(output_folder, f"{field}_frequency.csv"), index=False)

    #  Length Statistics. Manu: do we really need these??
    # text_fields = ['question', 'options']
    # for field in text_fields:
    #     if field in df_dataset.columns:
    #         length_stats = df_dataset[field].dropna().apply(len).describe()
    #         length_stats.to_csv(os.path.join(output_folder, f"{field}_length_statistics.csv"), header=True)

    # Correct answer distribution
    if 'answer' in df_dataset.columns:
        answer_stats = df_dataset['answer'].value_counts().reset_index()
        answer_stats.columns = ['answer', 'counts']
        answer_stats['proportion'] = answer_stats['counts'] / answer_stats['counts'].sum()
        answer_stats.to_csv(os.path.join(output_folder, "answer_balance.csv"), index=False)

    # Image metadata distribution
    if 'image_information' in df_dataset.columns:
        image_info_stats = df_dataset['image_information'].value_counts().reset_index()
        image_info_stats.columns = ['image_information', 'counts']
        image_info_stats['proportion'] = image_info_stats['counts'] / image_info_stats['counts'].sum()
        image_info_stats.to_csv(os.path.join(output_folder, "image_information_breakdown.csv"), index=False)
    if 'image_type' in df_dataset.columns:
        image_type_stats = df_dataset['image_type'].value_counts().reset_index()
        image_type_stats.columns = ['image_type', 'counts']
        image_type_stats['proportion'] = image_type_stats['counts'] / image_type_stats['counts'].sum()
        image_type_stats.to_csv(os.path.join(output_folder, "image_type_breakdown.csv"), index=False)

    print(f"Overall statistics saved to folder: {output_folder}")

def perform_experiments(dataset):

    image_blindess_experiment(dataset)
    

def image_blindess_experiment(dataset):
    #Just filter data by 'useful' and run accuracy eval
    image_blindness_dataset = dataset[dataset['image_information'] == 'useful']
    perform_accuracy_evaluation(image_blindness_dataset, 
                                output_folder='eval_results/experiments/image_blidness',
                                file_name = 'image_blidness_results.csv')