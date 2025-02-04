import os
import pandas as pd
from datasets import Dataset

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
  "zu": "Zulu"
}

def perform_complete_evaluation(df_dataset):

    perform_accuracy_evaluation(df_dataset, output_folder='eval_results/results_accuracy')
    perform_descriptive_statistics(df_dataset)
    print('not implemented yet: perform_experiments(df_dataset)')

def perform_accuracy_evaluation(df_dataset, output_folder):
    code2lang = LANGUAGES

    if 'language' in df_dataset.columns:
        df_dataset['language'] = df_dataset['language'].map(code2lang)

    model_columns = [col for col in df_dataset.columns if col.startswith('prediction_by_')]
    model_names = [col.replace('prediction_by_', '') for col in model_columns]
    df_dataset.rename(columns=dict(zip(model_columns, model_names)), inplace=True)


    # Group by language and calculate accuracies
    group_by_and_score(df_dataset, 'language', model_names, output_folder)

    # Group by country and calculate accuracies
    group_by_and_score(df_dataset, 'country', model_names, output_folder)

    # Calculate accuracies by language and category for each model.
    for model in model_names:
        accuracy_df = df_dataset[model] == df_dataset['answer']

        accuracies = (
            accuracy_df.groupby([df_dataset['language'], df_dataset['category_en']])
            .mean()
            .unstack(fill_value=0) 
        )
        os.makedirs(output_folder, exist_ok=True)
        file_name = model + "_accuracy_language&category.csv"
        output_file = os.path.join(output_folder, file_name)
        accuracies.to_csv(output_file)

        

    print(f"Accuracy results saved to: {output_file}")

def group_by_and_score(df_dataset, group, model_names, output_folder):
    
    #Group and calculate accuracy
    accuracy_df = df_dataset[model_names].eq(df_dataset['answer'], axis=0)
    accuracies_by_lang  = accuracy_df.groupby(df_dataset[group]).mean()
    overall_accuracies = accuracy_df.mean()
    accuracies_by_lang.loc['Overall'] = overall_accuracies
    
    #Save
    if not output_folder:
        output_folder = "eval_results/results_accuracy"
    os.makedirs(output_folder, exist_ok=True)
    file_name = "accuracy_across_" + group + ".csv"
    output_file = os.path.join(output_folder, file_name)
    accuracies_by_lang.to_csv(output_file)

def perform_descriptive_statistics(df_dataset):
    code2lang = LANGUAGES

    output_folder = "eval_results/statistics"
    os.makedirs(output_folder, exist_ok=True)

    # Frequency Tables
    categorical_fields = ['language', 'country', 'level', 'category_en', 'image_type', 'image_information'] # Manu: excluded 'category_original_lang' because it will be endless.
    for field in categorical_fields:
        if field in df_dataset.columns:
            freq_table = df_dataset[field].value_counts().reset_index()
            freq_table.columns = [field, 'counts']
            freq_table['proportion'] = freq_table['counts'] / freq_table['counts'].sum()
             # Map language codes to names if the column is 'language'
            # if field == 'language':
            #     freq_table['full_lang'] = freq_table[field].map(code2lang)
            freq_table.to_csv(os.path.join(output_folder, f"{field}_frequency.csv"), index=False)

    #  Length Statistics. Manu: do we really need these??
    # text_fields = ['question', 'options']
    # for field in text_fields:
    #     if field in df_dataset.columns:
    #         length_stats = df_dataset[field].dropna().apply(len).describe()
    #         length_stats.to_csv(os.path.join(output_folder, f"{field}_length_statistics.csv"), header=True)

    # Answer distribution
    if 'answer' in df_dataset.columns:
        answer_stats = df_dataset['answer'].value_counts().reset_index()
        answer_stats.columns = ['answer', 'correct answer counts']
        answer_stats = answer_stats.set_index('answer')
        answer_stats['proportion correct answer'] = answer_stats['correct answer counts'] / answer_stats['correct answer counts'].sum()

        model_columns = [col for col in df_dataset.columns if col.startswith('prediction_by_')]
        for col in model_columns:
            model_answer_distribution = df_dataset[col].value_counts().reset_index()
            model_answer_distribution.columns = ['answer', 'counts ' + col]
            model_answer_distribution = model_answer_distribution.set_index('answer')
            model_answer_distribution['proportion ' + col] = model_answer_distribution['counts ' + col] / model_answer_distribution['counts ' + col].sum()
            
            answer_stats = pd.merge(answer_stats, 
                                    model_answer_distribution, 
                                    left_index=True, 
                                    right_index=True,  
                                    how='outer')

        
        answer_stats.to_csv(os.path.join(output_folder, "answer_balance.csv"), index=True)


    # image_type, image_information and category_en distributions per language
    get_distribution_table(df_dataset, 'category_en', code2lang, output_folder)
    get_distribution_table(df_dataset, 'image_type', code2lang, output_folder)
    get_distribution_table(df_dataset, 'image_information', code2lang, output_folder)

    print(f"Overall statistics saved to folder: {output_folder}")

def get_distribution_table(df: pd.DataFrame, field: str, code2lang: dict, output_folder: str):

    #useful for image fields
    df = df[df[field].notna() & (df[field] != '')]

    pivot_table = df.pivot_table(
        index='language', 
        columns= field ,  
        aggfunc='size',  
        fill_value=0  
    )

    pivot_table.index = pivot_table.index.map(lambda x: code2lang.get(x, x))
    pivot_table.to_csv(os.path.join(output_folder, f"{field}_per_language.csv"), index=True)


def perform_experiments(df_dataset):

    image_blindess_experiment(df_dataset)
    # image_captioning_experiment
    

def image_blindess_experiment(df_dataset):
    #Just filter data by 'useful' and run accuracy eval
    image_blindness_dataset = df_dataset[df_dataset['image_information'] == 'useful']
    perform_accuracy_evaluation(image_blindness_dataset, 
                                output_folder='eval_results/experiments/image_blidness',
                                file_name = 'image_blidness_results.csv')
    
def perform_plots():
    # TODO
    results_folder = 'eval_results'
    accuracy_folder = results_folder + '/results_accuracy'
    statistics_folder = results_folder +'/statistics'
    output_dir = 'eval_results/plots'

    plottable_files_stats =[] # should we somehow plot accuracy results? maybe somekind of spider graph among languages for each model.

    if not os.path.exists(accuracy_folder): FileNotFoundError(f"The directory '{accuracy_folder}' does not exist.")
    if not os.path.exists(statistics_folder): FileNotFoundError(f"The directory '{statistics_folder}' does not exist.")

    for result_data in plottable_files_stats:
        element_path = os.path.join(statistics_folder, result_data)
        plot_data(element_path, output_dir)

def plot_data(path: str, output_dir: str):
    #TODO: write a function that given a path, creates and saves in output_dir its plotted data.
    raise NotImplementedError('function plot_data not implemented yet.')