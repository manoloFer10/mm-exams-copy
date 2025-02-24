import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EVALUATION_STYLES = ['complete', 'metrics', 'statistics', 'experiments', 'plotting']

LANGUAGES = {
  "ar": "Arabic",
  "bn": "Bengali",
  "de": "German",
  "en": "English",
  "es": "Spanish",
  "fa": "Persian",
  "fr": "French",
  "hi": "Hindi",
  "hr": "Croatian",
  "hu": "Hungarian",
  "lt": "Lithuanian",
  "ne": "Nepali",
  "nl": "Dutch; Flemish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sr": "Serbian",
  "te": "Telugu",
  "uk": "Ukrainian"
}

def perform_complete_evaluation(df_dataset, output_folder):

    perform_metrics(df_dataset, output_folder)
    perform_descriptive_statistics(df_dataset, output_folder)
    perform_plots(df_dataset, output_folder)
    print('not implemented yet: perform_experiments(df_dataset)')

def perform_metrics(df_dataset, output_folder):
    code2lang = LANGUAGES

    output_folder = output_folder +'/metrics'
    os.makedirs(output_folder, exist_ok=True)

    if 'language' in df_dataset.columns:
        df_dataset['language'] = df_dataset['language'].map(code2lang)

    model_columns = [col for col in df_dataset.columns if col.startswith('prediction_by_')]
    model_names = [col.replace('prediction_by_', '') for col in model_columns]
    df_dataset.rename(columns=dict(zip(model_columns, model_names)), inplace=True)

    # Group by different attributes and compute metrics.
    for group in ['language', 'country', 'level', 'category_en', 'general_category_en']:
        group_by_and_score(df_dataset, group, model_names, output_folder)

    # Calculate accuracies by language and category for each model.
    # for model in model_names:
    #     accuracy_df = df_dataset[model] == df_dataset['answer']

    #     accuracies = (
    #         accuracy_df.groupby([df_dataset['language'], df_dataset['category_en']])
    #         .mean()
    #         .unstack(fill_value=0) 
    #     )
    #     file_name = model + "_accuracy_language&category.csv"
    #     output_file = output_folder +'/' + file_name
    #     accuracies.to_csv(output_file)

    # Answer distribution

    if 'answer' in df_dataset.columns:
        answer_stats = calculate_answer_distribution(df_dataset, 'answer')
        answer_stats.columns = ['correct answer counts', 'proportion correct answer']

        distributions = [calculate_answer_distribution(df_dataset, col) for col in model_names]
        answer_stats = pd.concat([answer_stats] + distributions, axis=1)
        answer_stats.to_csv(output_folder +"/answer_balance.csv", index=True)
    

    print(f"Metrics results saved to: {output_folder}")

    
def group_by_and_score(df_dataset, group, model_names, output_folder):
    
    VALID_VALUES = {0, 1, 2, 3}
    results = {}

    # Group by the specified column.
    for grp, subset in df_dataset.groupby(group):
        metrics = {}
        total = len(subset)
        for model in model_names:
            # Create a boolean mask for valid predictions.
            valid_mask = subset[model].isin(VALID_VALUES)
            valid_count = valid_mask.sum()
            error_count = total - valid_count
            
            # Calculate accuracy only on valid predictions.
            if valid_count > 0:
                correct_count = (subset.loc[valid_mask, model] == subset.loc[valid_mask, 'answer']).sum()
                accuracy = round(correct_count / valid_count, 4)
            else:
                accuracy = np.nan  # or 0 if you prefer

            # Error rate: fraction of predictions that are invalid.
            error_rate = round(error_count / total, 4)

            # Save metrics with descriptive column names.
            metrics[f'{model}_accuracy'] = accuracy
            metrics[f'{model}_error_rate'] = error_rate
            metrics[f'{model}_valid_answers'] = valid_count
        results[grp] = metrics

    # Also compute overall metrics (across the entire dataset).
    overall_metrics = {}
    total_overall = len(df_dataset)
    for model in model_names:
        valid_mask = df_dataset[model].isin(VALID_VALUES)
        valid_count = valid_mask.sum()
        error_count = total_overall - valid_count
        if valid_count > 0:
            correct_count = (df_dataset.loc[valid_mask, model] == df_dataset.loc[valid_mask, 'answer']).sum()
            accuracy = round(correct_count / valid_count, 4)
        else:
            accuracy = np.nan
        error_rate = round(error_count / total_overall, 4)
        overall_metrics[f'{model}_accuracy'] = accuracy
        overall_metrics[f'{model}_error_rate'] = error_rate
        overall_metrics[f'{model}_valid_answers'] = valid_count
    results['Overall'] = overall_metrics

    # Convert the results to a DataFrame and save.
    results_df = pd.DataFrame(results).T  # transpose so rows are groups.
    output_file = os.path.join(output_folder, f"metrics_across_{group}.csv")
    results_df.to_csv(output_file)

def calculate_answer_distribution(df, column_name):
    """Calculate the distribution and proportion of answers in a given column."""
    distribution = df[column_name].value_counts().reset_index()
    distribution.columns = ['answer', f'counts {column_name}']
    distribution = distribution.set_index('answer')
    distribution[f'proportion {column_name}'] = distribution[f'counts {column_name}'] / distribution[f'counts {column_name}'].sum()
    distribution = distribution.round(2).astype(str)
    return distribution

def perform_descriptive_statistics(df_dataset, output_folder):
    code2lang = LANGUAGES

    output_folder = output_folder +'/statistics'
    os.makedirs(output_folder, exist_ok=True)

    if 'language' in df_dataset.columns:
        df_dataset['language'] = df_dataset['language'].map(code2lang)

    # Frequency Tables
    categorical_fields = ['language', 'country', 'level', 'category_en', 'general_category_en', 'image_type', 'image_information'] 
    for field in categorical_fields:
        if field in df_dataset.columns:
            freq_table = df_dataset[field].value_counts().reset_index()
            freq_table.columns = [field, 'counts']
            freq_table['proportion'] = freq_table['counts'] / freq_table['counts'].sum()
             # Map language codes to names if the column is 'language'
            # if field == 'language':
            #     freq_table['full_lang'] = freq_table[field].map(code2lang)
            freq_table.to_csv(output_folder+ f"/{field}_frequency.csv", index=False)


    #  Length Statistics. Manu: do we really need these??
    # text_fields = ['question', 'options']
    # for field in text_fields:
    #     if field in df_dataset.columns:
    #         length_stats = df_dataset[field].dropna().apply(len).describe()
    #         length_stats.to_csv(os.path.join(output_folder, f"{field}_length_statistics.csv"), header=True)

    # image_type, image_information, level, general_category_en and category_en distributions per language
    for field in categorical_fields[1:]:
        get_distribution_table_per_language(df_dataset, field, code2lang, output_folder)

    print(f"Overall statistics saved to folder: {output_folder}")

def get_distribution_table_per_language(df: pd.DataFrame, field: str, code2lang: dict, output_folder: str):

    #useful for image fields
    df = df[df[field].notna() & (df[field] != '')]

    pivot_table = df.pivot_table(
        index='language', 
        columns= field ,  
        aggfunc='size',  
        fill_value=0  
    )
    overall_counts = pivot_table.sum()
    pivot_table.loc['Overall'] = overall_counts

    pivot_table.index = pivot_table.index.map(lambda x: code2lang.get(x, x))
    pivot_table.to_csv(output_folder+ f"/{field}_per_language.csv", index=True)


def perform_experiments(df_dataset):

    image_blindess_experiment(df_dataset)
    # image_captioning_experiment
    

def image_blindess_experiment(df_dataset):
    #Just filter data by 'useful' and run accuracy eval
    image_blindness_dataset = df_dataset[df_dataset['image_information'] == 'useful']
    perform_metrics(image_blindness_dataset, 
                                output_folder='eval_results/experiments/image_blidness',
                                file_name = 'image_blidness_results.csv')
    
def perform_plots(df_dataset, output_folder):
    origin_folder = output_folder
    output_folder = output_folder +'/plots'
    os.makedirs(output_folder, exist_ok=True)

    #Spider graph; model accuracy by lang
    if os.path.exists(f'{origin_folder}/metrics'):
        generate_spidergraph(f'{origin_folder}/metrics/metrics_across_language.csv', 'language', output_folder)
        #generate_spidergraph(f'{origin_folder}/results_accuracy/accuracy_across_level.csv', 'level', output_folder)
    else:
        print('No accuracy results folder detected... passing to statistics plots.')

    #Multimodality distribution across lang grouped barplot.
    plot_multimodality_distribution(df_dataset, output_folder)

    #Category distribution across lang stacked barplot. 
    if os.path.exists(f'{origin_folder}/statistics'):
        plot_stacked_bar(f'{origin_folder}/statistics/category_en_per_language.csv', 'Categories', output_folder)
        plot_stacked_bar(f'{origin_folder}/statistics/level_per_language.csv', 'Levels', output_folder)
        plot_stacked_bar(f'{origin_folder}/statistics/image_type_per_language.csv', 'Image Types', output_folder)
        plot_stacked_bar(f'{origin_folder}/statistics/image_type_per_language.csv', 'Image Types', output_folder)

    else:
        print('No statistics results folder detected...')

    print(f'All plots saved to {output_folder}')



def generate_spidergraph(data_path: str,group: str, output_folder: str):
    # Read data
    df = pd.read_csv(data_path, index_col=0)
    df = df[df.index != 'Overall']
    
    models = [col for col in df.columns if col.endswith('accuracy')]
    model_names = [col.replace('_accuracy', '') for col in models]
    df.rename(columns=dict(zip(models, model_names)), inplace=True)

    group_values = df.index.tolist()
    num_vars = len(group_values)
    
    # Angles
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    
    # Now radial axis goes 0–100
    ax.set_rlim(0, 1)
    ax.set_rticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'],
                       fontsize=10, color='grey')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(group_values, fontsize=14, color='black')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    
    colors = plt.cm.tab10.colors
    for i, model in enumerate(model_names):
        values = df[model].tolist()
        values += values[:1]
        color = colors[i % len(colors)]
        
        ax.plot(angles, values, color=color, linewidth=2,
                marker='o', markersize=4, label=model)
        ax.fill(angles, values, color=color, alpha=0.2)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=3, fontsize=14, frameon=False)
    
    plt.tight_layout()
    output_path = f"{output_folder}/accuracy_{group}_spider.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Spider chart of models' accuracy across {group} saved to: {output_path}")

def plot_multimodality_distribution(df: pd.DataFrame, output_folder: str):
    """
    pd DataFrame as input.

    Should change this function to pick values from a csv generated by perform_descriptive_statistics
    """
    # Ensure required columns exist
    if not {'language', 'image'}.issubset(df.columns):
        raise ValueError("The JSON file must contain 'language' and 'image' columns.")
    
    df['language'] = df['language'].apply(lambda code: LANGUAGES.get(code, code))
    df['has_image'] = df['image'].notnull().map({True: 'Multimodal', False: 'Text Only'})
    
    grouped = df.groupby(['language', 'has_image']).size().reset_index(name='count')
    
    pivot_df = grouped.pivot(index='language', columns='has_image', values='count')
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = range(len(pivot_df.index))
    
    # Plot bars for each category
    for i, (category, counts) in enumerate(pivot_df.items()):
        ax.bar([pos + i * bar_width for pos in x], counts, width=bar_width, label=category)
    
    ax.set_xlabel('Language', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Multimodality distribution per Language', fontsize=14)
    ax.set_xticks([pos + bar_width / 2 for pos in x])
    ax.set_xticklabels(pivot_df.index, rotation=45, ha='right', fontsize=10)
    ax.legend(title='Question Multimodality', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = f"{output_folder}/question_multimodality_dist.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Grouped bar plots of question multimodality saved to: {output_path}")

def plot_stacked_bar(file_path:str, group_name:str , output_folder:str):
    df = pd.read_csv(file_path)
    
    if 'Overall' in df['language'].values:
        df = df[df['language'] != 'Overall']
    
    exam_subjects = [col for col in df.columns if col != 'language']
    df_pivot = df.set_index('language')[exam_subjects]
    
    ax = df_pivot.plot(kind='bar', stacked=True, figsize=(10, 6))
    
    ax.set_xlabel('Language')
    ax.set_ylabel('Count')
    ax.set_title(f'{group_name} distribution by Language')
    plt.legend(title=f'{group_name}', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save
    plt.tight_layout()
    output_path = f"{output_folder}/stacked_bar_{group_name.lower()}PerLang.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Stacked bar plots of {group_name} by language saved to: {output_path}")
