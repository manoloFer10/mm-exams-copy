import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

EVALUATION_STYLES = ['complete', 'metrics', 'statistics', 'plotting']

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

RESOURCE_LEVEL_MAP = {
    "Portuguese": "High",
    "Serbian": "High",
    "Persian": "High",
    "Hindi": "High",
    "Russian": "High",
    "English": "High",
    "Spanish": "High",
    "Hungarian": "High",
    "Dutch; Flemish": "High",
    "French": "High",
    "German": "High",
    "Arabic": "High",
    "Croatian": "High",
    "Ukrainian": "Mid/Low",
    "Bengali": "Mid/Low",
    "Lithuanian": "Mid/Low",
    "Telugu": "Mid/Low",
    "Nepali": "Mid/Low"
}

ENGLISH_MAP = {
    "Arabic": "Non-English",
    "Bengali": "Non-English",
    "German": "Non-English",
    "English": "English",
    "Spanish": "Non-English",
    "Persian": "Non-English",
    "French": "Non-English",
    "Hindi": "Non-English",
    "Croatian": "Non-English",
    "Hungarian": "Non-English",
    "Lithuanian": "Non-English",
    "Nepali": "Non-English",
    "Dutch; Flemish": "Non-English",
    "Portuguese": "Non-English",
    "Russian": "Non-English",
    "Serbian": "Non-English",
    "Telugu": "Non-English",
    "Ukrainian": "Non-English"
}

LATIN_SCRIPT_MAP = {
    "Arabic": "Non-Latin",
    "Bengali": "Non-Latin",
    "German": "Latin",
    "English": "Latin",
    "Spanish": "Latin",
    "Persian": "Non-Latin",
    "French": "Latin",
    "Hindi": "Non-Latin",
    "Croatian": "Latin",
    "Hungarian": "Latin",
    "Lithuanian": "Latin",
    "Nepali": "Non-Latin",
    "Dutch; Flemish": "Latin",
    "Portuguese": "Latin",
    "Russian": "Non-Latin",
    "Serbian": "Non-Latin", 
    "Telugu": "Non-Latin",
    "Ukrainian": "Non-Latin"
}

CLEAN_NAMES = {
    'gemini-1.5-pro': 'Gemini 1.5 Pro',
    'claude-3-5-sonnet-latest': 'Claude 3.5 Sonnet',
    'gpt-4o': 'GPT-4o',
    'molmo': 'Molmo-7B-D',
    'pangea': 'Pangea-7B',
    'qwen2.5-7b': 'Qwen2.5-VL-7B',
    'aya': 'Aya-Vision-8B',
    'qwen72b': 'Qwen2.5-VL-72B',
    'qwen3b': 'Qwen2.5-VL-3B',
    'qwen32': 'Qwen2.5-VL-32B',
    'aya-32b': 'Aya-Vision-32B'
}

MODEL_TYPE = {
    'gemini-1.5-pro': 'closed',
    'claude-3-5-sonnet-latest': 'closed',
    'gpt-4o': 'closed',
    'molmo': 'open',
    'pangea': 'open',
    'qwen2.5-7b': 'open',
    'aya': 'open',
    'qwen72b': 'open',
    'qwen3b': 'open',
    'aya-32b': 'open'
}

def perform_complete_evaluation(df_dataset, output_folder):

    perform_metrics(df_dataset, output_folder)
    perform_descriptive_statistics(df_dataset, output_folder)
    perform_plots(df_dataset, output_folder)

def perform_metrics(df_dataset, output_folder):
    code2lang = LANGUAGES

    output_folder = output_folder +'/metrics'
    os.makedirs(output_folder, exist_ok=True)

    if 'language' in df_dataset.columns:
        df_dataset['language'] = df_dataset['language'].map(code2lang)
        df_dataset['script'] = df_dataset['language'].map(LATIN_SCRIPT_MAP)
        df_dataset['is_english'] = df_dataset['language'].map(ENGLISH_MAP)
        df_dataset['resources'] = df_dataset['language'].map(RESOURCE_LEVEL_MAP)

    model_columns = [col for col in df_dataset.columns if col.startswith('prediction_by_')]
    for col in model_columns:
        df_dataset[col] = df_dataset[col].apply(lambda x: int(x) if x in [0,1,2,3,'0', '1', '2', '3'] else x)

    model_names = [col.replace('prediction_by_', '') for col in model_columns]
    df_dataset.rename(columns=dict(zip(model_columns, model_names)), inplace=True)

    # Group by different attributes and compute metrics.
    attributes = [
        'language', 
        'country', 
        'level', 
        'category_en', 
        'general_category_en', 
        'image_type', 
        'script', 
        'is_english', 
        'resources',
        ]
    
    for group in attributes:
        group_by_and_score(df_dataset, group, model_names, output_folder)


    # Answer distribution
    if 'answer' in df_dataset.columns:
        answer_stats = calculate_answer_distribution(df_dataset, 'answer')
        answer_stats.columns = ['correct answer counts', 'proportion correct answer']

        distributions = [calculate_answer_distribution(df_dataset, col) for col in model_names]
        answer_stats = pd.concat([answer_stats] + distributions, axis=1)
        answers_folder = os.path.join(output_folder, 'answer_distribution')
        os.makedirs(os.path.dirname(answers_folder), exist_ok=True)
        general_file_path = os.path.join(answers_folder, 'answer_balance.csv')
        os.makedirs(os.path.dirname(general_file_path), exist_ok=True)
        answer_stats.to_csv(general_file_path, index=True)

        for lang in df_dataset['language'].unique():
            filtered_df = df_dataset[df_dataset['language'] == lang]
            answer_stats = calculate_answer_distribution(filtered_df, 'answer')
            distributions = [calculate_answer_distribution(filtered_df, col) for col in model_names]
            lang_answer_stats = pd.concat([answer_stats] + distributions, axis=1)
            os.makedirs(os.path.dirname(answers_folder), exist_ok=True)
            lang_file_path = os.path.join(answers_folder, f'{lang}_answer_balance.csv')
            os.makedirs(os.path.dirname(lang_file_path), exist_ok=True)
            lang_answer_stats.to_csv(lang_file_path, index=True)
    

    print(f"Metrics results saved to: {output_folder}")

    
def group_by_and_score(df_dataset, group, model_names, output_folder):
    VALID_VALUES = {0, 1, 2, 3}
    results = {}

    # Group by the specified column.
    for grp, subset in df_dataset.groupby(group):
        metrics = {}
        total = len(subset)
        for model in model_names:
            valid_mask = subset[model].isin(VALID_VALUES)
            valid_count = valid_mask.sum()
            error_count = total - valid_count

            correct_count = (subset.loc[valid_mask, model] == subset.loc[valid_mask, 'answer']).sum()

            answer_accuracy = round(correct_count * 100 / valid_count, 1)
            error_rate = round(error_count * 100 / total, 1)
            total_accuracy = round(correct_count * 100 / total, 1)

            # Save metrics
            metrics[f'{model}_total_accuracy'] = total_accuracy
            metrics[f'{model}_answer_accuracy'] = answer_accuracy
            metrics[f'{model}_error_rate'] = error_rate
            metrics[f'{model}_error_count'] = error_count

        results[grp] = metrics

    # Overall (across the entire dataset))
    overall_metrics = {}
    total_overall = len(df_dataset)
    for model in model_names:
        valid_mask = df_dataset[model].isin(VALID_VALUES)
        valid_count = valid_mask.sum()
        error_count = total_overall - valid_count
        if valid_count > 0:
            correct_count = (df_dataset.loc[valid_mask, model] == df_dataset.loc[valid_mask, 'answer']).sum()
            answer_accuracy = round(correct_count * 100 / valid_count, 1)
        else:
            answer_accuracy = np.nan

        error_rate = round(error_count * 100 / total_overall, 1)
        total_accuracy = round(correct_count * 100 / total_overall, 1)

        overall_metrics[f'{model}_total_accuracy'] = total_accuracy
        overall_metrics[f'{model}_answer_accuracy'] = answer_accuracy
        overall_metrics[f'{model}_error_rate'] = error_rate
        overall_metrics[f'{model}_error_count'] = error_count

    results['Overall'] = overall_metrics

    results_df = pd.DataFrame(results).T

    total_acc_df = results_df[[col for col in results_df.columns if col.endswith('total_accuracy')]]
    answer_acc_df = results_df[[col for col in results_df.columns if col.endswith('answer_accuracy')]]
    error_rate_df = results_df[[col for col in results_df.columns if col.endswith('error_rate') or col.endswith('error_count')]]
    
    total_acc_file = os.path.join(output_folder, f"{group}/total_accuracy.csv")
    answer_acc_file = os.path.join(output_folder, f"{group}/answer_accuracy.csv")
    error_rate_file = os.path.join(output_folder, f"{group}/error_rate.csv")
    all_results_file = os.path.join(output_folder, f"{group}/all_results.csv")

    os.makedirs(os.path.dirname(total_acc_file), exist_ok=True)
    os.makedirs(os.path.dirname(answer_acc_file), exist_ok=True)
    os.makedirs(os.path.dirname(error_rate_file), exist_ok=True)
    os.makedirs(os.path.dirname(all_results_file), exist_ok=True)

    # Save each DataFrame.
    total_acc_df.to_csv(total_acc_file, index=True)
    answer_acc_df.to_csv(answer_acc_file, index=True)
    error_rate_df.to_csv(error_rate_file, index=True)
    results_df.to_csv(all_results_file, index=True)

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
            freq_table.to_csv(output_folder+ f"/{field}_frequency.csv", index=False)


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


    
def perform_plots(df_dataset, output_folder):
    origin_folder = output_folder
    output_folder = output_folder +'/plots'
    os.makedirs(output_folder, exist_ok=True)

    if os.path.exists(f'{origin_folder}/metrics'):
        # generate_barplot(f'{origin_folder}/metrics/language/answer_accuracy.csv', 'Language', output_folder)
        # generate_barplot(f'{origin_folder}/metrics/level/answer_accuracy.csv', 'Exam Level', output_folder)
        # generate_barplot(f'{origin_folder}/metrics/image_type/answer_accuracy.csv', 'Image Type', output_folder)
        # generate_barplot(f'{origin_folder}/metrics/category_en/answer_accuracy.csv', 'Subject', output_folder)
        # generate_model_barplots(f'{origin_folder}/metrics/language/answer_accuracy.csv', 'language', output_folder)
        # generate_group_barplots(f'{origin_folder}/metrics/language/answer_accuracy.csv', 'language', output_folder)
        # scatter_plot_accuracies(f'{origin_folder}/metrics/script/answer_accuracy.csv', 'Latin vs Non-Latin (script) Performance', output_folder)
        error_heatmap(f'{origin_folder}/metrics/language/error_rate.csv', output_folder)

    else:
        print('No metrics results folder detected... passing to statistics plots.')

    print(f'All plots saved to {output_folder}')


def generate_barplot(data_path: str, group: str, output_folder: str):
    '''
    Generates two barplots in a fig, one on top of another splitting by open and closed models.
    '''
     # Read data
    df = pd.read_csv(data_path, index_col=0)
    df = df[df.index != 'Overall']

    # Identify model columns and clean their names
    models = [col for col in df.columns if col.endswith('_answer_accuracy')]
    original_model_ids = [col.replace('_answer_accuracy', '') for col in models]
    model_names = [CLEAN_NAMES[model_id] for model_id in original_model_ids]
    df.rename(columns=dict(zip(models, model_names)), inplace=True)

    # Split models into closed and open
    closed_ids = [model_id for model_id in original_model_ids if MODEL_TYPE.get(model_id) == 'closed']
    open_ids = [model_id for model_id in original_model_ids if MODEL_TYPE.get(model_id) == 'open']
    closed_models = [CLEAN_NAMES[model_id] for model_id in closed_ids]
    open_models = [CLEAN_NAMES[model_id] for model_id in open_ids]

    # Prepare data for plotting
    df_reset = df.reset_index().rename(columns={'index': f'{group}'})
    plot_df_closed = df_reset.melt(id_vars=f'{group}', value_vars=closed_models, 
                                  var_name='Model', value_name='Accuracy')
    plot_df_open = df_reset.melt(id_vars=f'{group}', value_vars=open_models, 
                                var_name='Model', value_name='Accuracy')

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    closed_palette = {
        'Gemini 1.5 Pro': '#1F77B4',  # Blue
        'Claude 3.5 Sonnet': '#FF7F0E',  # Orange
        'GPT-4o': '#2CA02C',  # Green
    }

    open_palette = {
        'Molmo-7B-D': '#D62728',  # Red
        'Pangea-7B': '#9467BD',  # Purple
        'Qwen2.5-VL-7B': '#8C564B',  # Brown
        'Aya-Vision-8B': '#17BECF',
        'Qwen2.5-VL-72B': '#e377c2',  # Pink
        'Qwen2.5-VL-3B':  '#2ca02c',  # Green
        'Aya-Vision-32B': '#ff7f0e',  # Orange
    }

    # Plot closed models
    if closed_models:
        sns.barplot(x=f'{group}', y='Accuracy', hue='Model', data=plot_df_closed, 
                    palette=closed_palette.values(), ax=ax1, hue_order=closed_models)
        ax1.set_title(f"Closed Models", pad=20)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        ax1.legend(loc='upper left', title='Model')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_xticklabels(ax1.get_xticklabels(), ha='right')
    else:
        ax1.remove()  # Remove subplot if no closed models
    
    # Plot open models
    if open_models:
        sns.barplot(x=f'{group}', y='Accuracy', hue='Model', data=plot_df_open, 
                    palette=open_palette.values(), ax=ax2, hue_order=open_models)
        ax2.set_title(f"Light-weight Open Models", pad=20)
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', title='Model')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_xticklabels(ax2.get_xticklabels(), ha='right')
    else:
        ax2.remove()  # Remove subplot if no open models

    plt.xlabel(f'{group}')  # Common x-axis label
    plt.tight_layout()

    # Save output
    output_path = f"{output_folder}/accuracy_{group}_bar_split.svg"
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    output_path = f"{output_folder}/accuracy_{group}_bar_split.png"
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.close()
    
    print(f"Bar chart saved to: {output_path}")


def generate_model_barplots(data_path: str, group: str, output_folder: str):
    # Read and prepare data
    df = pd.read_csv(data_path, index_col=0)
    df = df[df.index != 'Overall']
    
    models = [col for col in df.columns if col.endswith('_answer_accuracy')]
    model_names = [col.replace('_answer_accuracy', '') for col in models]
    model_names = [CLEAN_NAMES[col] for col in model_names]
    df.rename(columns=dict(zip(models, model_names)), inplace=True)
    
    categories = df.index.tolist()
    num_models = len(model_names)
    num_categories = len(categories)
    
    # Create color palette
    colors = plt.cm.tab20.colors  # Using extended color palette
    if num_categories > len(colors):
        colors = plt.cm.gist_ncar(np.linspace(0, 1, num_categories))
    
    # Create subplot grid
    cols = 3
    rows = int(np.ceil(num_models / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(18, 4*rows))
    fig.suptitle(f'Model Accuracies Across {group}', y=1.02, fontsize=16)
    
    # Handle axes array
    axs = axs.flatten() if num_models > 1 else [axs]
    
    for idx, (model, ax) in enumerate(zip(model_names, axs)):
        values = df[model].values
        
        # Create bar plot with explicit x-ticks
        x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, values, color=colors[:num_categories])
        
        # Formatting
        ax.set_title(model, fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_xticks(x_pos)  # Set x-ticks first
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        
        max_acc = max([bar.get_height() for bar in bars])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height == max_acc:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=7, fontweight= 'bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=7)
        
        # Remove empty subplots
        if idx == num_models - 1:
            for ax in axs[idx+1:]:
                ax.remove()
    
    # Create common legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=colors[i]) 
                     for i in range(num_categories)]
    fig.legend(legend_handles, categories,
              loc='upper center',
              ncol=min(6, num_categories),
              bbox_to_anchor=(0.5, 0.98),
              fontsize=10)
    
    plt.tight_layout(pad=3.0)
    output_path = f"{output_folder}/accuracy_{group}_individual_bars.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Individual model bar plots for {group} saved to: {output_path}")


def generate_group_barplots(data_path: str, group: str, output_folder: str):
    # Read and prepare data
    df = pd.read_csv(data_path, index_col=0)
    df = df[df.index != 'Overall']
    
    models = [col for col in df.columns if col.endswith('_answer_accuracy')]
    model_names = [col.replace('_answer_accuracy', '') for col in models]
    model_names = [CLEAN_NAMES[col] for col in model_names]
    df.rename(columns=dict(zip(models, model_names)), inplace=True)
    
    categories = df.index.tolist()
    num_categories = len(categories)
    
    # Create subplot grid
    cols = 3
    rows = int(np.ceil(num_categories / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=(18, 4*rows))
    
    # Dynamic title positioning based on number of rows
    title_y = 1.02 - 0.02 * rows  # Adjust this factor based on your needs
    fig.suptitle(f'Model Accuracies by {group}', y=title_y, fontsize=16)
    
    axs = axs.flatten()
    colors = plt.cm.tab10.colors
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(model_names)}
    
    for idx, (category, ax) in enumerate(zip(categories, axs)):
        values = df.loc[category].values
        
        bars = ax.bar(model_names, values, color=[model_colors[m] for m in model_names])
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        
        ax.set_title(category, fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        
        max_acc = max([bar.get_height() for bar in bars])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height == max_acc:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=7, fontweight= 'bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=7)
    
    for ax in axs[num_categories:]:
        ax.remove()
    
    # Create common legend
    handles = [plt.Rectangle((0,0),1,1, color=model_colors[m]) for m in model_names]
    fig.legend(handles, model_names, 
             loc='upper center', 
             ncol=min(4, len(model_names)), 
             bbox_to_anchor=(0.5, title_y-0.02),  # Adjust legend position
             fontsize=10,
             title='Models')
    
    plt.tight_layout(pad=3.0)
    # Dynamic adjustment of top margin
    plt.subplots_adjust(top=0.9 - 0.02*rows)  # Adjusts based on number of rows
    
    output_path = f"{output_folder}/accuracy_{group}_group_bars.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Group-wise bar plots for {group} saved to: {output_path}")


def scatter_plot_accuracies(csv_file, title, output_folder):
    
    df = pd.read_csv(csv_file, index_col=0)
    
    # Ensure there are at least two rows to compare
    if len(df) < 2:
        raise ValueError("CSV file must contain at least two rows of accuracy values.")
    
    # Extract the first two rows for comparison
    first_row = df.iloc[0]
    second_row = df.iloc[1]
    
    # Map the raw model names to clean names
    models_cleaned = [CLEAN_NAMES.get(model.replace('_answer_accuracy', ''), model)
                      for model in first_row.index]
    
    # Combine data for plotting
    data = pd.DataFrame({
        'Model': models_cleaned,
        'FirstRow': first_row.values,
        'SecondRow': second_row.values
    })
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data['SecondRow'], data['FirstRow'], s=100, color='tab:blue')
    
    # Annotate each point with model names
    for _, row in data.iterrows():
        plt.annotate(row['Model'], (row['SecondRow'], row['FirstRow']),
                     textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9)
    
    plt.plot([0, 100], [0, 100], color='grey', linestyle='--', linewidth=1)

    plt.xlabel(f'{df.index[1]}')
    plt.ylabel(f'{df.index[0]}')
    plt.title(title)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure as PNG and SVG
    plt.savefig(f"{output_folder}/{title}.png", format="png", dpi=300)
    plt.savefig(f"{output_folder}/{title}.svg", format="svg")
    
    plt.show()

def error_heatmap(csv_file, output_folder):
    df = pd.read_csv(csv_file, index_col=0)
    df = df[df.index != 'Overall']
    
    models = [col for col in df.columns if col.endswith('_error_count')]
    model_names = [col.replace('_error_count', '') for col in models]
    model_names = [CLEAN_NAMES[col] for col in model_names]
    df.rename(columns=dict(zip(models, model_names)), inplace=True)
    df = df.T
    df = df.reindex(model_names)

    plt.figure(figsize=(12, 4))  
    
    sns.heatmap(df, annot=True, fmt=".0f", cmap="viridis_r", linewidths=0.5, 
                cbar_kws={'label': 'Unanswered Questions'}, 
                annot_kws={"size": 8})  

    
    plt.yticks(rotation=0, fontsize=8)  
    plt.xticks(fontsize=9)  

    plt.title("Distribution of Unanswered Questions Across Languages and Models", fontsize=12)
    plt.xlabel("Language", fontsize=10)
    plt.ylabel("Model", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/error_heatmap.png", format="png", dpi=300)
    plt.savefig(f"{output_folder}/error_heatmap.svg", format="svg")
    plt.show()