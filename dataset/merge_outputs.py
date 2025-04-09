import pandas as pd
from functools import reduce
from datasets import load_dataset
from model_utils import SUPPORTED_MODELS


unwanted_columns = ['prompt_used'] # just for storage means.

# Adjust accordingly.
json_files = [ f"outputs/zero-shot/model_{model}" for model in SUPPORTED_MODELS]

def normalize_id_components(df):
    df['question'] = df['question'].str.strip().astype(str)
    df['original_question_num'] = df['original_question_num'].astype(str).str.strip()
    df['file_name'] = df['file_name'].str.strip().astype(str)
    df['image_png'] = df['image_png'].str.strip().astype(str)
    return df


def main():
    dfs = []
    for idx, file in enumerate(json_files):
        print(f'processing: {file}')
        df = pd.read_json(file)
        print(f'len: {len(df)}')
        
        df = df.drop(columns=[col for col in unwanted_columns if col in df.columns])
        df = normalize_id_components(df)
        df['unique_id'] = df['question'] + '_' + df['original_question_num'] + '_' + df['file_name'] + '_' + df['image_png']
        
        if idx == 0:
            # For the first file, keep all metadata columns.
            df_keep = df.copy()
        else:
            # For subsequent files, keep only unique_id and model-specific predictions.
            model_pred_cols = [col for col in df.columns if col.startswith("prediction_by_") or col.startswith("reasoning_by_")]
            df_keep = df[['unique_id'] + model_pred_cols]
        
        dfs.append(df_keep)

    # Merge all DataFrames on the composite key 'unique_id' iteratively.
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='unique_id', how='outer')

    print("Merged df length:", len(merged_df))

    # Save the cleaned merged DataFrame to a new JSON file.
    merged_df.to_json("(zero-shot)new_prompt_all.json", orient='records', indent=2)

if __name__ == '__main__':
    main()