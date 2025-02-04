import json
import pandas as pd

def update_fields(data):

    category_mapping = {
        # STEM
        "Mathematics": "STEM > Mathematics",
        "mathematics": "STEM > Mathematics",
        "math": "STEM > Mathematics",
        "Math": "STEM > Mathematics",
        "Maths": "STEM > Mathematics",
        "maths": "STEM > Mathematics",
        "wiskunde": "STEM > Mathematics",  # Dutch for Mathematics

        "Chemistry": "STEM > Chemistry",
        "chemistry": "STEM > Chemistry",

        "Biology": "STEM > Biology",
        "biology": "STEM > Biology",
        "biologie": "STEM > Biology",  # Dutch for Biology

        "Physics": "STEM > Physics",
        "physics": "STEM > Physics",

        "informatic olympiad": "STEM > Informatics (Computer Science/Olympiad)",

        # Health Sciences
        "Medicine": "Health Sciences > Medicine",
        "Nursing": "Health Sciences > Nursing",

        # Humanities & Social Sciences
        "History": "Humanities & Social Sciences > History",
        "Ukrainian History": "Humanities & Social Sciences > History",
        "History and Civics": "Humanities & Social Sciences > History",

        "Geography": "Humanities & Social Sciences > Geography",

        "Philosophy": "Humanities & Social Sciences > Philosophy",

        "Ukrainian Language and Literature": "Humanities & Social Sciences > Ukrainian Language and Literature",

        "Economics": "Humanities & Social Sciences > Social Sciences > Economics",
        "Social Studies 2": "Humanities & Social Sciences > Social Sciences > Social Studies",

        # Cognitive Skills / Aptitude
        "skills": "Cognitive Skills / Aptitude > Skills",
        "vaardig": "Cognitive Skills / Aptitude > Skills",  # Dutch for "skilled/able"
        "reasoning": "Cognitive Skills / Aptitude > Reasoning",
        "mental ability": "Cognitive Skills / Aptitude > Mental Ability",
        "driving licence": "Cognitive Skills / Aptitude > Driving License",

        # Other / Uncategorized
        "stilleestekst": "Other",
        "clear": "Other"
    }

    broad_category_mapping = {
    # STEM
    "STEM > Mathematics": "STEM",
    "STEM > Chemistry": "STEM",
    "STEM > Biology": "STEM",
    "STEM > Physics": "STEM",
    "STEM > Informatics (Computer Science/Olympiad)": "STEM",

    # Health Sciences
    "Health Sciences > Medicine": "Health Sciences",
    "Health Sciences > Nursing": "Health Sciences",

    # Humanities & Social Sciences
    "Humanities & Social Sciences > History": "Humanities & Social Sciences",
    "Humanities & Social Sciences > Geography": "Humanities & Social Sciences",
    "Humanities & Social Sciences > Philosophy": "Humanities & Social Sciences",
    "Humanities & Social Sciences > Ukrainian Language and Literature": "Humanities & Social Sciences",
    "Humanities & Social Sciences > Social Sciences > Economics": "Humanities & Social Sciences",
    "Humanities & Social Sciences > Social Sciences > Social Studies": "Humanities & Social Sciences",

    # Cognitive Skills / Aptitude
    "Cognitive Skills / Aptitude > Reasoning": "Cognitive Skills / Aptitude",
    "Cognitive Skills / Aptitude > Mental Ability": "Cognitive Skills / Aptitude",
    "Cognitive Skills / Aptitude > Skills": "Cognitive Skills / Aptitude",
    "Cognitive Skills / Aptitude > Driving License": "Cognitive Skills / Aptitude",

    # Other / Uncategorized
    "Other / Uncategorized": "Other / Uncategorized"
}

    def update_level(level):
        if level == 'High school exam':
            level = 'High School'
        elif level == 'University entrance exam':
            level = 'University Entrance'
        elif level == 'Natinoal':
            level = 'National'
        return level
    
    def update_category(cat):
        return category_mapping.get(cat)
    def update_category_broad(cat):
        return broad_category_mapping.get(update_category(cat))
    
    data["category_en"] = data["category_en"].map(lambda x: update_category_broad(x))
    data["level"] = data["level"].map(lambda x: update_level(x))

    return data

def downsample(df):
    categories_to_downsample = ['hi', 'te']
    downsample_size = 2500  # Desired sample size per category

    df_downsampled_list = [
        df[(df['language'] == cat) & (df['category_en'] == 'Chemistry')].sample(n=downsample_size, random_state=42)
        for cat in categories_to_downsample
    ]

    df_downsampled = pd.concat(df_downsampled_list)

    df_remaining = df[~df['language'].isin(categories_to_downsample)]

    df_final = pd.concat([df_downsampled, df_remaining])

    df_final = df_final.reset_index(drop=True)

    return df_final

def main():
    # Load the JSON file
    input_file = 'eval_results\inference_results.json'
    output_file = 'eval_results\inference_results_cleaned.json'
    
    df_data = pd.read_json(input_file)
    
    print(len(df_data))
    df_data = downsample(df_data)
    df_data = update_fields(df_data)
    print(len(df_data))

    json_records = df_data.to_json(orient='records')
    json_records = json.loads(json_records)
    
    with open(output_file, 'w') as f:
        json.dump(json_records, f, indent=4)
    
    print(f"Updated JSON has been saved to {output_file}")

if __name__ == "__main__":
    main()