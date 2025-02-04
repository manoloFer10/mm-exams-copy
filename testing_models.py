from model_utils import (
    initialize_model,
    query_model,
    generate_prompt,
    fetch_system_message,
    SUPPORTED_MODELS,
    SYSTEM_MESSAGES,
    TEMPERATURE,
    MAX_TOKENS
)
import pandas as pd
import json

def get_keys():
    with open('tokens_mm_exams.json', 'r', encoding = 'utf-8') as f:
        keys = json.load(f)
    return keys

api_key = ''
model_name = 'gemini-2.0-flash-exp'

# Initialize model
model, processor = initialize_model(model_name, None, api_key=api_key)
temperature = TEMPERATURE
max_tokens = MAX_TOKENS

# Load dataset
data = pd.read_json('eval_results\inference_results.json')
dataset = data.sample(n=1)

# Evaluate each question
for _,question in dataset.iterrows():
    lang = question["language"]
    # Generate prompt. Note that only local models will need image_paths separatedly.

    question['image'] = 'Example_JSONSchema2.PNG'

    prompt, image_paths = generate_prompt(
        model_name,
        question,
        lang,
        fetch_system_message(SYSTEM_MESSAGES, lang),
        'zero-shot'
    )
    # Query model
    prediction = query_model(model_name,
                                model, 
                                processor, 
                                prompt, 
                                image_paths, 
                                temperature=temperature,
                                max_tokens=max_tokens)
    
    print(f'Prediction by {model_name}: {prediction}')