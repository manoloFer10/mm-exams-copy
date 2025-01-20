import tiktoken
import os
import json
from model_utils import SYSTEM_MESSAGE

def count_tokens(text, model="gpt-4o"):

    try:
        encoding = tiktoken.encoding_for_model(model)

        # Encode the text and count the tokens
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    tokens = 0
    image_count = 0
    question_tokens = 0
    length_db = 0
    system_message = SYSTEM_MESSAGE

    # Specify the folder containing JSON files
    folder_path = 'test data'

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Open and read the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:

                question_json = json.load(f)
                length_json = len(question_json)
                length_db += length_json
                for question_metadata in question_json:
                    question = question_metadata['question']
                    options = question_metadata['options']
                    options_text = ''
                    for i,option in enumerate(options):
                        divider = f'{i})'
                        if '.png' in option: 
                            image_count += 1
                            options_text += divider
                        else:
                            options_text += divider + option + '/n'

                    text_for_counting = system_message + '/n' + question + options_text
                    if question_metadata['image_png']: image_count += 1
                    tokens+=count_tokens(text_for_counting)



    print(f"(Estimated) Total number of input tokens for MCQ answering  per {length_db} questions: {tokens}")
    print(f"(Estimated) Total number of input images for MCQ answering  per {length_db} questions: {image_count}")
    print(f"(Estimated) Total number of output tokens for MCQ answering per {length_db} questions: {length_db}")

    print(f'Mean (Estimated) number of input tokens for MCQ answering per 1K questions: {1000*tokens/length_db}')
    print(f'Mean (Estimated) number of input images for MCQ answering per 1K questions: {1000*image_count/length_db}')
    print(f'Mean (Estimated) number of output tokens for MCQ answering per 1K questions: {1000}')
 