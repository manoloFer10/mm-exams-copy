# ----------------------------------------------------------------------------------------------------------------------------------
# CURRENT MODELS:
#   Cohere Maya, Hay que ver cómo lo agregamos.
#   Gemini 1.5,
#   GPT-4o, LISTO
#   Llama 3.1,
#   Claude 3.5,
#   Qwen2-VL-7B-Instruct LISTO
#   LlaVA- Next
#
# Comment: We should decide which models to include.
# ----------------------------------------------------------------------------------------------------------------------------------

from openai import OpenAI
from transformers import AutoModelForVision2Seq, AutoProcessor
from model_utils import (
    parse_qwen_input,
    parse_openai_input,
    format_answer,
    fetch_few_shot_examples,
    temperature,
    max_tokens,
    SYSTEM_MESSAGE,
    SUPPORTED_MODELS,
)
import torch
from typing import Dict, List
import warnings
from PIL import Image
from tqdm import tqdm


def predict_openAI(
    client,
    json_schema: Dict,
    system_message: List[Dict[str, str]],
    lang: str,
    model: str,
    few_shot_setting: str,
    temperature: int,
    max_tokens: int
):
    """
    Defaults to ZERO-SHOT.
    Returns: The predicted option by the model.
    """

    question_text = json_schema["question"]
    question_image = json_schema["image_png"]
    options_list = json_schema["options"]

    question, parsed_options = parse_openai_input(
        question_text, question_image, options_list
    )

    prompt_chat_dict = {"role": "user", "content": question + parsed_options}

    # Enable few-shot setting
    if few_shot_setting == 'few-shot':
        messages = [system_message, fetch_few_shot_examples(lang), prompt_chat_dict]
    if few_shot_setting == 'zero-shot':
        messages = [system_message, prompt_chat_dict]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    output_text = response.choices[0].message.content.strip()

    return format_answer(output_text)


def predict_qwen(
    qwen,
    processor, 
    json_schema: Dict,
    system_message: List[Dict[str, str]],
    lang: str,
    few_shot_setting: str
):
    """
    ZERO-SHOT
    Returns: The predicted option by the model.
    """

    question = json_schema["question"]
    question_image = json_schema["image_png"]
    options_list = json_schema["options"]

    prompt_chat_dict, image_paths = parse_qwen_input(question, question_image, options_list)

    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    # Enable few-shot setting
    if few_shot_setting == 'few-shot':
        messages = [system_message, fetch_few_shot_examples(lang), prompt_chat_dict]
    if few_shot_setting == 'zero-shot':
        messages = [system_message, prompt_chat_dict]


    inputs = processor(
        text=messages,  # This will still align images with text
        images=images,
        return_tensors="pt",
        padding=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate response
    output_ids = qwen.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return format_answer(response[0])


def run_answer_prediction(dataset, args):
    """
    Returns: JSON object with predictions made by the model.
    """
    model = args.model
    API_KEY = args.api_key
    few_shot_setting = args.setting 
    lang = args.selected_langs[0] # Change later

    results = []

    system_message = [{"role": "system", "content": SYSTEM_MESSAGE}]

    if model not in SUPPORTED_MODELS:
        raise NotImplementedError(
            f"Model {model} not currently implemented for prediction."
        )

    if model == "gpt-4o":
        client = OpenAI(api_key=API_KEY)
        for question_json in tqdm(dataset):
            prediction = predict_openAI(
                client,
                question_json,
                system_message,
                lang, 
                model = 'gpt-4o',
                few_shot_setting = few_shot_setting,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            question_json["prediction_by_" + model] = prediction
            # question_json['prompt_used'] = prompt
            result_metadata = question_json.copy()
            results.append(result_metadata) 

    if model == 'llama':
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=API_KEY
        )

        for question_json in tqdm(dataset):
            prediction = predict_openAI(
                client,
                question_json,
                system_message,
                lang, 
                model = 'llama-3.2-90b-vision-preview',
                few_shot_setting = few_shot_setting,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            question_json["prediction_by_" + model] = prediction
            # question_json['prompt_used'] = prompt
            result_metadata = question_json.copy()
            results.append(result_metadata) 

    if model == "maya":
        raise NotImplementedError(f'Model: {model} not currently implemented for prediction.')

    if model == "qwen":
        warnings.warn("Warning, you are about to load a model locally.")

        model_name= "Qwen/Qwen2-VL-7B-Instruct" # TODO: Set actual Qwen2 model.

        qwen = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        qwen_processor = AutoProcessor.from_pretrained(model_name)

        for question_json in tqdm(dataset):
            prediction = predict_qwen(
                qwen, 
                qwen_processor, 
                question_json, 
                system_message,
                lang,
                few_shot_setting
            )

            question_json["prediction_by_" + model] = prediction
            result_metadata = question_json.copy()
            results.append(result_metadata) 

    # Returns the json object with the field 'prediction'
    return results