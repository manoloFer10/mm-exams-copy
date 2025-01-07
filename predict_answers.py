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
    parse_qwen_inputs,
    parse_openai_input,
    format_answer,
    temperature,
    max_tokens,
    SYSTEM_MESSAGE,
    SUPPORTED_MODELS,
)
import torch
from typing import Dict, List
import os
import json
import base64
import warnings
from PIL import Image


def predict_gpt4(
    client,
    json_schema: Dict,
    system_message: List[Dict[str, str]],
    model: str = "gpt-4o",
    max_tokens: int = 128,
    temperature: float = 0,
):
    """
    ZERO-SHOT
    Returns: The predicted option by the model.
    """

    question_text = json_schema["question"]
    question_image = json_schema["image_png"]
    options_list = json_schema["options"]

    question, parsed_options = parse_openai_input(
        question, question_image, options_list
    )

    question_text_message = {"role": "user", "content": question + parsed_options}

    messages = [system_message, question_text_message]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    output_text = response.choices[0].message.content.strip()

    return output_text


def predict_qwen(qwen, processor, json_schema, system_message):
    """
    ZERO-SHOT
    Returns: The predicted option by the model.
    """

    question = json_schema["question"]
    question_image = json_schema["image_png"]
    options_list = json_schema["options"]

    prompt_dict, image_paths = parse_qwen_inputs(question, question_image, options_list)

    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    inputs = processor(
        text=prompt_dict,  # This will still align images with text
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

    return response[0]


def run_answer_prediction(model: str, data_path: str, API_KEY: str):
    """
    Returns: JSON object with predictions made by the model.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Invalid path for running model prediction: {data_path}"
        )

    with open(data_path, "r") as f:
        test_questions = json.load(f)

    system_message = [{"role": "system", "content": SYSTEM_MESSAGE}]

    if model not in SUPPORTED_MODELS:
        raise NotImplementedError(
            f"Model {model} not implemented for prediction in this code."
        )

    if model == "gpt-4o":
        client = OpenAI(api_key=API_KEY)
        for question_json in test_questions:
            prediction = predict_gpt4(
                client,
                question_json,
                system_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            question_json["prediction_by_" + model] = prediction

    if model == "maya":
        raise NotImplementedError

    if model == "qwen":
        warnings.warn("Warning, you are about to load a model locally.")

        qwen = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        for question_json in test_questions:
            prediction = predict_qwen(
                qwen, qwen_processor, question_json, system_message
            )
            question_json["prediction_by_" + model] = prediction

    # Returns the json object with the field 'prediction by:'
    return question_json


# Queda hacer la lógica de lectura de los datasets.
