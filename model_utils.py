import base64
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

temperature = 0
max_tokens = 1  # Only output the option chosen.

SUPPORTED_MODELS = ['gpt-4o', 'qwen', 'maya', 'llama']

# !!! System message should be a dictionary with language-codes as keys and system messages in that language as values.
SYSTEM_MESSAGE = "You are given a multiple choice question for answering. You MUST only answer with the correct number of the answer. For example, if you are given the options 1, 2, 3, 4 and option 2 (respectively B) is correct, then you should return the number 2. \n"


def initialize_model(
    model_name: str, model_path: str, api_key: str = None, device: str = "cuda"
):
    """
    Initialize the model and processor/tokenizer based on the model name.
    """
    if model_name == "qwen":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    elif model_name == "pangea":
        # Add Pangea initialization logic
        raise NotImplementedError(f"Model: {model_name} not available yet")
    elif model_name == "molmo":
        # Add Molmo initialization logic
        raise NotImplementedError(f"Model: {model_name} not available yet")
    elif model_name == "gpt-4o":
        # Add gpt initialization logic
        raise NotImplementedError(f"Model: {model_name} not available yet")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, processor


def query_model(
    model_name: str, model, processor, prompts: list, images=None, device: str = "cuda"
):
    """
    Query the model based on the model name.
    """
    if model_name == "qwen":
        return query_qwen(model, processor, prompts, device)
    elif model_name == "pangea":
        # Add pangea querying logic
        pass
    elif model_name == "molmo":
        # Add molmo querying logic
        pass
    elif model_name == "gpt-4o":
        # Add gpt querying logic
        pass
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def query_qwen(model, tokenizer, prompts: list, device="cuda"):
    predictions = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=128)
        predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return predictions


def parse_openai_input(question_text, question_image, options_list):

    def encode_image(image):
        try:
            return base64.b64encode(image).decode("utf-8")
        except Exception as e:
            raise TypeError(f'Image {image} could not be encoded. {e}')

    question = [{"type": "text", "text": question_text}]

    if question_image:
        base64_image = encode_image(question_image)
        question_image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }
        question.append(question_image_message)

    # Parse options. Handle options with images carefully by inclusion in the conversation.
    parsed_options = []
    only_text_option = {"type": "text", "text": "{text}"}
    only_image_option = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,{base64_image}"},
    }

    for i, option in enumerate(options_list):
        option_indicator = f"{i+1})"
        if option.lower().endswith(".png"):
            # Generating the dict format of the conversation if the option is an image
            new_image_option = only_image_option.copy()
            new_text_option = only_text_option.copy()
            formated_text = new_text_option["text"].format(text=option_indicator + "\n")
            new_text_option["text"] = formated_text

            parsed_options.append(new_text_option)
            parsed_options.append(
                new_image_option["image_url"]["url"].format(
                    base64_image=encode_image(option)
                )
            )

        else:
            # Generating the dict format of the conversation if the option is not an image
            new_text_option = only_text_option.copy()
            formated_text = new_text_option["text"].format(
                text=option_indicator + option + "\n"
            )
            new_text_option["text"] = formated_text
            parsed_options.append(new_text_option)

    return question, parsed_options


def parse_qwen_input(question_text, question_image, options_list):
    if question_image:
        question = [
            {
                "type": "text",
                "text": f"Question: {question_text}",
            },
            {"type": "image"},
        ]
    else:
        question = [
            {
                "type": "text",
                "text": f"Question: {question_text}",
            }
        ]

    parsed_options = []
    only_text_option = {"type": "text", "text": "{text}"}
    only_image_option = {"type": "image"}

    images_paths = []
    for i, option in enumerate(options_list):
        option_indicator = f"{i+1})"
        if option.lower().endswith(".png"):  # Checks if it is a png file
            # Generating the dict format of the conversation if the option is an image
            new_image_option = only_image_option.copy()
            new_text_option = only_text_option.copy()
            formated_text = new_text_option["text"].format(text=option_indicator + "\n")
            new_text_option["text"] = formated_text

            # option delimiter "1)", "2)", ...
            parsed_options.append(new_text_option)
            # image for option
            parsed_options.append(new_image_option)

            # Ads the image for output
            images_paths.append(option)

        else:
            # Generating the dict format of the conversation if the option is not an image
            new_text_option = only_text_option.copy()
            formated_text = new_text_option["text"].format(
                text=option_indicator + option + "\n"
            )
            new_text_option["text"] = formated_text
            parsed_options.append(
                new_text_option
            )  # Puts the option text if it isn't an image.

    prompt_text = [question] + parsed_options

    if question_image:
        image_paths = [question_image] + image_paths

    return prompt_text, images_paths


def format_answer(answer: str):
    """
    Returns: A zero-indexed integer corresponding to the answer.
    """
    if not isinstance(answer, str) or len(answer) != 1:
        raise ValueError(
            f"Invalid input: '{answer}'. Expected a single character string."
        )

    if "A" <= answer <= "Z":
        # Convert letter to zero-indexed number
        return ord(answer) - ord("A")
    elif "1" <= answer <= "9":
        # Convert digit to zero-indexed number
        return int(answer) - 1
    else:
        raise ValueError(
            f"Invalid answer: '{answer}'. Must be a letter (A-Z) or a digit (1-9)."
        )


def fetch_few_shot_examples(lang):
    # TODO: write function.
    raise NotImplementedError(
        "The function to fetch few_shot examples is not yet implemented, but should return the few shot examples regarding that language."
    )
