import base64
from PIL import Image
import io

keywords = {
    "en": {"question": "Question", "options": "Options", "answer": "Answer"},
    "es": {"question": "Pregunta", "options": "Opciones", "answer": "Respuesta"},
    "hi": {"question": "प्रश्न", "options": "विकल्प", "answer": "उत्तर"},
    "hu": {"question": "Kérdés", "options": "Lehetőségek", "answer": "Válasz"},
    "hr": {"question": "Pitanje", "options": "Opcije", "answer": "Odgovor"},
    "uk": {"question": "Питання", "options": "Варіанти", "answer": "Відповідь"},
    "pt": {"question": "Pergunta", "options": "Opções", "answer": "Resposta"},
    "bn": {"question": "প্রশ্ন", "options": "বিকল্প", "answer": "উত্তর"},
    "te": {"question": "ప్రశ్న", "options": "ఎంపికలు", "answer": "సమాధానం"},
    "ne": {"question": "प्रश्न", "options": "विकल्पहरू", "answer": "उत्तर"},
    "sr": {"question": "Pitanje", "options": "Opcije", "answer": "Odgovor"},
    "nl": {"question": "Vraag", "options": "Opties", "answer": "Antwoord"},
    "ar": {"question": "السؤال", "options": "الخيارات", "answer": "الإجابة"},
    "ru": {"question": "Вопрос", "options": "Варианты", "answer": "Ответ"},
    "fr": {"question": "Question", "options": "Options", "answer": "Réponse"},
    "fa": {"question": "سؤال", "options": "گزینه‌ها", "answer": "پاسخ"},
    "de": {"question": "Frage", "options": "Optionen", "answer": "Antwort"},
    "lt": {"question": "Klausimas", "options": "Pasirinkimai", "answer": "Atsakymas"},
}

SYS_MESSAGE = 'You are a helpful assistant who answers multiple-choice questions. For each question, output your final answer in JSON format with the following structure:\n\n{"choice": "The correct option (e.g., A, B, C, or D)"}\n\ONLY output this format exactly. Do not include any additional text or explanations outside the JSON structure.'
INSTRUCTION = "Output your choice in the specified JSON format."


# Molmo
def create_molmo_prompt_vllm(question, method, few_shot_samples):
    lang = question["language"]
    lang_keyword = keywords[lang]
    prompt = [SYS_MESSAGE]
    prompt.append(INSTRUCTION)
    if question["image"] is not None:
        images = [question["image"]]
    else:
        images = None
    prompt.append(
        f"\n{lang_keyword['question']}: {question['question'].replace('<image>', '')} \n{lang_keyword['options']}: \n"
    )
    for t, option in enumerate(question["options"]):
        index = f"{chr(65+t)}. "
        prompt.append(f"{index}) {option.replace('<image>', '')}\n")
    prompt.append(f"\nANSWER:")
    prompt = "".join(prompt)

    prompt = f"<|im_start|>user <image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prompt, images


# Pangea
def create_pangea_prompt(question, method, few_shot_samples):
    lang = question["language"]
    lang_keyword = keywords[lang]
    prompt = [f"<|im_start|>system\n{SYS_MESSAGE}<|im_end|>\n<|im_start|>user\n"]
    if question["image"] is not None:
        prompt.append("<image>\n")
        images = question["image"]
    else:
        images = None

    prompt.append(f"\n{INSTRUCTION}\n")
    prompt.append(
        f"\n{lang_keyword['question']}: {question['question'].replace('<image>', '')} \n{lang_keyword['options']}: \n"
    )
    for t, option in enumerate(question["options"]):
        index = f"{chr(65+t)}. "
        prompt.append(f"{index}) {option.replace('<image>', '')}\n")
    # prompt.append(f"\n{lang_keyword['answer']}:")
    prompt.append("<|im_end|>\n<|im_start|>assistant\n")
    message = "".join(prompt)
    return message, images

# Qwen
def create_qwen_prompt_vllm(question, method, few_shot_samples, experiment):
    # Determine the placeholder for images
    lang = question["language"]
    prompt = ""
    # Add the main question and options
    prompt += (
        f"\n{INSTRUCTION}\n"
        f"\n{keywords[lang]['question']}: {question['question']}\n"
        f"{keywords[lang]['options']}:\n"
    )
    for t, option in enumerate(question["options"]):
        prompt += f"{chr(65 + t)}. {option}\n"
    prompt += f"\nANSWER:"

    # Construct the final message
    if question["image"] is not None:
        images = [question["image"]]
        if experiment == 'captioned':
            caption = question['image_caption']
            ocr = question['image_ocr']
            message = (
            f"<|im_start|>system\n{SYS_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
            f"Caption: {caption} \n OCR: {ocr} \n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        else:
            message = (
                f"<|im_start|>system\n{SYS_MESSAGE}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
                f"{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
    else:
        message = (
            f"<|im_start|>system\n{SYS_MESSAGE}<|im_end|>\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        images = None

    return message, images

# OpenAI
def create_openai_prompt(
    question, lang, instruction, few_shot_setting, experiment
):
    """
    Outputs: conversation dictionary supported by OpenAI.
    """
    system_message = {"role": "system", "content": instruction}
    question_text = question['question']
    question_image = question['image']
    options_list = question['options']
    def encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                binary_data = image_file.read()
                base_64_encoded_data = base64.b64encode(binary_data)
                base64_string = base_64_encoded_data.decode("utf-8")
                return base64_string
        except Exception as e:
            raise TypeError(f"Image {image_path} could not be encoded. {e}")

    if question_image:
        base64_image = encode_image(question_image)
        if experiment == 'captioning':
            caption = question['image_caption']
            ocr = question['image_ocr']
            question = [
                {"type": "text", "text": question_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low",
                    },
                },
                {"type": "text", "text": f'Caption: {caption} \n OCR: {ocr}'}
            ]
        else:
            question = [
                {"type": "text", "text": question_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low",
                    },
                }
            ]
    else:
        question = [{"type": "text", "text": question_text}]

    # Parse options. Handle options with images carefully by inclusion in the conversation.
    parsed_options = [{"type": "text", "text": "Options:\n"}]
    only_text_option = {"type": "text", "text": "{text}"}
    only_image_option = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,{base64_image}", "detail": "low"},
    }

    for i, option in enumerate(options_list):
        option_indicator = f"{chr(65+i)}. "
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

    user_text = question + parsed_options
    user_message = {"role": "user", "content": user_text}

    messages = [system_message, user_message]
    
    return messages, None  # image paths not expected for openai client.


# Anthropic
def create_anthropic_prompt(
    question, lang, instruction, few_shot_setting
):
    """
    Outputs: conversation dictionary supported by Anthropic.
    """
    system_message = {"role": "system", "content": instruction}
    question_text = question['question']
    question_image = question['image']
    options_list = question['options']

    def resize_and_encode_image(image_path):
        try:
            with Image.open(image_path) as img:
                # Resize the image to 512x512 using an appropriate resampling filter
                resized_img = img.resize((512, 512), Image.LANCZOS)

                # Save the resized image to a bytes buffer in PNG format
                buffer = io.BytesIO()
                resized_img.save(buffer, format="PNG")
                buffer.seek(0)

                # Encode the image in base64
                base64_encoded = base64.b64encode(buffer.read()).decode("utf-8")
                return base64_encoded
        except Exception as e:
            raise TypeError(f"Image {image_path} could not be processed. {e}")

    if question_image:
        base64_image = resize_and_encode_image(question_image)
        question = [
            {"type": "text", "text": question_text},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image,
                },
            },
        ]
    else:
        question = [{"type": "text", "text": question_text}]

    # Parse options. Handle options with images carefully by inclusion in the conversation.
    parsed_options = [{"type": "text", "text": "Options:\n"}]
    only_text_option = {"type": "text", "text": "{text}"}

    for i, option in enumerate(options_list):
        option_indicator = f"{chr(65+i)}. "
        if option.lower().endswith(".png"):
            # Generating the dict format of the conversation if the option is an image
            new_image_option = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": resize_and_encode_image(option),
                },
            }
            new_text_option = only_text_option.copy()
            formated_text = new_text_option["text"].format(text=option_indicator + "\n")
            new_text_option["text"] = formated_text

            parsed_options.append(new_text_option)
            parsed_options.append(new_image_option)

        else:
            # Generating the dict format of the conversation if the option is not an image
            new_text_option = only_text_option.copy()
            formated_text = new_text_option["text"].format(
                text=option_indicator + option + "\n"
            )
            new_text_option["text"] = formated_text
            parsed_options.append(new_text_option)

    user_text = question + parsed_options
    user_message = {"role": "user", "content": user_text}

    messages = [system_message, user_message]
    
    return messages, None  # image paths not expected for openai client.



# Aya-Vision
def create_aya_prompt(question, method, few_shot_samples):
    lang = question["language"]
    prompt = []
    content = []
    # zero shot
    if question["image"] is not None:
        with open(question["image"], "rb") as img_file:
            img = Image.open(img_file).resize((512, 512))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            base64_image_url = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": base64_image_url},
            },
        )
    prompt.append(f"\n{INSTRUCTION}\n")
    prompt.append(f"\n{keywords[lang]['question']}: {question['question']}\n")
    prompt.append(f"{keywords[lang]['options']}:\n")
    for t, option in enumerate(question["options"]):
        index = f"{chr(65+t)}. "
        prompt.append(f"{index}) {option}\n")
    # prompt.append(f"\n{lang_keyword['answer']}:")
    prompt = "".join(prompt)
    content.append({"type": "text", "text": prompt})
    message = [
        {"role": "system", "content": SYS_MESSAGE},
        {"role": "user", "content": content},
    ]
    return message, None
