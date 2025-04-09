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

# GPT

# Gemini

# Claude


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
