import base64
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from pathlib import Path
from PIL import Image
from openai import OpenAI

TEMPERATURE = 0
MAX_TOKENS = 1  # Only output the option chosen.

SUPPORTED_MODELS = ["gpt-4o", "qwen2-7b", "maya", "llama"]

# Update manually with supported languages translation
SYSTEM_MESSAGES = {
    "en": "You are given a multiple choice question for answering. You MUST only answer with the correct number of the answer. For example, if you are given the options 1, 2, 3, 4 and option 2 is correct, then you should return the number 2.",
    "fa": "شما یک سوال چند گزینه‌ای برای پاسخ دادن دریافت می‌کنید. شما باید تنها با شماره صحیح پاسخ را بدهید. به عنوان مثال، اگر گزینه‌های ۱، ۲، ۳، ۴ داده شده باشد و گزینه ۲ درست باشد، باید عدد ۲ را بازگردانید.",
    "es": "Se te da una pregunta de opción múltiple para responder. DEBES responder solo con el número correcto de la respuesta. Por ejemplo, si se te dan las opciones 1, 2, 3, 4 y la opción 2 es correcta, entonces debes devolver el número 2.",
    "bn": "আপনাকে একটি একাধিক পছন্দ প্রশ্ন দেওয়া হয়েছে উত্তর দেওয়ার জন্য। আপনাকে শুধুমাত্র সঠিক উত্তরের সংখ্যা দিয়ে উত্তর দিতে হবে। উদাহরণস্বরূপ, যদি আপনাকে ১, ২, ৩, ৪ বিকল্প দেওয়া হয় এবং বিকল্প ২ সঠিক হয়, তবে আপনাকে সংখ্যা ২ ফেরত দিতে হবে।",
    "hi": "आपको उत्तर देने के लिए एक बहुविकल्पीय प्रश्न दिया गया है। आपको केवल सही उत्तर के नंबर से ही उत्तर देना चाहिए। उदाहरण के लिए, यदि आपको विकल्प 1, 2, 3, 4 दिए गए हैं और विकल्प 2 सही है, तो आपको नंबर 2 लौटाना चाहिए।",
    "lt": "Jums pateiktas klausimas su keliais pasirinkimais atsakyti. TURITE atsakyti tik teisingu atsakymo numeriu. Pavyzdžiui, jei pateikiami variantai 1, 2, 3, 4, o teisingas variantas yra 2, turite grąžinti skaičių 2.",
    "zh": "您被给出一个多项选择题来回答。您必须仅用正确的答案编号作答。例如，如果给出的选项是1, 2, 3, 4，而选项2是正确的，那么您应该返回数字2。",
    "nl": "Je krijgt een meerkeuzevraag om te beantwoorden. Je MOET alleen antwoorden met het juiste nummer van het antwoord. Bijvoorbeeld, als je de opties 1, 2, 3, 4 krijgt en optie 2 is correct, dan moet je nummer 2 retourneren.",
    "te": "మీకు ఒక బహుళ ఎంపిక ప్రశ్న ఇస్తారు. మీరు తప్పనిసరిగా సరైన సమాధానం సంఖ్యతో మాత్రమే సమాధానం ఇవ్వాలి. ఉదాహరణకు, మీకు 1, 2, 3, 4 ఎంపికలు ఇస్తే మరియు ఎంపిక 2 సరైనది అయితే, మీరు సంఖ్య 2ను తిరిగి ఇవ్వాలి.",
    "uk": "Вам надано питання з кількома варіантами відповідей. ВИ МАЄТЕ відповідати лише правильним номером відповіді. Наприклад, якщо вам дано варіанти 1, 2, 3, 4, і правильним є варіант 2, ви повинні повернути номер 2.",
    "pa": "ਤੁਹਾਨੂੰ ਉੱਤਰ ਦੇਣ ਲਈ ਇੱਕ ਬਹੁ-ਚੋਣ ਪ੍ਰਸ਼ਨ ਦਿੱਤਾ ਗਿਆ ਹੈ। ਤੁਹਾਨੂੰ ਸਿਰਫ਼ ਸਹੀ ਜਵਾਬ ਦੀ ਗਿਣਤੀ ਨਾਲ ਜਵਾਬ ਦੇਣਾ ਚਾਹੀਦਾ ਹੈ। ਉਦਾਹਰਣ ਲਈ, ਜੇ ਤੁਹਾਨੂੰ ਵਿਕਲਪ 1, 2, 3, 4 ਦਿੱਤੇ ਜਾਂਦੇ ਹਨ ਅਤੇ ਵਿਕਲਪ 2 ਸਹੀ ਹੈ, ਤਾਂ ਤੁਹਾਨੂੰ ਨੰਬਰ 2 ਵਾਪਸ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ।",
    "sk": "Máte zadanú otázku s viacerými možnosťami odpovede. MUSÍTE odpovedať iba správnym číslom odpovede. Napríklad, ak sú vám dané možnosti 1, 2, 3, 4 a správnou možnosťou je 2, mali by ste vrátiť číslo 2.",
    "pl": "Otrzymujesz pytanie wielokrotnego wyboru do odpowiedzi. MUSISZ odpowiedzieć tylko poprawnym numerem odpowiedzi. Na przykład, jeśli podano opcje 1, 2, 3, 4, a poprawną opcją jest 2, powinieneś zwrócić liczbę 2.",
    "cs": "Dostanete otázku s výběrem odpovědí. MUSÍTE odpovědět pouze správným číslem odpovědi. Například, pokud dostanete možnosti 1, 2, 3, 4 a možnost 2 je správná, měli byste vrátit číslo 2.",
    "de": "Ihnen wird eine Multiple-Choice-Frage zur Beantwortung gegeben. Sie DÜRFEN nur mit der richtigen Nummer der Antwort antworten. Wenn Ihnen beispielsweise die Optionen 1, 2, 3, 4 gegeben werden und Option 2 korrekt ist, sollten Sie die Nummer 2 zurückgeben.",
    "vi": "Bạn được cung cấp một câu hỏi trắc nghiệm để trả lời. Bạn PHẢI chỉ trả lời bằng số đúng của câu trả lời. Ví dụ, nếu bạn được đưa ra các lựa chọn 1, 2, 3, 4 và lựa chọn 2 là đúng, thì bạn nên trả về số 2.",
    "ne": "तपाईंलाई उत्तर दिनको लागि एक बहुविकल्पीय प्रश्न दिइएको छ। तपाईंले मात्र सही उत्तरको नम्बरले उत्तर दिनुपर्छ। उदाहरणका लागि, यदि तपाईंलाई विकल्पहरू 1, 2, 3, 4 दिइन्छ र विकल्प 2 सही छ भने, तपाईंले नम्बर 2 फिर्ता गर्नुपर्छ।"
}


def initialize_model(
    model_name: str, model_path: str, api_key: str = None, device: str = "cuda"
):
    """
    Initialize the model and processor/tokenizer based on the model name.
    """
    temperature = TEMPERATURE
    max_tokens = MAX_TOKENS

    if model_name == "qwen2-7b":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            Path(model_path) / "model",
            torch_dtype=torch.float16,
            temperature=temperature,
            device_map=device,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            Path(model_path) / "processor", local_files_only=True
        )
    elif model_name == "pangea":
        # Add Pangea initialization logic
        raise NotImplementedError(f"Model: {model_name} not available yet")
    elif model_name == "molmo":
        # Add Molmo initialization logic
        raise NotImplementedError(f"Model: {model_name} not available yet")
    elif model_name == "gpt-4o":
        client = OpenAI(api_key=api_key)
        model = client
        processor = None
    else:
        raise NotImplementedError(
            f"Model {model} not currently implemented for prediction. Supported Models: {SUPPORTED_MODELS}"
        )
    return model, processor


def query_model(
    model_name: str, 
    model, 
    processor, 
    prompt: list, 
    images=None, 
    device: str = "cuda", 
    temperature = TEMPERATURE,
    max_tokens = MAX_TOKENS
):
    """
    Query the model based on the model name.
    """
    if model_name == "qwen2-7b":
        return query_qwen(model, processor, prompt, images, device)
    elif model_name == "pangea":
        # Add pangea querying logic
        raise NotImplementedError(f'Model {model_name} not implemented for querying.')
    elif model_name == "molmo":
        # Add molmo querying logic
        raise NotImplementedError(f'Model {model_name} not implemented for querying.')
    elif model_name == "gpt-4o":
        return query_openai(model, model_name, prompt, temperature, max_tokens)
    elif model_name == "maya":
        # Add Maya-specific parsing
        raise NotImplementedError(f'Model {model_name} not implemented for querying.')
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def query_pangea():
    pass


def query_molmo():
    pass


def query_openai(client, model_name, prompt, temperature, max_tokens):
    response = client.chat.completions.create(
            model=model_name, messages=prompt, temperature=temperature, max_tokens=max_tokens
    )
    output_text = response.choices[0].message.content.strip()
    return format_answer(output_text)


def query_qwen(
    model,
    processor,
    prompt: list,
    image_paths: list,
    device="cuda",
    max_tokens= MAX_TOKENS
):
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    inputs = processor(
        text=prompt,  
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Generate response
    output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return format_answer(response[0])



def generate_prompt(model_name: str, question: dict,lang: str, system_message, few_shot_setting: str = 'zero-shot'):
    if model_name == "qwen2-7b":
        return parse_qwen_input(
            question["question"], question["image"], question["options"], lang, system_message, few_shot_setting
        )
    elif model_name == "gpt-4o":
        return parse_openai_input(
            question["question"], question["image"], question["options"], lang, system_message, few_shot_setting
        )
    elif model_name == "maya":
        # Add Maya-specific parsing
        raise NotImplementedError(f'Model {model_name} not implemented for parsing.')
    elif model_name == "pangea":
        # Add pangea querying logic
        raise NotImplementedError(f'Model {model_name} not implemented for parsing.')
    elif model_name == "molmo":
        # Add molmo querying logic
        raise NotImplementedError(f'Model {model_name} not implemented for parsing.')
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def parse_openai_input(question_text, question_image, options_list, lang, system_message, few_shot_setting):
    '''
    Outputs: conversation dictionary supported by OpenAI.
    '''
    system_message = [{"role": "system", "content": system_message}]

    def encode_image(image):
        try:
            return base64.b64encode(image).decode("utf-8")
        except Exception as e:
            raise TypeError(f"Image {image} could not be encoded. {e}")

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

    user_text = [question] + parsed_options
    user_message = {
        'role': 'user',
        'content': user_text
    }

    # Enable few-shot setting
    if few_shot_setting == "few-shot":
        user_message['content'] = fetch_few_shot_examples(lang) + user_message['content']
        messages = [system_message, user_message]
    elif few_shot_setting == "zero-shot":
        messages = [system_message, user_message]
    else:
        raise ValueError(f'Invalid few_shot_setting: {few_shot_setting}')

    return messages, None

def parse_qwen_input(question_text, question_image, options_list, lang, system_message, few_shot_setting):
    '''
    Outputs: conversation dictionary supported by qwen.
    '''
    system_message = [{"role": "system", "content": system_message}]

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

    user_text = [question] + parsed_options
    user_message = {
        'role': 'user',
        'content': user_text
    }

    # Enable few-shot setting
    if few_shot_setting == "few-shot":
        user_message['content'] = fetch_few_shot_examples(lang) + user_message['content']
        messages = [system_message, user_message]
    elif few_shot_setting == "zero-shot":
        messages = [system_message, user_message]
    else:
        raise ValueError(f'Invalid few_shot_setting: {few_shot_setting}')

    if question_image:
        image_paths = [question_image] + image_paths

    return messages, images_paths


def format_answer(answer: str):
    """
    Returns: A zero-indexed integer corresponding to the answer.
    """
    if not isinstance(answer, str):
        raise ValueError(f"Invalid input: '{answer}'.")
    if len(answer) != 1:
        answer = answer[0]

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
