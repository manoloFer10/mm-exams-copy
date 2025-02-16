import base64
import torch
import re
from io import BytesIO
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig,
)
from qwen_vl_utils import (
    process_vision_info,
)

# from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from deepseek_vl.utils.io import load_pil_images

from pathlib import Path
from PIL import Image
from openai import OpenAI
from anthropic import Anthropic
from torch.cuda.amp import autocast
from llava.model.builder import load_pretrained_model

from model_zoo import create_qwen2_prompt


TEMPERATURE = 0.7
MAX_TOKENS = 512

SUPPORTED_MODELS = [
    "gpt-4o",
    "qwen2-7b",
    "qwen2.5-7b",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "claude-3-5-sonnet-latest",
    "molmo",
    "deepseekVL2-small",
]  # "claude-3-5-haiku-latest" haiku does not support image input


INSTRUCTIONS_COT = {
    "en": "The following is a multiple-choice question. Think step by step and then provide your FINAL answer between the tags <ANSWER> X </ANSWER> where X is ONLY the correct letter of your choice. Do not write additional text between the tags.",
    "es": "Lo siguiente es una pregunta de opción múltiple. Piensa paso a paso y luego proporciona tu RESPUESTA FINAL entre las etiquetas <ANSWER> X </ANSWER>, donde X es ÚNICAMENTE la letra correcta de tu elección. No escribas texto adicional entre las etiquetas.",
    "hi": "निम्नलिखित एक बहुविकल्पीय प्रश्न है। चरणबद्ध सोचें और फिर <ANSWER> X </ANSWER> टैग के बीच अपना अंतिम उत्तर प्रदान करें, जहाँ X केवल आपके चयन का सही अक्षर है। टैग के बीच अतिरिक्त कोई पाठ न लिखें।",
    "hu": "A következő egy feleletválasztós kérdés. Gondolkodj lépésről lépésre, majd add meg a VÉGSŐ válaszodat a <ANSWER> X </ANSWER> címkék között, ahol X CSAK a választott helyes betű. Ne írj további szöveget a címkék közé.",
    "hr": "Sljedeće je pitanje s višestrukim izborom. Razmislite korak po korak, a zatim dajte svoj ZAVRŠNI odgovor između oznaka <ANSWER> X </ANSWER> gdje je X SAMO ispravno slovo vašeg izbora. Nemojte pisati dodatni tekst između oznaka.",
    "uk": "Наступне — це питання з множинним вибором. Думайте крок за кроком, а потім надайте вашу ОСТАННЮ відповідь між тегами <ANSWER> X </ANSWER>, де X — ЛИШЕ правильна літера за вашим вибором. Не пишіть додаткового тексту між тегами.",
    "pt": "A seguir, temos uma questão de múltipla escolha. Pense passo a passo e depois forneça sua RESPOSTA FINAL entre as tags <ANSWER> X </ANSWER>, onde X é SOMENTE a letra correta da sua escolha. Não escreva texto adicional entre as tags.",
    "bn": "নিম্নলিখিতটি একটি বহু-বিকল্প প্রশ্ন। ধাপে ধাপে চিন্তা করুন এবং তারপর <ANSWER> X </ANSWER> ট্যাগের মধ্যে আপনার চূড়ান্ত উত্তর প্রদান করুন, যেখানে X শুধুমাত্র আপনার পছন্দের সঠিক অক্ষর। ট্যাগগুলির মধ্যে অতিরিক্ত কোনো লেখা লিখবেন না।",
    "te": "కింద ఇచ్చినది ఒక బహుళ ఎంపిక ప్రశ్న. దశల వారీగా ఆలోచించి, <ANSWER> X </ANSWER> ట్యాగ్లలో మీ తుది సమాధానాన్ని ఇవ్వండి, ఇక్కడ X మీ ఎంపికలోని సరైన అక్షరం మాత్రమే. ట్యాగ్లలో అదనపు వచనం రాయవద్దు.",
    "ne": "तलको प्रश्न बहुविकल्पीय छ। चरणबद्ध सोच्नुहोस् र त्यसपछि <ANSWER> X </ANSWER> ट्यागहरूबीच आफ्नो अन्तिम उत्तर प्रदान गर्नुहोस्, जहाँ X केवल तपाईंको रोजाइको सही अक्षर हो। ट्यागहरूबीच अतिरिक्त पाठ नलेख्नुहोस्।",
    "sr": "Sledeće je pitanje sa višestrukim izborom. Razmislite korak po korak, a zatim dajte svoj KONAČNI odgovor između oznaka <ANSWER> X </ANSWER>, gde je X SAMO tačno slovo vašeg izbora. Nemojte pisati dodatni tekst između oznaka.",
    "nl": "Het volgende is een meerkeuzevraag. Denk stap voor stap na en geef dan je UITEINDLIJKE antwoord tussen de tags <ANSWER> X </ANSWER>, waarbij X ALLEEN de juiste letter van je keuze is. Schrijf geen extra tekst tussen de tags.",
    "ar": "التالي هو سؤال اختيار من متعدد. فكر خطوة بخطوة ثم قدم إجابتك النهائية بين الوسوم <ANSWER> X </ANSWER> حيث X هي الحرف الصحيح فقط من اختيارك. لا تكتب نصًا إضافيًا بين الوسوم.",
    "ru": "Следующее — это вопрос с выбором ответа. Думайте шаг за шагом, а затем предоставьте ваш ОКОНЧАТЕЛЬНЫЙ ответ между тегами <ANSWER> X </ANSWER>, где X — ТОЛЬКО правильная буква вашего выбора. Не пишите дополнительный текст между тегами.",
    "fr": "Ce qui suit est une question à choix multiple. Réfléchissez étape par étape, puis donnez votre RÉPONSE FINALE entre les balises <ANSWER> X </ANSWER>, où X est UNIQUEMENT la lettre correcte de votre choix. N'écrivez pas de texte supplémentaire entre les balises.",
    "fa": "متن زیر یک سوال چندگزینه‌ای است. مرحله به مرحله فکر کنید و سپس پاسخ نهایی خود را بین تگ‌های <ANSWER> X </ANSWER> قرار دهید، جایی که X تنها حرف صحیح انتخاب شماست. متن اضافی بین تگ‌ها ننویسید.",
    "de": "Im Folgenden ist eine Multiple-Choice-Frage. Denken Sie Schritt für Schritt nach und geben Sie dann Ihre ENDGÜLTIGE Antwort zwischen den Tags <ANSWER> X </ANSWER> an, wobei X NUR der korrekte Buchstabe Ihrer Wahl ist. Schreiben Sie keinen zusätzlichen Text zwischen den Tags.",
    "lt": "Toliau pateikiamas klausimas su keliomis pasirinkimo galimybėmis. Mąstykite žingsnis po žingsnio ir pateikite savo GALUTINĮ atsakymą tarp žymų <ANSWER> X </ANSWER>, kur X yra TIK teisinga jūsų pasirinkta raidė. Nerašykite jokio papildomo teksto tarp žymų.",
}


def initialize_model(
    model_name: str, model_path: str, api_key: str = None, device: str = "cuda"
):
    """
    Initialize the model and processor/tokenizer based on the model name.
    """
    if model_name == "qwen2-7b":

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            Path(model_path) / "model",
            # torch_dtype=torch.float16,
            temperature=TEMPERATURE,
            device_map=device,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            Path(model_path) / "processor", local_files_only=True
        )

    elif model_name == "qwen2.5-7b":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            temperature=TEMPERATURE,
            device_map=device,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )

    elif model_name == "deepseekVL2-small":
        model_path = "deepseek-ai/deepseek-vl2-small"
        processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
            Path(model_path) / "model", local_files_only=True
        )

        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            Path(model_path) / "processor",
            trust_remote_code=True,
            temperature=0.7,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).eval()

    elif model_name == "molmo":
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            temperature=TEMPERATURE,
            do_sample=True,
            device_map=device,
            local_files_only=True,
        ).eval()

    elif model_name == "pangea":
        model_name = "Pangea-7B-qwen"
        args = {"multimodal": True}
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, **args
        )
        processor = {"tokenizer": tokenizer, "image_processor": processor}
    elif model_name in ["gpt-4o", "gpt-4o-mini"]:
        model = OpenAI(api_key=api_key)
        processor = None
    elif model_name in ["gemini-2.0-flash-exp", "gemini-1.5-pro"]:
        model = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        processor = None
    elif model_name == "claude-3-5-sonnet-latest":
        model = Anthropic(api_key=api_key)
        processor = None
    else:
        raise NotImplementedError(
            f"Model {model_name} not currently implemented for prediction. Supported Models: {SUPPORTED_MODELS}"
        )

    print(f"Model {model_name} loaded from {model_path}")

    return model, processor


def query_model(
    model_name: str,
    model,
    processor,
    prompt: list,
    images,
    device: str = "cuda",
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
):
    """
    Query the model based on the model name.
    """
    if model_name == "qwen2-7b":  # ERASE: should erase after 2.5 works well
        answer = query_qwen2(model, processor, prompt, images, device)

    elif model_name == "qwen2.5-7b":
        answer = query_qwen25(model, processor, prompt, device, max_tokens)
    elif model_name == "pangea":
        # Add pangea querying logic
        raise NotImplementedError(f"Model {model_name} not implemented for querying.")
    elif model_name == "deepseekVL2-small":
        answer = query_deepseek(model, processor, prompt, max_tokens)
    elif model_name == "molmo":
        answer = query_molmo(model, processor, prompt, images, max_tokens)
    elif model_name in [
        "gpt-4o",
        "gpt-4o-mini",
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
    ]:
        answer = query_openai(model, model_name, prompt, temperature, max_tokens)

    elif model_name == "claude-3-5-sonnet-latest":
        answer = query_anthropic(model, model_name, prompt, temperature, max_tokens)
    elif model_name == "maya":
        # Add Maya-specific parsing
        raise NotImplementedError(f"Model {model_name} not implemented for querying.")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return format_answer(answer)


def query_pangea():
    pass


def query_deepseek(model, processor, prompt: list, max_tokens=MAX_TOKENS):
    instruction = prompt[0]
    conversation = prompt[1]

    tokenizer = model.tokenizer
    pil_images = load_pil_images(prompt)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=instruction,
    ).to(model.device)

    # run image encoder to get the image embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = model.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_tokens,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    return answer


def query_molmo(model, processor, prompt: list, image_path: list, max_tokens):
    if prompt == "multi-image":
        print("Question was multi-image, molmo does not support multi-image inputs.")
        return "multi-image detected"
    else:
        inputs = processor.process(
            images=[Image.open(image_path).convert("RGB").resize((224, 224))],
            text=prompt,
        )
        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_tokens, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return generated_text


def query_openai(client, model_name, prompt, temperature, max_tokens):
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_text = response.choices[0].message.content.strip()
    return output_text


def query_anthropic(client, model_name, prompt, temperature, max_tokens):

    system_message = prompt[0]["content"]
    user_messages = prompt[1]

    response = client.messages.create(
        model=model_name,
        messages=[user_messages],
        system=system_message,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_text = response.content[0].text
    return output_text


# ERASE: should erase after 2.5 works well
def query_qwen2(
    model,
    processor,
    prompt: list,
    image_paths: list,
    device="cuda",
    max_tokens=MAX_TOKENS,
):
    # images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    try:
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    except:
        return "Image not found"

    text_prompt = processor.apply_chat_template(prompt, add_generation_prompt=True)

    if len(images) == 0:
        images = None
    inputs = processor(
        text=[text_prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    torch.cuda.empty_cache()
    return response[0]


def query_qwen25(model, processor, prompt: list, device="cuda", max_tokens=MAX_TOKENS):
    text = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(prompt)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate response
    # with torch.no_grad():
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    torch.cuda.empty_cache()
    return response[0]


def generate_prompt(
    model_name: str,
    question: dict,
    lang: str,
    instruction,
    method: str = "zero-shot",
):
    if model_name == "qwen2-7b":
        return create_qwen2_prompt(question, method)
    if model_name == "qwen2.5-7b":
        return parse_qwen25_input(
            question["question"],
            question["image"],
            question["options"],
            lang,
            instruction,
            method,
        )
    elif model_name in [
        "gpt-4o",
        "gpt-4o-mini",
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
    ]:
        return parse_openai_input(
            question["question"],
            question["image"],
            question["options"],
            lang,
            instruction,
            method,
        )
    elif model_name == "maya":
        # Add Maya-specific parsing
        raise NotImplementedError(f"Model {model_name} not implemented for parsing.")
    elif model_name == "pangea":
        return parse_pangea_inputs(
            question["question"],
            question["image"],
            question["options"],
            instruction,
            method,
        )
        raise NotImplementedError(f"Model {model_name} not implemented for parsing.")
    elif model_name == "deepseekVL2-small":
        return parse_deepseek_inputs(
            question["question"],
            question["image"],
            question["options"],
            instruction,
            method,
        )
    elif model_name == "molmo":
        return parse_molmo_inputs(
            question["question"],
            question["image"],
            question["options"],
            instruction,
            method,
        )
    elif model_name == "claude-3-5-sonnet-latest":
        return parse_anthropic_input(
            question["question"],
            question["image"],
            question["options"],
            lang,
            instruction,
            method,
        )
    else:
        raise ValueError(f"Unsupported model for parsing inputs: {model_name}")


def parse_openai_input(
    question_text, question_image, options_list, lang, instruction, few_shot_setting
):
    """
    Outputs: conversation dictionary supported by OpenAI.
    """
    system_message = {"role": "system", "content": instruction}

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
        question = [
            {"type": "text", "text": question_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low",
                },
            },
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

    # Enable few-shot setting
    if few_shot_setting == "few-shot":
        user_message["content"] = (
            fetch_few_shot_examples(lang) + user_message["content"]
        )
        messages = [system_message, user_message]
    elif few_shot_setting == "zero-shot":
        messages = [system_message, user_message]
    else:
        raise ValueError(f"Invalid few_shot_setting: {few_shot_setting}")

    return messages, None  # image paths not expected for openai client.


def parse_anthropic_input(
    question_text, question_image, options_list, lang, instruction, few_shot_setting
):
    """
    Outputs: conversation dictionary supported by Anthropic.
    """
    system_message = {"role": "system", "content": instruction}

    def resize_and_encode_image(image_path):
        try:
            with Image.open(image_path) as img:
                # Resize the image to 512x512 using an appropriate resampling filter
                resized_img = img.resize((512, 512), Image.LANCZOS)

                # Save the resized image to a bytes buffer in PNG format
                buffer = BytesIO()
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

    # Enable few-shot setting
    if few_shot_setting == "few-shot":
        user_message["content"] = (
            fetch_few_shot_examples(lang) + user_message["content"]
        )
        messages = [system_message, user_message]
    elif few_shot_setting == "zero-shot":
        messages = [system_message, user_message]
    else:
        raise ValueError(f"Invalid few_shot_setting: {few_shot_setting}")

    return messages, None  # image paths not expected for openai client.


def parse_deepseek_inputs(
    question_text, question_image, options_list, instruction, few_shot_setting
):
    images = [question_image]
    for option in options_list:
        if ".png" in option:
            images.append(option)

    user_msg = "Question: " + question_text + "\n\n"

    if question_image:
        user_msg += "<image>"

    options = "Options:\n"
    for i, option in enumerate(options_list):
        if ".png" in option:
            options += chr(65 + i) + ". " + "<image>" + "\n"
        else:
            options += chr(65 + i) + ". " + option + "\n"

    user_msg += options + "\nAnswer:"

    conversation = [
        {
            "role": "<|User|>",
            "content": user_msg,
            "images": images,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return [
        instruction,
        conversation,
    ], None  # image paths not expected for deepseek. Processing of images is within prompt.


def parse_molmo_inputs(
    question_text, question_image, options_list, instruction, few_shot_setting
):
    for option in options_list:
        if ".png" in option:
            return "multi-image", None

    if instruction != "":
        prompt = instruction + "\n\n"
        prompt += "Question: " + question_text + "\n\n"
    else:
        prompt = "Question: " + question_text + "\n\n"

    if question_image:
        prompt += "<image>"

    options = "Options:\n"
    for i, option in enumerate(options_list):
        options += chr(65 + i) + ". " + option + "\n"

    prompt += options + "\nAnswer:"

    return prompt, question_image


def parse_qwen25_input(
    question_text, question_image, options_list, lang, instruction, few_shot_setting
):
    """
    Outputs: conversation dictionary supported by qwen2.5 .
    """
    system_message = {"role": "system", "content": instruction}

    if question_image:
        question = [
            {"type": "text", "text": f"Question: {question_text}"},
            {"type": "image", "image": f"file:///{question_image}"},
        ]
    else:
        question = [
            {
                "type": "text",
                "text": f"Question: {question_text}",
            }
        ]

    parsed_options = [{"type": "text", "text": "Options:\n"}]
    only_text_option = {"type": "text", "text": "{text}"}
    only_image_option = {"type": "image", "image": "file:///{image_path}"}

    images_paths = []
    for i, option in enumerate(options_list):
        option_indicator = f"{chr(65+i)}. "
        if option.lower().endswith(".png"):  # Checks if it is a png file
            # Generating the dict format of the conversation if the option is an image
            new_image_option = only_image_option.copy()
            new_image_option["image"] = new_image_option["image"].format(
                image_path=option
            )
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

    user_text = question + parsed_options
    user_message = {"role": "user", "content": user_text}

    # Enable few-shot setting
    if few_shot_setting == "few-shot":
        user_message["content"] = (
            fetch_few_shot_examples(lang) + user_message["content"]
        )
        messages = [system_message, user_message]
    elif few_shot_setting == "zero-shot":
        messages = [system_message, user_message]
    else:
        raise ValueError(f"Invalid few_shot_setting: {few_shot_setting}")

    return messages, None  # image paths processed in messages by process_vision_info.


# ERASE: should erase after 2.5 works well
def parse_qwen2_input(
    question_text, question_image, options_list, lang, instruction, few_shot_setting
):
    """
    Outputs: conversation dictionary supported by qwen.
    """
    system_message = {"role": "system", "content": instruction}

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

    parsed_options = [{"type": "text", "text": "Options:\n"}]
    only_text_option = {"type": "text", "text": "{text}"}
    only_image_option = {"type": "image"}

    images_paths = []
    for i, option in enumerate(options_list):
        option_indicator = f"{chr(65+i)}. "
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

    user_text = question + parsed_options
    user_message = {"role": "user", "content": user_text}

    # Enable few-shot setting
    if few_shot_setting == "few-shot":
        user_message["content"] = (
            fetch_few_shot_examples(lang) + user_message["content"]
        )
        messages = [system_message, user_message]
    elif few_shot_setting == "zero-shot":
        messages = [system_message, user_message]
    else:
        raise ValueError(f"Invalid few_shot_setting: {few_shot_setting}")

    if question_image:
        images_paths = [question_image] + images_paths

    return messages, images_paths


def format_answer(answer: str):
    """
    Searchs for the answer between tags <Answer>.

    Returns: A zero-indexed integer corresponding to the answer.
    """
    pattern = r"<ANSWER>\s*([A-Za-z])\s*</ANSWER>"
    match = re.search(pattern, answer, re.IGNORECASE)

    if match:
        # Extract and convert answer letter
        letter = match.group(1).upper()
        election = ord(letter) - ord("A")

        # Extract reasoning by removing answer tag section
        start, end = match.span()
        reasoning = (answer[:start] + answer[end:]).strip()
        # Clean multiple whitespace
        reasoning = re.sub(r"\s+", " ", reasoning)
    else:
        # Error handling cases
        election = "No valid answer tag found"
        if re.search(r"<ANSWER>.*?</ANSWER>", answer):
            election = "Answer tag exists but contains invalid format"
        reasoning = answer.strip()

    return reasoning, election


def parse_pangea_inputs(question):
    return messages, image_paths


def fetch_cot_instruction(lang: str) -> str:
    """
    Retrieves the CoT Instruction for the given lang.
    """
    if lang in INSTRUCTIONS_COT.keys():
        return INSTRUCTIONS_COT[lang]
    else:
        raise ValueError(f"{lang} language code not in INSTRUCTIONS_COT")


def fetch_few_shot_examples(lang):
    # TODO: write function. Should output a list of dicts in the conversation format expected.
    # I reckon we should do as parse_client_input with these. Add few-shot image examples regarding the format the input model expects.
    raise NotImplementedError(
        "The function to fetch few_shot examples is not yet implemented, but should return the few shot examples regarding that language."
    )
