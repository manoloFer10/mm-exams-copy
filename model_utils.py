import base64
import torch
import re
from transformers import (
                        Qwen2VLForConditionalGeneration,
                        Qwen2_5_VLForConditionalGeneration,
                        AutoProcessor,
                        AutoModelForCausalLM, 
                        GenerationConfig
                        )
from qwen_vl_utils import process_vision_info # pip install git+https://github.com/huggingface/transformers accelerate
from pathlib import Path
from PIL import Image
from openai import OpenAI
from anthropic import Anthropic
from torch.cuda.amp import autocast

TEMPERATURE = 0 # Set to 0.7
MAX_TOKENS = 1  # Only output the option chosen.
# MAX_TOKENS = 256 

SUPPORTED_MODELS = ["gpt-4o-mini", "qwen2-7b", "qwen2.5-7b", "gemini-2.0-flash-exp"] # "claude-3-5-haiku-latest" haiku does not support image input

# Update manually with supported languages translation
# SYSTEM_MESSAGES = {
#     "en": "You are given a multiple choice question for answering. You MUST only answer with the correct number of the answer. For example, if you are given the options 1, 2, 3, 4 and option 2 is correct, then you should return the number 2.",
#     "fa": "شما یک سوال چند گزینه‌ای برای پاسخ دادن دریافت می‌کنید. شما باید تنها با شماره صحیح پاسخ را بدهید. به عنوان مثال، اگر گزینه‌های ۱، ۲، ۳، ۴ داده شده باشد و گزینه ۲ درست باشد، باید عدد ۲ را بازگردانید.",
#     "es": "Se te da una pregunta de opción múltiple para responder. DEBES responder solo con el número correcto de la respuesta. Por ejemplo, si se te dan las opciones 1, 2, 3, 4 y la opción 2 es correcta, entonces debes devolver el número 2.",
#     "bn": "আপনাকে একটি একাধিক পছন্দ প্রশ্ন দেওয়া হয়েছে উত্তর দেওয়ার জন্য। আপনাকে শুধুমাত্র সঠিক উত্তরের সংখ্যা দিয়ে উত্তর দিতে হবে। উদাহরণস্বরূপ, যদি আপনাকে ১, ২, ৩, ৪ বিকল্প দেওয়া হয় এবং বিকল্প ২ সঠিক হয়, তবে আপনাকে সংখ্যা ২ ফেরত দিতে হবে।",
#     "hi": "आपको उत्तर देने के लिए एक बहुविकल्पीय प्रश्न दिया गया है। आपको केवल सही उत्तर के नंबर से ही उत्तर देना चाहिए। उदाहरण के लिए, यदि आपको विकल्प 1, 2, 3, 4 दिए गए हैं और विकल्प 2 सही है, तो आपको नंबर 2 लौटाना चाहिए।",
#     "lt": "Jums pateiktas klausimas su keliais pasirinkimais atsakyti. TURITE atsakyti tik teisingu atsakymo numeriu. Pavyzdžiui, jei pateikiami variantai 1, 2, 3, 4, o teisingas variantas yra 2, turite grąžinti skaičių 2.",
#     "zh": "您被给出一个多项选择题来回答。您必须仅用正确的答案编号作答。例如，如果给出的选项是1, 2, 3, 4，而选项2是正确的，那么您应该返回数字2。",
#     "nl": "Je krijgt een meerkeuzevraag om te beantwoorden. Je MOET alleen antwoorden met het juiste nummer van het antwoord. Bijvoorbeeld, als je de opties 1, 2, 3, 4 krijgt en optie 2 is correct, dan moet je nummer 2 retourneren.",
#     "te": "మీకు ఒక బహుళ ఎంపిక ప్రశ్న ఇస్తారు. మీరు తప్పనిసరిగా సరైన సమాధానం సంఖ్యతో మాత్రమే సమాధానం ఇవ్వాలి. ఉదాహరణకు, మీకు 1, 2, 3, 4 ఎంపికలు ఇస్తే మరియు ఎంపిక 2 సరైనది అయితే, మీరు సంఖ్య 2ను తిరిగి ఇవ్వాలి.",
#     "uk": "Вам надано питання з кількома варіантами відповідей. ВИ МАЄТЕ відповідати лише правильним номером відповіді. Наприклад, якщо вам дано варіанти 1, 2, 3, 4, і правильним є варіант 2, ви повинні повернути номер 2.",
#     "pa": "ਤੁਹਾਨੂੰ ਉੱਤਰ ਦੇਣ ਲਈ ਇੱਕ ਬਹੁ-ਚੋਣ ਪ੍ਰਸ਼ਨ ਦਿੱਤਾ ਗਿਆ ਹੈ। ਤੁਹਾਨੂੰ ਸਿਰਫ਼ ਸਹੀ ਜਵਾਬ ਦੀ ਗਿਣਤੀ ਨਾਲ ਜਵਾਬ ਦੇਣਾ ਚਾਹੀਦਾ ਹੈ। ਉਦਾਹਰਣ ਲਈ, ਜੇ ਤੁਹਾਨੂੰ ਵਿਕਲਪ 1, 2, 3, 4 ਦਿੱਤੇ ਜਾਂਦੇ ਹਨ ਅਤੇ ਵਿਕਲਪ 2 ਸਹੀ ਹੈ, ਤਾਂ ਤੁਹਾਨੂੰ ਨੰਬਰ 2 ਵਾਪਸ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ।",
#     "sk": "Máte zadanú otázku s viacerými možnosťami odpovede. MUSÍTE odpovedať iba správnym číslom odpovede. Napríklad, ak sú vám dané možnosti 1, 2, 3, 4 a správnou možnosťou je 2, mali by ste vrátiť číslo 2.",
#     "pl": "Otrzymujesz pytanie wielokrotnego wyboru do odpowiedzi. MUSISZ odpowiedzieć tylko poprawnym numerem odpowiedzi. Na przykład, jeśli podano opcje 1, 2, 3, 4, a poprawną opcją jest 2, powinieneś zwrócić liczbę 2.",
#     "cs": "Dostanete otázku s výběrem odpovědí. MUSÍTE odpovědět pouze správným číslem odpovědi. Například, pokud dostanete možnosti 1, 2, 3, 4 a možnost 2 je správná, měli byste vrátit číslo 2.",
#     "de": "Ihnen wird eine Multiple-Choice-Frage zur Beantwortung gegeben. Sie DÜRFEN nur mit der richtigen Nummer der Antwort antworten. Wenn Ihnen beispielsweise die Optionen 1, 2, 3, 4 gegeben werden und Option 2 korrekt ist, sollten Sie die Nummer 2 zurückgeben.",
#     "vi": "Bạn được cung cấp một câu hỏi trắc nghiệm để trả lời. Bạn PHẢI chỉ trả lời bằng số đúng của câu trả lời. Ví dụ, nếu bạn được đưa ra các lựa chọn 1, 2, 3, 4 và lựa chọn 2 là đúng, thì bạn nên trả về số 2.",
#     "ne": "तपाईंलाई उत्तर दिनको लागि एक बहुविकल्पीय प्रश्न दिइएको छ। तपाईंले मात्र सही उत्तरको नम्बरले उत्तर दिनुपर्छ। उदाहरणका लागि, यदि तपाईंलाई विकल्पहरू 1, 2, 3, 4 दिइन्छ र विकल्प 2 सही छ भने, तपाईंले नम्बर 2 फिर्ता गर्नुपर्छ।",
# }

INSTRUCTIONS_COT = {
    "en": "The following is a multiple-choice question. Think step by step and then provide your final answer between the tags <ANSWER> X </ANSWER> where X is the correct letter choice.",
    "es": "La siguiente es una pregunta de opción múltiple. Piensa paso a paso y luego proporciona tu respuesta final entre las etiquetas <ANSWER> X </ANSWER>, donde X es la letra correcta.",
    "hi": "निम्नलिखित एक बहुविकल्पीय प्रश्न है। चरण दर चरण सोचें और फिर अपने अंतिम उत्तर को <ANSWER> X </ANSWER> टैग के बीच प्रदान करें, जहाँ X सही विकल्प का अक्षर है।",
    "hu": "A következő egy feleletválasztós kérdés. Gondolkodj lépésről lépésre, majd add meg a végső válaszodat a <ANSWER> X </ANSWER> címkék között, ahol X a helyes betű.",
    "hr": "Sljedeće je pitanje s višestrukim izborom. Razmišljajte korak po korak, a zatim dajte svoj konačni odgovor između oznaka <ANSWER> X </ANSWER> gdje je X ispravno slovo opcije.",
    "uk": "Нижче наведено питання з множинним вибором. Думайте крок за кроком, а потім надайте свою кінцеву відповідь між тегами <ANSWER> X </ANSWER>, де X — правильна буква.",
    "pt": "A seguir está uma questão de múltipla escolha. Pense passo a passo e, em seguida, forneça sua resposta final entre as tags <ANSWER> X </ANSWER>, onde X é a letra correta.",
    "bn": "নিম্নলিখিত একটি বহুনির্বাচনী প্রশ্ন। ধাপে ধাপে চিন্তা করুন এবং তারপর আপনার চূড়ান্ত উত্তর <ANSWER> X </ANSWER> ট্যাগের মধ্যে প্রদান করুন, যেখানে X সঠিক অক্ষর চয়েস।",
    "te": "కింద ఇచ్చినది ఒక బహుళ ఎంపిక ప్రశ్న. దశల వారీగా ఆలోచించండి మరియు తర్వాత మీ తుది సమాధానాన్ని <ANSWER> X </ANSWER> ట్యాగ్ల మధ్య అందించండి, ఇక్కడ X సరైన అక్షర ఎంపిక.",
    "ne": "तलको प्रश्न बहुविकल्पीय छ। चरण दर चरण सोच्नुहोस् र त्यसपछि आफ्नो अन्तिम उत्तर <ANSWER> X </ANSWER> ट्यागहरू बीचमा दिनुहोस्, जहाँ X सही अक्षर विकल्प हो।",
    "sr": "Следеће је питање са више избора. Размишљајте корак по корак, а затим дајте свој коначни одговор између тагова <ANSWER> X </ANSWER>, где је X исправно слово опције.",
    "nl": "De volgende is een meerkeuzevraag. Denk stap voor stap na en geef vervolgens je definitieve antwoord tussen de tags <ANSWER> X </ANSWER>, waarbij X de juiste letterkeuze is.",
    "ar": "التالي هو سؤال اختيار من متعدد. فكر خطوة بخطوة ثم قدم إجابتك النهائية بين الوسوم <ANSWER> X </ANSWER>، حيث X هو الحرف الصحيح.",
    "ru": "Ниже приведён вопрос с множественным выбором. Думайте шаг за шагом, а затем предоставьте ваш окончательный ответ между тегами <ANSWER> X </ANSWER>, где X — правильный буквенный вариант.",
    "fr": "Ce qui suit est une question à choix multiples. Réfléchissez étape par étape, puis fournissez votre réponse finale entre les balises <ANSWER> X </ANSWER>, où X représente la bonne lettre.",
    "fa": "موارد زیر یک سوال چندگزینه‌ای است. گام به گام فکر کنید و سپس پاسخ نهایی خود را بین تگ‌های <ANSWER> X </ANSWER> ارائه دهید، جایی که X حرف گزینه صحیح است.",
    "de": "Die folgende Frage ist eine Multiple-Choice-Frage. Denken Sie Schritt für Schritt nach und geben Sie dann Ihre endgültige Antwort zwischen den Tags <ANSWER> X </ANSWER> an, wobei X der richtige Buchstabe ist.",
    "lt": "Toliau pateikiamas klausimas su keliomis atsakymų galimybėmis. Mąstykite žingsnis po žingsnio ir pateikite savo galutinį atsakymą tarp žymų <ANSWER> X </ANSWER>, kur X yra teisinga raidė."
}


# Deprecated
SYSTEM_MESSAGES = {
    "en": "You are given a multiple-choice question to answer. You MUST respond only with the number corresponding to the correct answer, without any additional text or explanation.",
    "fa": "شما یک سوال چند گزینه‌ای برای پاسخ دادن دریافت می‌کنید. شما باید فقط با شماره صحیح پاسخ دهید و هیچ توضیح اضافی ارائه ندهید.",
    "es": "Se te da una pregunta de opción múltiple para responder. DEBES responder solo con el número correspondiente a la respuesta correcta, sin ningún texto o explicación adicional.",
    "bn": "আপনাকে একটি একাধিক পছন্দ প্রশ্ন দেওয়া হয়েছে উত্তর দেওয়ার জন্য। আপনাকে শুধুমাত্র সঠিক উত্তরের সংখ্যা দিয়ে উত্তর দিতে হবে, কোনো অতিরিক্ত টেক্সট বা ব্যাখ্যা ছাড়া।",
    "hi": "आपको उत्तर देने के लिए एक बहुविकल्पीय प्रश्न दिया गया है। आपको केवल सही उत्तर के नंबर से उत्तर देना चाहिए, बिना किसी अतिरिक्त पाठ या व्याख्या के।",
    "lt": "Jums pateiktas klausimas su keliais pasirinkimais atsakyti. TURITE atsakyti tik teisingu atsakymo numeriu, be jokio papildomo teksto ar paaiškinimų.",
    "zh": "您被给出一个多项选择题来回答。您必须仅用正确的答案编号作答，不添加任何额外的文本或解释。",
    "nl": "Je krijgt een meerkeuzevraag om te beantwoorden. Je MOET alleen antwoorden met het juiste nummer van het antwoord, zonder enige extra tekst of uitleg.",
    "te": "మీకు ఒక బహుళ ఎంపిక ప్రశ్న ఇస్తారు. మీరు తప్పనిసరిగా సరైన సమాధానం సంఖ్యతో మాత్రమే సమాధానం ఇవ్వాలి, అదనపు పాఠ్యం లేదా వివరణ లేకుండా.",
    "uk": "Вам надано питання з кількома варіантами відповідей. ВИ МАЄТЕ відповідати лише номером правильної відповіді, без додаткового тексту чи пояснень.",
    "pa": "ਤੁਹਾਨੂੰ ਉੱਤਰ ਦੇਣ ਲਈ ਇੱਕ ਬਹੁ-ਚੋਣ ਪ੍ਰਸ਼ਨ ਦਿੱਤਾ ਗਿਆ ਹੈ। ਤੁਹਾਨੂੰ ਸਿਰਫ਼ ਸਹੀ ਜਵਾਬ ਦੀ ਗਿਣਤੀ ਨਾਲ ਜਵਾਬ ਦੇਣਾ ਚਾਹੀਦਾ ਹੈ, ਬਿਨਾਂ ਕਿਸੇ ਵਾਧੂ ਪਾਠ ਜਾਂ ਵਿਆਖਿਆ ਦੇ।",
    "sk": "Máte zadanú otázku s viacerými možnosťami odpovede. MUSÍTE odpovedať iba číslom správnej odpovede, bez akéhokoľvek ďalšieho textu alebo vysvetlenia.",
    "pl": "Otrzymujesz pytanie wielokrotnego wyboru do odpowiedzi. MUSISZ odpowiedzieć tylko numerem poprawnej odpowiedzi, bez żadnego dodatkowego tekstu ani wyjaśnień.",
    "cs": "Dostanete otázku s výběrem odpovědí. MUSÍTE odpovědět pouze číslem správné odpovědi, bez jakéhokoliv dalšího textu nebo vysvětlení.",
    "de": "Ihnen wird eine Multiple-Choice-Frage zur Beantwortung gegeben. Sie DÜRFEN nur mit der richtigen Nummer der Antwort antworten, ohne zusätzlichen Text oder Erklärungen.",
    "vi": "Bạn được cung cấp một câu hỏi trắc nghiệm để trả lời. Bạn PHẢI chỉ trả lời bằng số đúng của câu trả lời, không thêm bất kỳ văn bản hoặc giải thích nào.",
    "ne": "तपाईंलाई उत्तर दिनको लागि एक बहुविकल्पीय प्रश्न दिइएको छ। तपाईंले केवल सही उत्तरको नम्बरले मात्र उत्तर दिनुपर्छ, कुनै अतिरिक्त पाठ वा व्याख्या बिना।",
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
            Path(model_path) / "processor", 
            local_files_only=True
        )
        print(f"Model loaded from {model_path}")

    elif model_name =="qwen2.5-7b":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            Path(model_path) / "model", #"Qwen/Qwen2.5-VL-7B-Instruct" 
            temperature=TEMPERATURE,
            device_map=device,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(
            Path(model_path) / "processor", #"Qwen/Qwen2.5-VL-7B-Instruct"
            local_files_only=True
        )

    elif model_name == "pangea":
        # Add Pangea initialization logic
        raise NotImplementedError(f"Model: {model_name} not available yet")
    elif model_name in ["gpt-4o", "gpt-4o-mini"]:

        client = OpenAI(api_key=api_key)
        model = client
        processor = None

    elif model_name == 'gemini-2.0-flash-exp':

        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        model = client
        processor = None

    elif model_name == 'claude-3-5-sonnet-latest':
        client = Anthropic(api_key=api_key)
        model = client
        processor = None

    elif model_name == 'molmo':

        processor = AutoProcessor.from_pretrained(
            Path(model_path) / "processor", # 'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto',
            local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            Path(model_path) / "model", #'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            temperature=TEMPERATURE,
            device_map=device,
            torch_dtype=torch.bfloat16,
            temperature= TEMPERATURE,
            local_files_only=True
        )
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
    images,
    device: str = "cuda",
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
):
    """
    Query the model based on the model name.
    """
    if model_name == "qwen2-7b": # ERASE: should erase after 2.5 works well
        return query_qwen2(model, processor, prompt, images, device)

    elif model_name == 'qwen2.5-7b':
        return query_qwen25(model, processor, prompt, device)
    elif model_name == "pangea":
        # Add pangea querying logic
        raise NotImplementedError(f"Model {model_name} not implemented for querying.")
    elif model_name == "molmo":
        # Add molmo querying logic
        query_molmo(model, processor, prompt, images)
    elif model_name in ['gpt-4o', 'gpt-4o-mini', 'gemini-2.0-flash-exp']:
        return query_openai(model, model_name, prompt, temperature, max_tokens)
    elif model_name == 'claude-3-5-sonnet-latest':
        return query_anthropic(model, model_name, prompt, temperature, max_tokens)
    elif model_name == "maya":
        # Add Maya-specific parsing
        raise NotImplementedError(f"Model {model_name} not implemented for querying.")
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def query_pangea():
    pass


def query_molmo(model,
    processor,
    prompt: list,
    image_path: list,
):
    if prompt == 'multi-image':
        print('Question was multi-image, molmo does not support multi-image inputs.')
        return 'multi-image detected'
    else:
        inputs = processor.process(
            images=[Image.open(image_path).convert("RGB")],
            text=prompt
        )
        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return format_answer(generated_text)




def query_openai(client, model_name, prompt, temperature, max_tokens):
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_text = response.choices[0].message.content.strip()
    return format_answer(output_text)

def query_anthropic(client, model_name, prompt, temperature, max_tokens):

    system_message = prompt[0]['content']
    user_messages = prompt[1]

    response = client.messages.create(
        model=model_name,
        messages=[user_messages],
        system=system_message,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_text = response.content[0].text
    return format_answer(output_text)

# ERASE: should erase after 2.5 works well
def query_qwen2(
    model,
    processor,
    prompt: list,
    image_paths: list,
    device="cuda",
    max_tokens=MAX_TOKENS
):
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    text_prompt = processor.apply_chat_template(prompt, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Generate response
    with autocast("cuda"):
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    for img in images:
        img.close()
    torch.cuda.empty_cache()
    return format_answer(response[0])

def query_qwen25(
    model,
    processor,
    prompt: list,
    device="cuda",
    max_tokens=MAX_TOKENS
):
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
    with autocast("cuda"):
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return format_answer(response[0])


def generate_prompt(
    model_name: str,
    question: dict,
    lang: str,
    system_message,
    few_shot_setting: str = "zero-shot",
):
    if model_name == "qwen2-7b": # ERASE: should erase after 2.5 works well
        return parse_qwen2_input(
            question["question"],
            question["image"],
            question["options"],
            lang,
            system_message,
            few_shot_setting,
        )
    elif model_name == 'qwen2.5-7b':
        return parse_qwen25_input(
            question["question"],
            question["image"],
            question["options"],
            lang,
            system_message,
            few_shot_setting
        )
    elif model_name in ['gpt-4o', 'gpt-4o-mini', 'gemini-2.0-flash-exp']:
        return parse_openai_input(
            question["question"],
            question["image"],
            question["options"],
            lang,
            system_message,
            few_shot_setting,
        )
    elif model_name == "maya":
        # Add Maya-specific parsing
        raise NotImplementedError(f"Model {model_name} not implemented for parsing.")
    elif model_name == "pangea":
        # Add pangea querying logic
        raise NotImplementedError(f"Model {model_name} not implemented for parsing.")
    elif model_name == "molmo":
        # Add molmo querying logic
        return parse_molmo_inputs(
            question["question"],
            question["image"],
            question["options"],
            lang,
            system_message,
            few_shot_setting
        )
    elif model_name == 'claude-3-5-sonnet-latest':
        return parse_anthropic_input(
            question["question"],
            question["image"],
            question["options"],
            lang,
            system_message,
            few_shot_setting,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def parse_openai_input(
    question_text, question_image, options_list, lang, system_message, few_shot_setting
):
    """
    Outputs: conversation dictionary supported by OpenAI.
    """
    system_message = {"role": "system", "content": system_message}

    def encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                binary_data = image_file.read()
                base_64_encoded_data = base64.b64encode(binary_data)
                base64_string = base_64_encoded_data.decode('utf-8')
                return base64_string
        except Exception as e:
            raise TypeError(f"Image {image_path} could not be encoded. {e}")

    if question_image:
        base64_image = encode_image(question_image)
        question = [
            {
                "type": "text", 
                "text": question_text
            },
            {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        ]
    else:
        question = [{"type": "text", "text": question_text}]

    # Parse options. Handle options with images carefully by inclusion in the conversation.
    parsed_options = [{"type": "text", "text": "Options:\n"}]
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

    user_text = question + parsed_options
    user_message = {"role": "user", "content": user_text}

    # Enable few-shot setting
    if few_shot_setting == "few-shot":
        user_message["content"] = fetch_few_shot_examples(lang) + user_message["content"]
        messages = [system_message, user_message]
    elif few_shot_setting == "zero-shot":
        messages = [system_message, user_message]
    else:
        raise ValueError(f"Invalid few_shot_setting: {few_shot_setting}")

    return messages, None #image paths not expected for openai client.

def parse_anthropic_input(
    question_text, question_image, options_list, lang, system_message, few_shot_setting
):
    """
    Outputs: conversation dictionary supported by Anthropic.
    """
    # review, claude-3-5-sonnet-latest might not support conversation items inside content
    system_message = {"role": "system", "content": system_message}

    def encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                binary_data = image_file.read()
                base_64_encoded_data = base64.b64encode(binary_data)
                base64_string = base_64_encoded_data.decode('utf-8')
                return base64_string
        except Exception as e:
            raise TypeError(f"Image {image_path} could not be encoded. {e}")


    if question_image:
        base64_image = encode_image(question_image)
        question = [
            {
                "type": "text", 
                "text": question_text
            },
            {
                "type": "image", 
                "source": 
                    {"type": "base64", 
                    "media_type": "image/png", 
                    "data": base64_image}
            }
        ]
    else:
        question = {"type": "text", "text": question_text}

    # Parse options. Handle options with images carefully by inclusion in the conversation.
    parsed_options = [{"type": "text", "text": "Options:\n"}]
    only_text_option = {"type": "text", "text": "{text}"}
    only_image_option = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,{base64_image}"},
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

    user_text = question + parsed_options
    user_message = {"role": "user", "content": user_text}

    # Enable few-shot setting
    if few_shot_setting == "few-shot":
        user_message["content"] = fetch_few_shot_examples(lang) + user_message["content"]
        messages = [system_message, user_message]
    elif few_shot_setting == "zero-shot":
        messages = [system_message, user_message]
    else:
        raise ValueError(f"Invalid few_shot_setting: {few_shot_setting}")

    return messages, None #image paths not expected for openai client.

def parse_molmo_inputs(question_text, question_image, options_list, lang, instruction, few_shot_setting):
  for option in options_list:
    if '.png' in option:
      return 'multi-image', None

  prompt = instruction + '\n\n'
  question = 'Question: ' + question_text + '\n\n'

  options = 'Options:\n'
  for i, option in enumerate(options_list):
    options += chr(65+i) + '. ' + option + '\n'

  prompt += question + options + '\nAnswer:'
  return prompt, question_image

def parse_qwen25_input(
    question_text, question_image, options_list, lang, system_message, few_shot_setting
):
    """
    Outputs: conversation dictionary supported by qwen2.5 .
    """
    system_message = {"role": "system", "content": system_message}

    if question_image:
        question = [
            {
                "type": "text",
                "text": f"Question: {question_text}"
            },
            {
                "type": "image",
                "image": f"file:///{question_image}"
            }
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
        option_indicator = f"{i+1})"
        if option.lower().endswith(".png"):  # Checks if it is a png file
            # Generating the dict format of the conversation if the option is an image
            new_image_option = only_image_option.copy()
            new_image_option["image"] = new_image_option["image"].format(image_path=option)
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
        user_message["content"] = fetch_few_shot_examples(lang) + user_message["content"]
        messages = [system_message, user_message]
    elif few_shot_setting == "zero-shot":
        messages = [system_message, user_message]
    else:
        raise ValueError(f"Invalid few_shot_setting: {few_shot_setting}")

    return messages, None #image paths processed in messages by process_vision_info.      


# ERASE: should erase after 2.5 works well
def parse_qwen2_input(
    question_text, question_image, options_list, lang, system_message, few_shot_setting
): 
    """
    Outputs: conversation dictionary supported by qwen.
    """
    system_message = {"role": "system", "content": system_message}

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

    user_text = question + parsed_options
    user_message = {"role": "user", "content": user_text}

    # Enable few-shot setting
    if few_shot_setting == "few-shot":
        user_message["content"] = fetch_few_shot_examples(lang) + user_message["content"]
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
    match = re.search(pattern, answer)

    if match:
        letter = match.group(1).upper()  #
        return ord(letter) - ord('A')
    else:
        return None


def fetch_cot_instruction(lang: str) -> str:
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
