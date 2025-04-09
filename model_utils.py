import torch
import re
from transformers import (
    AutoProcessor,
    GenerationConfig,
    LlavaNextForConditionalGeneration,
)

from vllm import LLM, SamplingParams

from pathlib import Path
from PIL import Image
from openai import OpenAI
from anthropic import Anthropic
from cohere import ClientV2

from tenacity import retry, stop_after_attempt, wait_exponential

from model_zoo import (
    create_pangea_prompt,
    create_qwen_prompt_vllm,
    create_aya_prompt,
    create_molmo_prompt_vllm,
    create_anthropic_prompt,
    create_openai_prompt
)


TEMPERATURE = 0.7
MAX_TOKENS = 1024

SUPPORTED_MODELS = [
    "gpt-4o",
    "qwen2.5-72b",
    "qwen2.5-32b",
    "qwen2.5-7b",
    "qwen2.5-3b",
    "gemini-1.5-pro",
    "claude-3-5-sonnet-latest",
    "molmo",
    "aya-vision",
] 


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
    model_name: str,
    model_path: str,
    api_key: str = None,
    device: str = "cuda",
    ngpu=1,
):
    """
    Initialize the model and processor/tokenizer based on the model name.
    """
    if model_name in ["qwen2.5-7b", "qwen2.5-3b", "qwen2.5-32b", "qwen2.5-72b"]:
        model = LLM(
            model_path,
            tensor_parallel_size=ngpu,
            max_model_len=8192,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=True,
        )

    elif model_name == "molmo":
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
        )
        processor = None

    elif model_name == "pangea":
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        processor.patch_size = 14
        model.resize_token_embeddings(len(processor.tokenizer))
    elif model_name == "gpt-4o":
        model = OpenAI(api_key=api_key)
        processor = None
    elif model_name == "gemini-1.5-pro":
        model = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        processor = None
    elif model_name == "claude-3-5-sonnet-latest":
        model = Anthropic(api_key=api_key)
        processor = None
    elif model_name == "aya-vision":
        model = ClientV2(api_key=api_key)
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
    if model_name in [
        "qwen2-7b",
        "qwen2.5-72b",
        "qwen2.5-32b",
        "qwen2.5-7b",
        "qwen2.5-3b",
    ]:
        answer = query_vllm(model, processor, prompt, images, max_tokens)
    elif model_name == "pangea":
        answer = query_pangea(model, processor, prompt, images, device)
    elif model_name == "molmo":
        answer = query_vllm(model, processor, prompt, images, max_tokens)
    elif model_name == "aya-vision":
        answer = query_aya(model, prompt, 0.3, 1024)
    elif model_name in [
        "gpt-4o",
        "gemini-1.5-pro",
    ]:
        answer = query_openai(model, model_name, prompt, temperature, max_tokens)

    elif model_name == "claude-3-5-sonnet-latest":
        answer = query_anthropic(model, model_name, prompt, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return answer, None  
  
  
def query_molmo(model, processor, prompt: list, images: list, max_tokens):
    if prompt == "multi-image":
        print("Question was multi-image, molmo does not support multi-image inputs.")
        return "multi-image detected"
    else:
        if images is not None:
            try:
                images = [Image.open(images).convert("RGB").resize((224, 224))]
            except:
                print(images)
                images = None
        inputs = processor.process(
            images=images,
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

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_openai(client, model_name, prompt, temperature, max_tokens):
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_text = response.choices[0].message.content.strip()
    return output_text


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
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


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_aya(client, prompt, temperature, max_tokens):
    response = client.chat(
        model="c4ai-aya-vision-32b",
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_text = response.message.content[0].text
    return output_text


def query_vllm(model, processor, prompt, images, max_tokens=MAX_TOKENS):
    # Prepare the text prompt
    # text_prompt = processor.apply_chat_template(prompt, add_generation_prompt=True)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,  # Adjust as needed
        top_p=0.9,  # Adjust as needed
    )

    if images is not None:
        try:
            images = [Image.open(image).resize((512, 512)) for image in images]
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": images},
            }
        except:
            print(images)
            images = None
    else:
        inputs = {"prompt": prompt}

    # Generate response using vLLM
    with torch.inference_mode():
        outputs = model.generate(inputs, sampling_params=sampling_params)
        response = outputs[0].outputs[0].text

    return response


def query_pangea(
    model, processor, prompt, images, device="cuda", max_tokens=MAX_TOKENS
):
    if images is not None:
        try:
            images = Image.open(images).convert("RGB").resize((512, 512))
        except Exception as e:
            print("Failed to load image:", e)
            images = None

    model_inputs = processor(images=images, text=prompt, return_tensors="pt").to(
        "cuda", torch.float16
    )
    with torch.inference_mode():
        output = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=1.0,
            top_p=0.9,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        output = output[0]
    result = processor.decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return result


def generate_prompt(
    model_name: str,
    question: dict,
    lang: str,
    instruction,
    few_shot_samples: dict,
    method: str = "zero-shot",
    experiment: str = 'normal'
):

    if model_name in ["qwen2-7b", "qwen2.5-7b", "qwen2.5-72b", "qwen2.5-3b"]:
        return create_qwen_prompt_vllm(question, method, few_shot_samples, experiment)
    elif model_name == "molmo":
        return create_molmo_prompt_vllm(question, method, few_shot_samples)
    elif model_name == "pangea":
        return create_pangea_prompt(question, method, few_shot_samples)
    elif model_name == "aya-vision":
        return create_aya_prompt(question, method, few_shot_samples)
    elif model_name in ["gpt-4o", "gemini-1.5-pro"]:
        return create_openai_prompt(
            question,
            lang,
            instruction,
            method,
            experiment
        )
    elif model_name == "claude-3-5-sonnet-latest":
        return create_anthropic_prompt(
            question,
            lang,
            instruction,
            method,
        )
    else:
        raise ValueError(f"Unsupported model for parsing inputs: {model_name}")



# def extract_answer_from_tags(answer: str):
#     """
#     Searchs for the answer between tags <Answer>.

#     Returns: A zero-indexed integer corresponding to the answer.
#     """
#     pattern = r"<ANSWER>\s*([A-Za-z])\s*</ANSWER>"
#     match = re.search(pattern, answer, re.IGNORECASE)

#     if match:
#         # Extract and convert answer letter
#         letter = match.group(1).upper()
#         election = ord(letter) - ord("A")

#         # Extract reasoning by removing answer tag section
#         start, end = match.span()
#         reasoning = answer.strip()
#         # Clean multiple whitespace
#         reasoning = re.sub(r"\s+", " ", reasoning)
#     elif len(answer) == 1:
#         reasoning = answer
#         if "A" <= answer <= "Z":
#             # Convert letter to zero-indexed number
#             election = ord(answer) - ord("A")
#         elif "1" <= answer <= "9":
#             # Convert digit to zero-indexed number
#             election = int(answer) - 1
#         else:
#             election = answer
#     else:
#         # Error handling cases
#         election = "No valid answer tag found"
#         if re.search(r"<ANSWER>.*?</ANSWER>", answer):
#             election = "Answer tag exists but contains invalid format"
#         reasoning = answer.strip()

#     return reasoning, election


def fetch_cot_instruction(lang: str) -> str:
    """
    Retrieves the CoT Instruction for the given lang.
    """
    if lang in INSTRUCTIONS_COT.keys():
        return INSTRUCTIONS_COT[lang]
    else:
        raise ValueError(f"{lang} language code not in INSTRUCTIONS_COT")

