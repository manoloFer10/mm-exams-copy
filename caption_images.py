import pandas as pd
import base64
import json
import os
import argparse 
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Users\shepe\AppData/Local/Programs/Tesseract-OCR/tesseract.exe'
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from datasets import load_dataset
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="dir to resume from",
    )
    args = parser.parse_args()
    return args

def get_token(provider):
    with open('tokens_mm_exams.json', 'r') as f:
        keys = json.load(f)
    
    return keys[provider]

def load_image_dataset():
    print('Logging in...')
    login(get_token('huggingface'))
    def multimodal_only(example):
        return example["image"]
    
    dataset = load_dataset('CohereForAI/mumu-exams-stratiefied')['train']
    dataset = dataset.filter(multimodal_only)
    print(f'Final dataset length: {len(dataset)}')

    return dataset

def instantiate_captioner(token):
    client = OpenAI(
            api_key=token,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    return client

def generate_captioning_prompt(image_path):
    def encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                binary_data = image_file.read()
                base_64_encoded_data = base64.b64encode(binary_data)
                base64_string = base_64_encoded_data.decode("utf-8")
                return base64_string
        except Exception as e:
            raise TypeError(f"Image {image_path} could not be encoded. {e}")
        
    system_message ='''**Instruction:**You are an expert image captioner. Generate highly detailed, precise, and academically relevant textual descriptions of images sourced from exam questions, ensuring all critical visual elements are captured for accurate problem-solving.

**Guidelines:**

Exam-Specific Analysis:

- Primary Elements: Identify and describe key components (e.g., diagrams, charts, graphs, labels, symbols, annotations) and their exact attributes (e.g., numerical values, units, directional arrows, text annotations).

- Secondary Details: Note stylistic features (e.g., "black-and-white schematic," "color-coded bars in a graph"), spatial relationships (e.g., "force vectors pointing northwest"), and contextual clues (e.g., axes labels, legends, scales).

- Textual Elements: Explicitly transcribe all visible text (e.g., labels like "Mitochondria," numbers like "5Ω," titles like "Figure 2: Velocity vs. Time").

Academic Precision:

- Technical Focus: Prioritize details critical to exam questions (e.g., "a right triangle with hypotenuse labeled c = 10 cm," "a bar graph comparing GDP of 5 countries, with Japan’s bar shaded blue at 4.3 trillion").

- Diagrams/Charts: Specify type (e.g., "pie chart," "circuit diagram") and components (e.g., "resistor symbol connected to a battery").

- Scientific Relevance: Highlight measurements, units, symbols (e.g., "ΔT = 25°C," "a pulley system with frictionless ropes").

Structure & Clarity:

- Begin with the image’s purpose (e.g., "A biology diagram of a plant cell") followed by a systematic breakdown (left-to-right, top-to-bottom, or by functional layers).

- Use neutral, objective language. Avoid assumptions unless implied by context (e.g., "a downward arrow labeled 9.8 m/s² likely representing gravitational acceleration").

Output Format:

- Single paragraph (4–6 sentences).

- Example:
"A physics diagram depicts two blocks on a frictionless inclined plane: Block A (5 kg) is connected via a rope to Block B (3 kg) over a pulley. Angle θ = 30°, with vectors labeled F_normal and F_gravity. A scale beside the plane shows time t = 0s to t = 5s. Text at the bottom reads: ‘Calculate tension in the rope.’ The image is monochrome, with dashed lines indicating motion direction."

Constraints:

- Avoid Omissions: Ensure no labels, numbers, or symbols are overlooked, even if small or peripheral.

- Neutral Tone: Exclude subjective interpretations (e.g., "messy handwriting" or "complex diagram") unless style is exam-relevant (e.g., "a hand-drawn sketch with annotations").'''

    prompt=[
        {'role': 'system', 'content':system_message},
        {'role': 'user', 'content': 
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                        "detail": "low"
                        }
                    }
                ]
        }
    ]

    return prompt

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

def image_to_text(image_path):
    try:
        image = Image.open(image_path)
    except IOError as e:
        print(f'Error decoding {image_path}')
        raise e
    
    # Perform OCR using Tesseract via pytesseract
    text = pytesseract.image_to_string(image)
    
    return text

def main():
    args = parse_args()
    dataset = load_image_dataset()

    client = instantiate_captioner(get_token('google'))

    if args.resume:
        output_path = args.resume
        with open(output_path, "r") as f:
            results = json.load(f)
            continue_from = len(results)
            print(f'Continuing from {continue_from}')
        
    else:
        output_folder = f"outputs/captions/gemini_1.5_pro"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"captions_ocr.json")
        continue_from = 0
        results = []

    temperature = 0.7
    max_tokens = 1024

    for t, question in tqdm(enumerate(dataset), total=len(dataset)):
        if t < continue_from:
            continue

        image_path = question['image']

        prompt = generate_captioning_prompt(image_path)
        caption = query_openai(client, 'gemini-1.5-pro', prompt, temperature, max_tokens)

        question["image_caption"] = caption
        question["image_ocr"] = image_to_text(image_path)

        result_metadata = question.copy()
        results.append(result_metadata)

        if (t + 1) % 100 == 0 or (t + 1) == len(dataset):
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Ongoing {t} results saved to {output_path}")

    # Save results to file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Captioning completed. Results saved to {output_path}")

if __name__ == '__main__':
    main()

