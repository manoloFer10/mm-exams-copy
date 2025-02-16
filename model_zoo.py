from model_utils import INSTRUCTIONS_COT

# Molmo

# Pangea


# Qwen2
def create_qwen2_prompt(question, method):
    content = []
    lang = question["language"]
    prompt = [INSTRUCTIONS_COT[lang]]
    if question["image"] is not None:
        content.append(
            {
                "type": "image",
                "image": question["image"],
                "resized_height": 256,
                "resized_width": 256,
            }
        )
    if method == "zero-shot":
        prompt.append(f"\n{question['question']} Options: \n")
        for t, option in enumerate(question["options"]):
            index = f"{chr(65+t)}. "
            prompt.append(f"{index}) {option}\n")
    content.append({"type": "text", "text": "".join(prompt)})
    message = {"role": "user", "content": content}
    return message, [question["image"]]


# GPT

# Gemini

# Claude
