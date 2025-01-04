import base64

temperature= 0
max_tokens = 1 # Only output the option chosen.

SUPPORTED_MODELS = ['gpt-4o', 'qwen', 'maya']

SYSTEM_MESSAGE = "You are given a multiple choice question for answering. You MUST only answer with the correct number of the answer. For example, if you are given the options 1, 2, 3, 4 and option 2 (respectively B) is correct, then you should return the number 2. \n"

def parse_openai_input(question_text, question_image, options_list):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    question = [{
        'type': 'text', 'text': question_text
    }]

    if question_image: 
        base64_image = encode_image(question_image)
        question_image_message= {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }
        question.append(question_image_message)
    
    # Parse options. Handle options with images carefully by inclusion in the conversation.
    parsed_options = []
    only_text_option = {
            "type":"text",
            "text":"{text}"
        }
    only_image_option = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}
        }

    for i, option in enumerate(options_list):
        option_indicator = f"{i+1})"
        if option.lower().endswith('.png'):
        # Generating the dict format of the conversation if the option is an image
            new_image_option = only_image_option.copy()
            new_text_option = only_text_option.copy()
            formated_text = new_text_option['text'].format(text=option_indicator + '\n')
            new_text_option['text'] = formated_text

            parsed_options.append(new_text_option)
            parsed_options.append(new_image_option['image_url']['url'].format(base64_image=encode_image(option)))

        else:
        # Generating the dict format of the conversation if the option is not an image
            new_text_option = only_text_option.copy()
            formated_text = new_text_option['text'].format(text=option_indicator + option + '\n')
            new_text_option['text'] = formated_text
            parsed_options.append(new_text_option)

    return question, parsed_options

def parse_qwen_input(question_text, question_image, options_list):
    if question_image:
        question = [
                {
                    "type":"text",
                    "text":f"Question: {question_text}",
                },
                {
                    "type":"image"
                }
            ]
    else:
        question = [
            {
                "type":"text",
                "text":f"Question: {question_text}",
            }
        ]

    parsed_options = []
    only_text_option = {
            "type":"text",
            "text":"{text}"
        }
    only_image_option = {
            "type":"image"
        }

    images_paths = []
    for i, option in enumerate(options_list):
        option_indicator = f"{i+1})"
        if option.lower().endswith('.png'): # Checks if it is a png file
        # Generating the dict format of the conversation if the option is an image
            new_image_option = only_image_option.copy()
            new_text_option = only_text_option.copy()
            formated_text = new_text_option['text'].format(text=option_indicator + '\n')
            new_text_option['text'] = formated_text

            # option delimiter "1)", "2)", ...
            parsed_options.append(new_text_option)
            # image for option
            parsed_options.append(new_image_option)

            # Ads the image for output
            images_paths.append(option)

        else:
        # Generating the dict format of the conversation if the option is not an image
            new_text_option = only_text_option.copy()
            formated_text = new_text_option['text'].format(text=option_indicator + option + '\n')
            new_text_option['text'] = formated_text
            parsed_options.append(new_text_option) # Puts the option text if it isn't an image.

        prompt_text = [question] + parsed_options

        if question_image:
            image_paths = [question_image] + image_paths

        return prompt_text, images_paths
    
def format_answer(answer: str):
        """
        Returns: A zero-indexed integer corresponding to the answer.
        """
        if not isinstance(answer, str) or len(answer) != 1:
            raise ValueError(f"Invalid input: '{answer}'. Expected a single character string.")

        if 'A' <= answer <= 'Z':
            # Convert letter to zero-indexed number
            return ord(answer) - ord('A')
        elif '1' <= answer <= '9':
            # Convert digit to zero-indexed number
            return int(answer) - 1
        else:
            raise ValueError(f"Invalid answer: '{answer}'. Must be a letter (A-Z) or a digit (1-9).")