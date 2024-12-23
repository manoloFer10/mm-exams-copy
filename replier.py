#----------------------------------------------------------------------------------------------------------------------------------
# CURRENT MODELS: 
#   Gemini 1.5,
#   GPT-4o, 
#   Llama 3.1, 
#   Claude 3.5,
#   Qwen2-VL-7B-Instruct
#
# Comment: We should decide which models to include. 
#----------------------------------------------------------------------------------------------------------------------------------

#TODO: should we start thinking about few shot/ instruction prompting? Zero shot is only implemented for the moment.


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
import modelWrapper as extras
import warnings

class Replier:
    def __init__(self, name:str):
        self.replier = self._call_model(name)
        self.parser = StrOutputParser()

    def _call_model(self, model:str):
        """Instantiates the replier in supported models."""

        # All models are instantiated with temperature = 0 for more a controlled format in answers.
        if model == 'gemini':
            replier = ChatGoogleGenerativeAI(
                        model="gemini-1.5-pro",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                        google_api_key='GOOGLE_API_KEY'
                    )

        if model == 'chatgpt':
            replier = ChatOpenAI(
                        model='gpt-4o',
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                        api_key='OPENAI_API_KEY'
                    )

        if model == 'claude':
            replier = ChatAnthropic(
                        model='model',
                        temperature=0,
                        max_tokens=None,
                        max_retries=2,
                        timeout=None,
                        api_key='ANTHROPIC_API_KEY'
                    )
        if model == 'llama':
            replier = ChatGroq(
                        model="llama-3.1-8b-instant",
                        groq_api_key="GROQ_API_KEY",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )
        if model == 'test':
            replier = ChatGroq(
                        model="llama-3.1-8b-instant",
                        groq_api_key="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", #Watchout.
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )
        if model == 'qwen':

            warnings.warn("Warning, you are about to load a model locally.")

            # Particular exception for Qwen (or any HuggingFace model) handling.
            qwen = AutoModelForVision2Seq.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16, device_map="auto"
                )
            qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

            replier = extras.ModelWrapper(qwen, qwen_processor)

        if model not in {'llama', 'chatgpt', 'claude', 'gemini','qwen', 'test'}:
            raise ValueError('Model not implemented for answering DB.')
        
        return replier

    def predict_zero_shot(self,question: str, question_image: str, options: list[str], index: int, file_path:str):
        """
        Predicts the MCQ answer with a zero-shot prompt. 
        """

        # Parse the question and options text into a LangChain conversation for unification purposes between models. 
        prompt, images = self._structure_prompt(question, question_image, options)

        try:
            answer = self.parser.invoke(self.replier.invoke(prompt, images))
            answer = self._format_answer(answer)

            return answer
        
        except Exception as e:
            print(f'Could not parse the answer to the question of index N°{index} in file: {file_path}. \n Due to {e}')

    def _structure_prompt(self, question: str, question_image: str, options: list[str]):
        """
        Structures the prompt into conversation format for MCQ answering.

        TODO: Upgrade the prompt.
        """

        question, options, image_paths = self._parse_options_into_chat_template(question, question_image, options)

        system_message = [
            {"role": "system",
            "content":
                    "You are given a multiple choice question for answering. You MUST only answer with the correct number of the answer. "
                    "For example, if you are given the options 1, 2, 3, 4 and option 2 (respectively B) is correct, then you should return the number 2. \n"
                    }]

        user_message= {"role": "user",
                        "content":[question] + options}

        complete_message = system_message + [user_message]

        return complete_message, image_paths
    
    def _parse_options_into_chat_template(self, question: str, question_image: str, options: list[str]):
        """
        Structures the question and options list into a conversation supported by LangChain.
        """
        # Parse the question. If there is an image in the question, then it should be included in the conversation
        # schema.
        if question_image:
            question = [
                {
                    "type":"text",
                    "text":f"Question: {question}",
                },
                {
                    "type":"image"
                }
            ]
        else:
            question = [
                {
                    "type":"text",
                    "text":f"Question: {question}",
                }
            ]

        # Parse options. Handle options with images carefully by inclusion in the conversation.
        parsed_options = []
        only_text_option = {
                "type":"text",
                "text":"{text}"
            }
        only_image_option = {
                "type":"image"
            }

        images_paths = []
        for i, option in enumerate(options):
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

        # Correct list of image paths (ordered)
        if question_image:
            image_paths = [question_image] + image_paths

        return question, parsed_options, images_paths
    
    def _format_answer(self, answer: str):
        # Note: the model only sees "1)", "2)" ... Might be unuseful.
        """
        Converts a multiple-choice answer (e.g., 'A', 'B', '1', '2') into a zero-indexed number.
        For answer comparison between options['answer'] which is zero-indexed.

        Parameters:
        answer (str): The answer option (e.g., 'A', 'B', ..., '1', '2').

        Returns:
        int: A zero-indexed integer corresponding to the answer.

        Raises:
        ValueError: If the answer is not a valid option.
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