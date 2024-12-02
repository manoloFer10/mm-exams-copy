#----------------------------------------------------------------------------------------------------------------------------------
# CURRENT MODELS: 
#   Gemini 1.5,
#   GPT-4o, 
#   Llama 3.1, 
#   Claude 3.5
#
# Comment: We should decide which models to include. 
#----------------------------------------------------------------------------------------------------------------------------------

#TODO: should we start thinking about few shot/ instruction prompting? Zero shot is only implemented for the moment.


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser


class Replier:
    def __init__(self, name:str):
        self.replier = self._call_model(name)
        self.parser = StrOutputParser()

    def _call_model(self, model:str):

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
                        groq_api_key="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",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )
        if model not in {'llama', 'chatgpt', 'claude', 'gemini', 'test'}:
            raise ValueError('Model not implemented for answering DB.')
        
        return replier

    def predict_zero_shot(self,question: str, options: list[str]):
        options_txt = self._parse_options(options)

        prompted_q = [
            (
                "system",
                "You are given a multiple choice question for answering. You MUST only answer with the correct number of the answer."
                "For example, if you are given the options 1, 2, 3, 4 and option 2 is correct, then you should return the number 2.",
            ),
            (
                "human", 
                f"Question: {question} /n"
                f"Options: /n {options_txt}"
            ),
        ]

        answer = self.parser.invoke(self.replier.invoke(prompted_q))
        answer = int(answer) # Is there anything better for formatting ? 

        return answer - 1 # For answer comparison between options['answer'] which is zero-indexed.

    def _parse_options(self, options: list[str]):
        parsed_options = "\n".join(f"{i + 1}) {option}" for i, option in enumerate(options))
        return parsed_options