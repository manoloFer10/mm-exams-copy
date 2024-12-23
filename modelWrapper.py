from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult
from PIL import Image
import torch
from typing import List, Dict, Any, Union, Awaitable
import asyncio


class ModelWrapper(BaseLanguageModel):
    """
    Custom LangChain wrapper for supporting interleaved images (for the moment, Qwen2VL).
    Implements required synchronous and asynchronous methods.
    """

    model_config = {"model_private_attrs": {"_model", "_processor"}}

    def __init__(self, model, processor):

        # Initialize the parent class
        super().__init__()
        # Load model and processor
        self._model = model
        self._processor = processor

    def _generate(self, prompt: list[str], image_paths: list) -> str:
        """
        Generate response based on a prompt and a list of images.

        Args:
            prompt : The input chat processed in template format.
            image_paths (list): List of image file paths corresponding to <image> tokens.

        Returns:
            str: Generated model response.
        """
        #TODO: Agregar un handler por si no hay imagenes. directamente no pasarle el argumento images al processor.

        # Load images
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

        # Pass tokenized text and images to the processor
        inputs = self._processor(
            text=prompt,  # This will still align images with text
            images=images,
            return_tensors="pt",
            padding=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Debug inputs
        print("Processor Input IDs:", inputs["input_ids"])
        print("Processor Pixel Values Shape:", inputs["pixel_values"].shape)

        # Generate response
        output_ids = self._model.generate(**inputs, max_new_tokens=50)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        response = self._processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return response #Response is a list of len(prompt)

    async def apredict_messages(self, messages: List[Dict[str, str]], **kwargs) -> Awaitable[str]:
        """
        Asynchronous prediction method for message-based inputs.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries.
            kwargs: Additional keyword arguments including image_paths.

        Returns:
            Awaitable[str]: Generated model response.
        """
        return await asyncio.to_thread(self.predict_messages, messages, **kwargs)

    def predict_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Synchronous prediction method for LangChain compatibility.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries.
            kwargs: Additional keyword arguments, including image_paths.

        Returns:
            str: Generated model response.
        """

        # Preprocess the chat
        text_input = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        #Debug
        print('Preprocessed input: ', text_input)
        image_paths = kwargs.get("image_paths", [])
        return self._generate(text_input, image_paths)

    def agenerate_prompt(self, prompts: List[str], **kwargs) -> LLMResult:
        """
        Asynchronous batch generation method.

        Args:
            prompts (List[str]): List of input prompts.
            kwargs: Additional keyword arguments.

        Returns:
            LLMResult: LangChain-compatible result object.
        """
        raise NotImplementedError("Asynchronous generation is not supported yet.")

    def generate_prompt(self, prompts: List[str], **kwargs) -> LLMResult:
        """
        Synchronous batch generation method.

        Args:
            prompts (List[str]): List of input prompts.
            kwargs: Additional keyword arguments.

        Returns:
            LLMResult: LangChain-compatible result object.
        """
        outputs = [self._generate(prompt, kwargs.get("image_paths", [])) for prompt in prompts]
        return LLMResult(generations=[[{"text": output}] for output in outputs])

    def apredict(self, text: str, **kwargs) -> str:
        """
        Asynchronous predict method.

        Args:
            text (str): Input text.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated model response.
        """
        raise NotImplementedError("Asynchronous predict is not implemented yet.")

    def predict(self, text: str, **kwargs) -> str:
        """
        Synchronous predict method.

        Args:
            text (str): Input text with interleaved <image> tokens.

        Returns:
            str: Generated response.
        """
        image_paths = kwargs.get("image_paths", [])
        return self._generate(text, image_paths)

    def invoke(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        Unified invoke method for LangChain compatibility.

        Args:
            input_data (Union[str, Dict[str, Any]]): Input text or input dictionary.

        Returns:
            str: Generated response.

        #TODO: it should be modified in order to call this method for inference. For unification purposes.
        """
        if isinstance(input_data, str):
            return self.predict(input_data)
        elif isinstance(input_data, dict):
            text = input_data.get("text", "")
            image_paths = input_data.get("image_paths", [])
            return self._generate(text, image_paths)
        else:
            raise ValueError("Input must be a string or a dictionary.")

    def _llm_type(self) -> str:
        """
        Describes the type of model for LangChain introspection.
        """
        return "Qwen2-VL"
