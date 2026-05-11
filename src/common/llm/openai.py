import os
from typing import Union
from openai import AzureOpenAI
from src.primitives.singleton_metaclass import SingletonMetaClass


class OpenAiClient(metaclass=SingletonMetaClass):
    def __init__(self):
        # Initialize your singleton instance variables here

        self.target_model = os.getenv(
            "AZURE_TARGET_MODEL", os.getenv("OPENAI_MODEL", None)
        )
        assert (
            self.target_model is not None
        ), "Target model must be specified in environment variables."

        print(f"Using target model: {self.target_model}")

        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_SUBSCRIPTION_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )

    def make_request(
        self,
        system_prompt: str,
        user_prompt: Union[str, list[str]],
        return_content_only: bool = True,
    ):

        if isinstance(user_prompt, str):
            user_prompt = [user_prompt]

        messages = [{"role": "system", "content": system_prompt}]

        for prompt in user_prompt:
            messages.append({"role": "user", "content": prompt})

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.target_model,
        )

        if return_content_only:
            return chat_completion.choices[0].message.content.strip()
        return chat_completion.choices[0].message


# Usage:
# instance1 = OpenARequests()
# instance2 = OpenARequests()
# assert instance1 is instance2
