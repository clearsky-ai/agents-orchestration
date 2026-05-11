import os
from typing import Literal, cast

from dotenv import load_dotenv

load_dotenv()


def get_azure_lm(cached: bool = True, overrides: dict = None):
    from autogen_ext.cache_store.redis import RedisStore
    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
    from autogen_ext.models.cache import CHAT_CACHE_VALUE_TYPE, ChatCompletionCache

    cache_store = None
    overrides = overrides or {}
    if cached:
        import redis
        import certifi

        redis_instance = redis.Redis(
            host=overrides.get("AZURE_REDIS_HOST", os.getenv("AZURE_REDIS_HOST")),
            port=int(overrides.get("AZURE_REDIS_PORT", os.getenv("AZURE_REDIS_PORT"))),
            password=overrides.get(
                "AZURE_REDIS_PASSWORD", os.getenv("AZURE_REDIS_PASSWORD")
            ),
            username=overrides.get(
                "AZURE_REDIS_USERNAME", os.getenv("AZURE_REDIS_USERNAME")
            ),
            ssl=True,
            ssl_ca_certs=certifi.where(),
            decode_responses=True,
        )
        cache_store = RedisStore[CHAT_CACHE_VALUE_TYPE](redis_instance)

    return ChatCompletionCache(
        AzureOpenAIChatCompletionClient(
            azure_deployment=overrides.get("AZURE_MODEL", os.getenv("AZURE_MODEL")),
            api_version=overrides.get(
                "AZURE_API_VERSION",
                os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            ),
            azure_endpoint=overrides.get("AZURE_ENDPOINT", os.getenv("AZURE_ENDPOINT")),
            api_key=overrides.get(
                "AZURE_SUBSCRIPTION_KEY", os.getenv("AZURE_SUBSCRIPTION_KEY")
            ),
            model=overrides.get("AZURE_MODEL", os.getenv("AZURE_MODEL")),
            model_info={
                "vision": overrides.get("IS_VISION_MODEL", False),
                "function_calling": overrides.get("IS_FUNCTION_CALLING", True),
                "json_output": overrides.get("IS_JSON_OUTPUT", False),
                "family": overrides.get("FAMILY", "unknown"),
                "structured_output": overrides.get("IS_STRUCTURED_OUTPUT", True),
            },
            # reasoning_effort=cast(
            #     Literal["low", "medium", "high"] | None,
            #     overrides.get(
            #         "REASONING_EFFORT", os.getenv("REASONING_EFFORT", "medium")
            #     ),
            # ),
        ),
        cache_store,
    )
