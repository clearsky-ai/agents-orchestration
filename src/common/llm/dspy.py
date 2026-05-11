import os

import dspy

dspy.configure_cache(enable_disk_cache=False)


def get_lm(overrides: dict = None):  # TODO: check if we need to make this singleton

    overrides = overrides or {}
    return dspy.LM(
        model=overrides.get("model", f"azure/{os.getenv('DSPY_MODEL')}"),
        api_key=overrides.get("api_key", os.getenv("AZURE_SUBSCRIPTION_KEY")),
        api_base=overrides.get("api_base", os.getenv("AZURE_ENDPOINT")),
        api_version=overrides.get("api_version", os.getenv("AZURE_API_VERSION")),
        provider=overrides.get("provider", "azure"),
        cache=overrides.get("cache", False),
    )
