import os

import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure_cache(enable_disk_cache=False)


def _azure_deployment_id(overrides: dict) -> str:
    """LiteLLM/Azure model id: first non-empty from overrides or env."""
    for key in ("model", "deployment"):
        v = overrides.get(key)
        if v:
            return str(v).removeprefix("azure/")
    for env in (
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_TARGET_MODEL",
        "DSPY_MODEL",
        "AZURE_MODEL",
    ):
        v = os.getenv(env)
        if v:
            return v.strip().removeprefix("azure/")
    return ""


def get_lm(overrides: dict | None = None) -> dspy.LM:
    """DSPy language model pointing at Azure OpenAI (env-driven)."""
    overrides = overrides or {}
    deployment = _azure_deployment_id(overrides)
    if not deployment:
        raise RuntimeError(
            "Set AZURE_OPENAI_DEPLOYMENT or AZURE_TARGET_MODEL (or DSPY_MODEL / AZURE_MODEL) "
            "for DSPy Azure calls."
        )
    model = overrides.get("model")
    if not model:
        model = deployment if deployment.startswith("azure/") else f"azure/{deployment}"
    api_version = overrides.get("api_version") or os.getenv("AZURE_API_VERSION") or "2024-12-01-preview"
    return dspy.LM(
        model=model,
        api_key=overrides.get("api_key", os.getenv("AZURE_SUBSCRIPTION_KEY")),
        api_base=overrides.get("api_base", os.getenv("AZURE_ENDPOINT")),
        api_version=api_version,
        provider=overrides.get("provider", "azure"),
        cache=overrides.get("cache", False),
    )
