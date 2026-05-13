"""Shared Entra ID auth helpers for Azure OpenAI / Foundry endpoints.

`exclude_environment_credential=True` skips AZURE_CLIENT_ID/SECRET env vars
(used by Power BI here) so DefaultAzureCredential falls through to AzureCli
locally and ManagedIdentity / WorkloadIdentity in cluster.

`AZURE_OPENAI_CLIENT_ID` (optional) selects which user-assigned managed
identity to use when the AKS node has more than one attached.
"""

import os
from functools import cache

_SCOPE = "https://cognitiveservices.azure.com/.default"


def _credential_kwargs() -> dict:
    kwargs = {"exclude_environment_credential": True}
    mi_client_id = os.getenv("AZURE_OPENAI_CLIENT_ID")
    if mi_client_id:
        kwargs["managed_identity_client_id"] = mi_client_id
    return kwargs


@cache
def sync_token_provider():
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    return get_bearer_token_provider(
        DefaultAzureCredential(**_credential_kwargs()), _SCOPE
    )


@cache
def async_token_provider():
    from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

    return get_bearer_token_provider(
        DefaultAzureCredential(**_credential_kwargs()), _SCOPE
    )


def resolve_azure_endpoint(overrides: dict | None = None) -> str:
    """Host-only URL for LiteLLM ``api_base`` (no ``/openai/v1`` suffix)."""
    overrides = overrides or {}
    endpoint = overrides.get("AZURE_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
    if not endpoint:
        base_url = (
            overrides.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
        ).rstrip("/")
        if base_url.endswith("/openai/v1"):
            endpoint = base_url[: -len("/openai/v1")]
    if not endpoint:
        raise ValueError(
            "Set AZURE_ENDPOINT (resource host) or OPENAI_BASE_URL ending in /openai/v1/."
        )
    return endpoint.rstrip("/")


def resolve_base_url(overrides: dict | None = None) -> str:
    """OpenAI v1 base URL: OPENAI_BASE_URL or <AZURE_ENDPOINT>/openai/v1/."""
    overrides = overrides or {}
    base_url = overrides.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    return base_url or f"{resolve_azure_endpoint(overrides)}/openai/v1/"
