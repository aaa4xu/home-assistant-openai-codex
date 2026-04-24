"""Runtime client for OpenAI Codex."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.storage import Store

from .auth import (
    CodexAuthClient,
    CodexAuthError,
    CodexAuthPermanentError,
    CodexTokenData,
    token_data_from_config,
    token_refresh_needed,
)
from .const import (
    CODEX_BACKEND_BASE_URL,
    CODEX_MODELS_CLIENT_VERSION,
    CONF_ACCESS_TOKEN,
    CONF_ACCOUNT_ID,
    CONF_ACCOUNT_IS_FEDRAMP,
    CONF_CHAT_MODEL,
    CONF_EXPIRES_AT,
    CONF_FAST_MODE,
    CONF_ID_TOKEN,
    CONF_LAST_REFRESH,
    CONF_REASONING_EFFORT,
    CONF_REFRESH_TOKEN,
    CONF_USER_ID,
    CONF_WEB_SEARCH,
    LOGGER,
    ORIGINATOR,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_FAST_MODE,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_WEB_SEARCH,
    USER_AGENT,
)

MODELS_CACHE_SCHEMA_VERSION = 4
MODELS_CACHE_STORAGE_VERSION = 1
MODELS_CACHE_TTL = timedelta(minutes=5)
MODELS_REQUEST_TIMEOUT = 5.0
AUTH_DATA_KEYS = {
    CONF_ID_TOKEN,
    CONF_ACCESS_TOKEN,
    CONF_REFRESH_TOKEN,
    CONF_EXPIRES_AT,
    CONF_LAST_REFRESH,
    CONF_ACCOUNT_IS_FEDRAMP,
    CONF_USER_ID,
    CONF_ACCOUNT_ID,
    "email",
}
DEPRECATED_CONF_MODEL_CAPABILITIES = "model_capabilities"


class CodexModelsError(Exception):
    """Raised when the Codex models catalog cannot be fetched or parsed."""


class CodexModelParseError(CodexModelsError):
    """Raised when a single Codex model item cannot be parsed."""


@dataclass(slots=True, frozen=True)
class CodexReasoningOption:
    """Picker-ready Codex reasoning effort option."""

    effort: str
    label: str

    @classmethod
    def from_api_option(cls, data: Any) -> CodexReasoningOption | None:
        """Create a reasoning option from a Codex models API item."""
        if not isinstance(data, dict):
            return None

        effort = data.get("effort")
        label = data.get("description")
        if not isinstance(effort, str) or not effort:
            return None
        if not isinstance(label, str) or not label:
            label = effort
        return cls(effort=effort, label=label)

    @classmethod
    def from_storage(cls, data: Any) -> CodexReasoningOption | None:
        """Create a reasoning option from stored cache data."""
        if not isinstance(data, dict):
            return None
        effort = data.get("effort")
        label = data.get("label")
        if not isinstance(effort, str) or not effort:
            return None
        if not isinstance(label, str) or not label:
            label = effort
        return cls(effort=effort, label=label)

    def as_storage(self) -> dict[str, str]:
        """Return JSON-serializable cache data."""
        return {
            "effort": self.effort,
            "label": self.label,
        }


@dataclass(slots=True, frozen=True)
class CodexModelOption:
    """Picker-ready Codex model option."""

    model: str
    label: str
    supports_fast: bool
    supports_web_search: bool
    web_search_tool_type: str | None
    priority: int
    default_reasoning_effort: str | None
    reasoning_efforts: tuple[CodexReasoningOption, ...]

    @classmethod
    def from_api_model(cls, model: dict[str, Any]) -> CodexModelOption | None:
        """Create a model option from a Codex models API item."""
        slug = model.get("slug")
        if not isinstance(slug, str) or not slug:
            raise CodexModelParseError("Model item is missing slug")

        visibility = model.get("visibility")
        if visibility != "list":
            return None

        if model.get("supported_in_api") is not True:
            return None

        label = model.get("display_name")
        if not isinstance(label, str) or not label:
            label = slug

        speed_tiers = model.get("additional_speed_tiers", [])
        if not isinstance(speed_tiers, list):
            speed_tiers = []
        speed_tiers = [tier for tier in speed_tiers if isinstance(tier, str)]
        supports_fast = "fast" in speed_tiers

        supports_search_tool = model.get("supports_search_tool", False)
        if not isinstance(supports_search_tool, bool):
            raise CodexModelParseError(
                f"Model {slug} has invalid supports_search_tool"
            )
        web_search_tool_type = model.get("web_search_tool_type")
        if not isinstance(web_search_tool_type, str):
            web_search_tool_type = None

        priority = model.get("priority", 1000)
        if not isinstance(priority, int):
            priority = 1000

        reasoning_levels = model.get("supported_reasoning_levels")
        reasoning_efforts = _parse_reasoning_options_from_api(
            slug, reasoning_levels
        )
        default_reasoning_effort = model.get("default_reasoning_level")
        if not isinstance(default_reasoning_effort, str):
            default_reasoning_effort = None
        if default_reasoning_effort and not any(
            option.effort == default_reasoning_effort for option in reasoning_efforts
        ):
            LOGGER.warning(
                "Model %s has unsupported default reasoning effort %s",
                slug,
                default_reasoning_effort,
            )
            default_reasoning_effort = None

        return cls(
            model=slug,
            label=label,
            supports_fast=supports_fast,
            supports_web_search=supports_search_tool,
            web_search_tool_type=web_search_tool_type,
            priority=priority,
            default_reasoning_effort=default_reasoning_effort,
            reasoning_efforts=reasoning_efforts,
        )

    @classmethod
    def from_storage(cls, data: dict[str, Any]) -> CodexModelOption | None:
        """Create a model option from stored cache data."""
        model = data.get("model")
        label = data.get("label")
        if not isinstance(model, str) or not model:
            return None
        if not isinstance(label, str) or not label:
            label = model
        priority = data.get("priority", 1000)
        if not isinstance(priority, int):
            priority = 1000
        default_reasoning_effort = data.get("default_reasoning_effort")
        if not isinstance(default_reasoning_effort, str):
            default_reasoning_effort = None
        reasoning_efforts = _parse_reasoning_options_from_storage(
            data.get("reasoning_efforts")
        )
        supports_fast = data.get("supports_fast", False)
        if not isinstance(supports_fast, bool):
            supports_fast = False
        supports_web_search = data.get("supports_web_search", False)
        if not isinstance(supports_web_search, bool):
            supports_web_search = False
        return cls(
            model=model,
            label=label,
            supports_fast=supports_fast,
            supports_web_search=supports_web_search,
            web_search_tool_type=data.get("web_search_tool_type")
            if isinstance(data.get("web_search_tool_type"), str)
            else None,
            priority=priority,
            default_reasoning_effort=default_reasoning_effort,
            reasoning_efforts=reasoning_efforts,
        )

    def as_storage(self) -> dict[str, Any]:
        """Return JSON-serializable cache data."""
        return {
            "model": self.model,
            "label": self.label,
            "supports_fast": self.supports_fast,
            "supports_web_search": self.supports_web_search,
            "web_search_tool_type": self.web_search_tool_type,
            "priority": self.priority,
            "default_reasoning_effort": self.default_reasoning_effort,
            "reasoning_efforts": [
                option.as_storage() for option in self.reasoning_efforts
            ],
        }


async def async_load_cached_model_options(
    hass: HomeAssistant, entry_id: str
) -> list[CodexModelOption]:
    """Load cached model options for a config entry."""
    store = _models_store(hass, entry_id)
    return _model_options_from_cache(await store.async_load(), allow_stale=True)


@dataclass(slots=True, frozen=True)
class CodexResolvedOptions:
    """Selected model plus normalized entry options."""

    model: CodexModelOption
    options: dict[str, Any]


class OpenAICodexRuntime:
    """Runtime holder for auth state and the OpenAI SDK client."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize runtime."""
        self.hass = hass
        self.entry = entry
        self.auth_client = CodexAuthClient(hass)
        self.token_data = token_data_from_config(dict(entry.data))
        self._client: openai.AsyncOpenAI | None = None
        self._refresh_lock = asyncio.Lock()
        self._models_lock = asyncio.Lock()
        self._models_store = _models_store(hass, entry.entry_id)
        self._models_cache: list[CodexModelOption] | None = None
        self._models_cache_updated_at: float | None = None

    async def async_prepare(self) -> None:
        """Prepare the runtime before platforms are set up."""
        if token_refresh_needed(self.token_data):
            await self.async_refresh_tokens(force=False)
        self._client = self._create_client()

    async def async_refresh_tokens(self, *, force: bool = True) -> None:
        """Refresh tokens and persist them into the config entry."""
        async with self._refresh_lock:
            if not force and not token_refresh_needed(self.token_data):
                return

            try:
                refreshed = await self.auth_client.refresh_tokens(self.token_data)
            except CodexAuthPermanentError as err:
                raise ConfigEntryAuthFailed(str(err)) from err

            self.token_data = refreshed
            data = {
                key: value
                for key, value in self.entry.data.items()
                if key not in AUTH_DATA_KEYS
            }
            data.update(refreshed.as_config_data())
            self.hass.config_entries.async_update_entry(self.entry, data=data)
            self._client = self._create_client()

    async def async_get_client(self) -> openai.AsyncOpenAI:
        """Return a prepared OpenAI SDK client."""
        if token_refresh_needed(self.token_data):
            await self.async_refresh_tokens(force=False)
        if self._client is None:
            self._client = self._create_client()
        return self._client

    async def async_create_response(self, **kwargs: Any) -> Any:
        """Create a streaming response, refreshing once after a 401."""
        client = await self.async_get_client()
        try:
            return await client.responses.create(**kwargs)
        except openai.AuthenticationError:
            await self.async_refresh_tokens()
            client = await self.async_get_client()
            return await client.responses.create(**kwargs)

    async def async_get_models(
        self,
        *,
        allow_stale_on_error: bool = True,
        force_refresh: bool = False,
    ) -> list[CodexModelOption]:
        """Return available Codex models, using only real API/cache data."""
        async with self._models_lock:
            if not force_refresh and self._models_cache_is_fresh():
                return list(self._models_cache or [])

            stored_cache = await self._models_store.async_load()
            if not force_refresh:
                cached_models = _model_options_from_cache(stored_cache)
                if cached_models:
                    self._set_models_cache(
                        cached_models, _cache_updated_at(stored_cache)
                    )
                    return list(cached_models)

            stale_models = _model_options_from_cache(stored_cache, allow_stale=True)

            try:
                models = await self._async_fetch_models()
            except ConfigEntryAuthFailed:
                raise
            except CodexModelsError as err:
                if not allow_stale_on_error or not stale_models:
                    raise
                LOGGER.warning(
                    "Using stale Codex model catalog after models API error: %s",
                    err,
                )
                self._set_models_cache(stale_models, _cache_updated_at(stored_cache))
                return list(stale_models)

            await self._async_save_models_cache(models)
            return list(models)

    async def async_normalize_model_options(self) -> CodexResolvedOptions | None:
        """Normalize saved options against the model catalog."""
        options = dict(self.entry.options)
        models = await self.async_get_models()
        resolved = resolve_model_options_from_catalog(models, options)
        if resolved is None:
            return None
        self._update_options_if_changed(resolved.options)
        return resolved

    def _update_options_if_changed(self, options: dict[str, Any]) -> None:
        """Persist options only when normalization changed them."""
        if options != dict(self.entry.options):
            self.hass.config_entries.async_update_entry(self.entry, options=options)

    async def _async_fetch_models(self) -> list[CodexModelOption]:
        """Fetch model options from the Codex models endpoint."""
        return await self._async_fetch_models_with_refresh(retry_auth=True)

    async def _async_fetch_models_with_refresh(
        self, *, retry_auth: bool
    ) -> list[CodexModelOption]:
        """Fetch model options, refreshing auth once after a 401."""
        try:
            if token_refresh_needed(self.token_data):
                await self.async_refresh_tokens(force=False)
        except ConfigEntryAuthFailed:
            raise
        except CodexAuthError as err:
            raise CodexModelsError(
                f"Could not refresh auth before models request: {err}"
            ) from err

        http_client = get_async_client(self.hass)
        try:
            response = await http_client.get(
                f"{CODEX_BACKEND_BASE_URL}/models",
                params={"client_version": CODEX_MODELS_CLIENT_VERSION},
                headers=self._request_headers(include_auth=True),
                timeout=MODELS_REQUEST_TIMEOUT,
            )
        except httpx.HTTPError as err:
            raise CodexModelsError(f"Models request failed: {err}") from err

        if response.status_code == 401 and retry_auth:
            try:
                await self.async_refresh_tokens()
            except ConfigEntryAuthFailed:
                raise
            except CodexAuthError as err:
                raise CodexModelsError(
                    f"Could not refresh auth after models request was rejected: {err}"
                ) from err
            return await self._async_fetch_models_with_refresh(retry_auth=False)

        if response.status_code >= 400:
            raise CodexModelsError(
                f"Models request failed with status {response.status_code}"
            )

        try:
            data = response.json()
        except ValueError as err:
            raise CodexModelsError("Models response was not valid JSON") from err

        raw_models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(raw_models, list):
            raise CodexModelsError("Models response did not include a models list")

        models: list[CodexModelOption] = []
        skipped_models = 0
        for raw_model in raw_models:
            if not isinstance(raw_model, dict):
                skipped_models += 1
                continue
            try:
                option = CodexModelOption.from_api_model(raw_model)
            except CodexModelParseError as err:
                skipped_models += 1
                LOGGER.warning("Skipping invalid Codex model metadata: %s", err)
                continue
            if option is not None:
                models.append(option)

        if not models:
            raise CodexModelsError("Models response did not include usable models")
        if skipped_models:
            LOGGER.debug(
                "Skipped %s unusable Codex model catalog item(s)",
                skipped_models,
            )

        return sorted(models, key=lambda option: (option.priority, option.label))

    async def _async_save_models_cache(self, models: list[CodexModelOption]) -> None:
        """Persist the latest successful models catalog in HA storage."""
        now = datetime.now(UTC).timestamp()
        self._set_models_cache(models, now)
        await self._models_store.async_save(
            {
                "cache_schema_version": MODELS_CACHE_SCHEMA_VERSION,
                "client_version": CODEX_MODELS_CLIENT_VERSION,
                "updated_at": now,
                "models": [model.as_storage() for model in models],
            }
        )

    def _set_models_cache(
        self, models: list[CodexModelOption], updated_at: float | None
    ) -> None:
        """Update the in-memory models cache."""
        self._models_cache = models
        self._models_cache_updated_at = updated_at

    def _models_cache_is_fresh(self) -> bool:
        """Return whether the in-memory models cache can be reused."""
        if self._models_cache is None or self._models_cache_updated_at is None:
            return False
        age = datetime.now(UTC).timestamp() - self._models_cache_updated_at
        return age < MODELS_CACHE_TTL.total_seconds()

    def _create_client(self) -> openai.AsyncOpenAI:
        """Create an OpenAI SDK client pointed at the ChatGPT Codex backend."""
        return openai.AsyncOpenAI(
            api_key=self.token_data.access_token,
            base_url=CODEX_BACKEND_BASE_URL,
            default_headers=self._request_headers(),
            http_client=get_async_client(self.hass),
        )

    def _request_headers(self, *, include_auth: bool = False) -> dict[str, str]:
        """Return headers shared by Codex backend requests."""
        headers = {
            "originator": ORIGINATOR,
            "User-Agent": USER_AGENT,
        }
        if include_auth:
            headers["Authorization"] = f"Bearer {self.token_data.access_token}"
        if self.token_data.chatgpt_account_id:
            headers["ChatGPT-Account-ID"] = self.token_data.chatgpt_account_id
        if self.token_data.chatgpt_account_is_fedramp:
            headers["X-OpenAI-Fedramp"] = "true"
        return headers


type OpenAICodexConfigEntry = ConfigEntry[OpenAICodexRuntime]


def configured_chat_model(options: Mapping[str, Any]) -> str | None:
    """Return the configured model when explicitly stored in options."""
    selected_model = options.get(CONF_CHAT_MODEL)
    if not isinstance(selected_model, str) or not selected_model:
        return None
    return selected_model


def resolve_model_options_from_catalog(
    models: list[CodexModelOption],
    options: Mapping[str, Any],
    *,
    model: str | None = None,
) -> CodexResolvedOptions | None:
    """Resolve normalized options from the live model catalog."""
    selected_model = model or configured_chat_model(options) or RECOMMENDED_CHAT_MODEL
    model_option = next(
        (option for option in models if option.model == selected_model),
        None,
    )
    if model_option is None:
        return None
    return CodexResolvedOptions(
        model=model_option,
        options=normalize_model_options(options, model_option),
    )


def normalize_model_options(
    options: Mapping[str, Any], model: CodexModelOption
) -> dict[str, Any]:
    """Normalize options against the selected model capabilities."""
    normalized = dict(options)
    normalized[CONF_CHAT_MODEL] = model.model
    normalized.pop(DEPRECATED_CONF_MODEL_CAPABILITIES, None)

    if model.reasoning_efforts:
        normalized[CONF_REASONING_EFFORT] = selected_reasoning_effort(
            model, normalized
        )
    else:
        normalized.pop(CONF_REASONING_EFFORT, None)

    normalized[CONF_WEB_SEARCH] = (
        bool(normalized.get(CONF_WEB_SEARCH, RECOMMENDED_WEB_SEARCH))
        if model.supports_web_search
        else False
    )
    normalized[CONF_FAST_MODE] = (
        bool(normalized.get(CONF_FAST_MODE, RECOMMENDED_FAST_MODE))
        if model.supports_fast
        else False
    )
    return normalized


def selected_reasoning_effort(
    model: CodexModelOption, options: Mapping[str, Any]
) -> str:
    """Return the selected reasoning effort for a model."""
    supported_efforts = {option.effort for option in model.reasoning_efforts}
    selected_effort = options.get(CONF_REASONING_EFFORT)
    if isinstance(selected_effort, str) and selected_effort in supported_efforts:
        return selected_effort
    if (
        model.default_reasoning_effort
        and model.default_reasoning_effort in supported_efforts
    ):
        return model.default_reasoning_effort
    if RECOMMENDED_REASONING_EFFORT in supported_efforts:
        return RECOMMENDED_REASONING_EFFORT
    return model.reasoning_efforts[0].effort


def _models_store(hass: HomeAssistant, entry_id: str) -> Store[dict[str, Any]]:
    """Return the HA storage helper for a config entry model cache."""
    return Store(
        hass,
        MODELS_CACHE_STORAGE_VERSION,
        f"openai_codex.models.{entry_id}",
    )


def _cache_updated_at(cache: dict[str, Any] | None) -> float | None:
    """Return cache updated timestamp when valid."""
    if not isinstance(cache, dict):
        return None
    updated_at = cache.get("updated_at")
    return float(updated_at) if isinstance(updated_at, int | float) else None


def _parse_reasoning_options_from_api(
    model: str, data: Any
) -> tuple[CodexReasoningOption, ...]:
    """Parse API reasoning options."""
    if data is None:
        return ()
    if not isinstance(data, list):
        raise CodexModelParseError(
            f"Model {model} has invalid reasoning options"
        )

    options: list[CodexReasoningOption] = []
    seen: set[str] = set()
    for item in data:
        option = CodexReasoningOption.from_api_option(item)
        if option is None:
            raise CodexModelParseError(
                f"Model {model} has invalid reasoning option"
            )
        if option.effort in seen:
            raise CodexModelParseError(
                f"Model {model} has duplicate reasoning option {option.effort}"
            )
        options.append(option)
        seen.add(option.effort)
    return tuple(options)


def _parse_reasoning_options_from_storage(
    data: Any,
) -> tuple[CodexReasoningOption, ...]:
    """Parse stored reasoning options."""
    if not isinstance(data, list):
        return ()

    options: list[CodexReasoningOption] = []
    seen: set[str] = set()
    for item in data:
        option = CodexReasoningOption.from_storage(item)
        if option is None or option.effort in seen:
            continue
        options.append(option)
        seen.add(option.effort)
    return tuple(options)


def _model_options_from_cache(
    cache: dict[str, Any] | None, *, allow_stale: bool = False
) -> list[CodexModelOption]:
    """Return model options from HA storage cache."""
    if not isinstance(cache, dict):
        return []
    cache_schema_version = cache.get("cache_schema_version")
    is_current_schema = cache_schema_version == MODELS_CACHE_SCHEMA_VERSION
    if not is_current_schema:
        return []
    if cache.get("client_version") != CODEX_MODELS_CLIENT_VERSION:
        return []
    updated_at = _cache_updated_at(cache)
    if updated_at is None:
        return []
    if not allow_stale:
        age = datetime.now(UTC).timestamp() - updated_at
        if age >= MODELS_CACHE_TTL.total_seconds():
            return []

    raw_models = cache.get("models") if isinstance(cache, dict) else None
    if not isinstance(raw_models, list):
        return []

    models = [
        option
        for raw_model in raw_models
        if isinstance(raw_model, dict)
        if (option := CodexModelOption.from_storage(raw_model)) is not None
    ]
    return sorted(models, key=lambda option: (option.priority, option.label))
