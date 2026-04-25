"""OpenAI Codex custom integration."""

from __future__ import annotations

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady

from .client import (
    CodexModelsError,
    OpenAICodexConfigEntry,
    OpenAICodexRuntime,
    configured_chat_model,
)
from .const import DOMAIN, RECOMMENDED_CHAT_MODEL

PLATFORMS = (Platform.CONVERSATION, Platform.TTS)


async def async_setup_entry(
    hass: HomeAssistant, entry: OpenAICodexConfigEntry
) -> bool:
    """Set up OpenAI Codex from a config entry."""
    runtime = OpenAICodexRuntime(hass, entry)

    try:
        await runtime.async_prepare()
    except ConfigEntryAuthFailed:
        raise
    except Exception as err:
        raise ConfigEntryNotReady from err

    entry.runtime_data = runtime
    try:
        resolved = await runtime.async_normalize_model_options()
    except ConfigEntryAuthFailed:
        raise
    except CodexModelsError as err:
        raise ConfigEntryNotReady from err
    if resolved is None:
        selected_model = configured_chat_model(entry.options) or RECOMMENDED_CHAT_MODEL
        raise ConfigEntryNotReady(
            f"Selected Codex model is not available: {selected_model}"
        )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: OpenAICodexConfigEntry
) -> bool:
    """Unload OpenAI Codex."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
