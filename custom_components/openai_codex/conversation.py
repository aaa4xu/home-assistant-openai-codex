"""Conversation support for OpenAI Codex."""

from __future__ import annotations

from typing import Literal

from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .client import OpenAICodexConfigEntry
from .const import CONF_PROMPT, DOMAIN
from .entity import OpenAICodexBaseLLMEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAICodexConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    async_add_entities([OpenAICodexConversationEntity(config_entry)])


class OpenAICodexConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    OpenAICodexBaseLLMEntity,
):
    """OpenAI Codex conversation agent."""

    _attr_supports_streaming = True

    def __init__(self, entry: OpenAICodexConfigEntry) -> None:
        """Initialize the agent."""
        super().__init__(entry)
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Register this entity as a conversation agent."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister this entity as a conversation agent."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process user input through Codex."""
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                self.entry.options.get(CONF_LLM_HASS_API),
                self.entry.options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)
        return conversation.async_get_result_from_chat_log(user_input, chat_log)
