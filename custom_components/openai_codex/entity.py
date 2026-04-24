"""Conversation entity helpers for OpenAI Codex."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping
import json
import logging
from typing import Any, Literal

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.exceptions import ConfigEntryAuthFailed, HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.json import json_dumps

from .auth import CodexAuthError
from .client import (
    CodexModelOption,
    CodexResolvedOptions,
    CodexModelsError,
    OpenAICodexConfigEntry,
    resolve_model_options_from_catalog,
)
from .const import (
    CODEX_FAST_SERVICE_TIER,
    CONF_CHAT_MODEL,
    CONF_FAST_MODE,
    CONF_REASONING_EFFORT,
    CONF_WEB_SEARCH,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
)

MAX_TOOL_ITERATIONS = 10
MAX_TRANSIENT_API_ATTEMPTS = 3
WEB_SEARCH_TEXT_AND_IMAGE_TOOL_TYPE = "text_and_image"
WEB_SEARCH_TEXT_AND_IMAGE_CONTENT_TYPES = ["text", "image"]
DEFAULT_INSTRUCTIONS = "You are a voice assistant for Home Assistant."
ENTITY_ALIASES_INSTRUCTIONS = (
    "In the Home Assistant context below, each YAML list item is one "
    "Home Assistant entity. The `names` field contains alternative names "
    "or aliases for the same entity, not separate devices. Use any of these "
    "names to refer to that one entity."
)


def _openai_error_message(err: openai.OpenAIError) -> str:
    """Extract a user-facing message from an OpenAI SDK error."""
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            return detail

        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code")
            if isinstance(message, str) and message:
                return message
        if isinstance(error, str) and error:
            return error

        message = body.get("message")
        if isinstance(message, str) and message:
            return message

    message = getattr(err, "message", None)
    if isinstance(message, str) and message:
        return message

    return str(err) or "Unknown Codex API error"


def _openai_error_context(err: openai.OpenAIError) -> dict[str, Any]:
    """Return safe diagnostic fields from an OpenAI SDK error."""
    context: dict[str, Any] = {
        "type": type(err).__name__,
    }

    for attr_name in ("status_code", "request_id", "code"):
        value = getattr(err, attr_name, None)
        if value is not None:
            context[attr_name] = value

    body = getattr(err, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            for key in ("type", "code"):
                if isinstance(error.get(key), str):
                    context[f"error_{key}"] = error[key]

    return context


def _is_transient_openai_error(err: openai.OpenAIError) -> bool:
    """Return whether an OpenAI SDK error is worth retrying."""
    status_code = getattr(err, "status_code", None)
    if status_code in (408, 409, 429, 500, 502, 503, 504):
        return True

    message = _openai_error_message(err).lower()
    return (
        "retry your request" in message
        or "overloaded" in message
        or "temporarily unavailable" in message
    )


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> FunctionToolParam:
    """Format a Home Assistant LLM tool for Responses API."""
    schema = convert(tool.parameters, custom_serializer=custom_serializer)

    return FunctionToolParam(
        type="function",
        name=tool.name,
        parameters=schema,
        description=tool.description,
        strict=False,
    )


def _format_web_search_tool(web_search_tool_type: str | None) -> dict[str, Any]:
    """Format the Codex hosted web search tool for Responses API."""
    web_search_tool: dict[str, Any] = {
        "type": "web_search",
        "external_web_access": True,
    }
    if web_search_tool_type == WEB_SEARCH_TEXT_AND_IMAGE_TOOL_TYPE:
        web_search_tool["search_content_types"] = (
            WEB_SEARCH_TEXT_AND_IMAGE_CONTENT_TYPES
        )
    return web_search_tool


def _convert_content_to_param(
    chat_content: Iterable[conversation.Content],
) -> ResponseInputParam:
    """Convert Home Assistant chat content to Responses API input."""
    messages: ResponseInputParam = []

    for content in chat_content:
        if isinstance(content, conversation.ToolResultContent):
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": content.tool_call_id,
                    "output": json_dumps(content.tool_result),
                }
            )
            continue

        if content.content:
            role: Literal["user", "assistant", "system", "developer"] = content.role
            if role in ("system", "developer"):
                continue
            messages.append(
                EasyInputMessageParam(
                    type="message",
                    role=role,
                    content=content.content,
                )
            )

        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            for tool_call in content.tool_calls:
                if not tool_call.tool_name:
                    raise HomeAssistantError(
                        "Cannot send a tool call without a function name"
                    )
                messages.append(
                    {
                        "type": "function_call",
                        "name": tool_call.tool_name,
                        "arguments": json_dumps(tool_call.tool_args),
                        "call_id": tool_call.id,
                    }
                )

    return messages


def _extract_instructions(
    chat_content: Iterable[conversation.Content],
    *,
    include_entity_aliases: bool,
) -> str:
    """Extract top-level instructions required by the Codex backend."""
    instructions = []
    if include_entity_aliases:
        instructions.append(ENTITY_ALIASES_INSTRUCTIONS)
    instructions.extend(
        content.content
        for content in chat_content
        if content.role in ("system", "developer") and content.content
    )
    if not instructions:
        instructions.append(DEFAULT_INSTRUCTIONS)
    return "\n\n".join(instructions) or DEFAULT_INSTRUCTIONS


async def _resolve_request_options(
    entry: OpenAICodexConfigEntry,
) -> CodexResolvedOptions:
    """Resolve the model and normalized options for a conversation request."""
    current_options = dict(entry.options)
    configured_model = current_options.get(CONF_CHAT_MODEL)
    has_configured_model = isinstance(configured_model, str) and bool(configured_model)
    selected_model = (
        configured_model if has_configured_model else RECOMMENDED_CHAT_MODEL
    )

    LOGGER.debug(
        "Resolving Codex model capabilities for %s from the model catalog",
        selected_model,
    )
    try:
        models = await entry.runtime_data.async_get_models()
        resolved = resolve_model_options_from_catalog(
            models,
            current_options,
            model=selected_model if has_configured_model else None,
        )
    except CodexModelsError as err:
        raise HomeAssistantError(
            f"Could not load Codex model catalog: {err}"
        ) from err

    if resolved is None:
        raise HomeAssistantError(
            f"Selected Codex model is not available: {selected_model}"
        )
    return resolved


def _apply_model_capability_options(
    model_args: dict[str, Any],
    tools: list[Any],
    options: Mapping[str, Any],
    model: CodexModelOption,
) -> None:
    """Apply options that must be validated against model metadata."""
    if reasoning_effort := options.get(CONF_REASONING_EFFORT):
        supported_efforts = {option.effort for option in model.reasoning_efforts}
        if reasoning_effort not in supported_efforts:
            raise HomeAssistantError(
                f"Reasoning effort {reasoning_effort} is not supported by {model.model}"
            )
        model_args["reasoning"] = {"effort": reasoning_effort}

    if options.get(CONF_FAST_MODE):
        if not model.supports_fast:
            raise HomeAssistantError(f"Fast mode is not supported by {model.model}")
        model_args["service_tier"] = CODEX_FAST_SERVICE_TIER

    if options.get(CONF_WEB_SEARCH):
        if not model.supports_web_search:
            raise HomeAssistantError(f"Web search is not supported by {model.model}")
        tools.append(_format_web_search_tool(model.web_search_tool_type))


def _log_stream_event(event: ResponseStreamEvent) -> None:
    """Log a stream event without user content or tool arguments."""
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return

    metadata = {
        key: getattr(event, key)
        for key in ("type", "item_id", "output_index", "sequence_number")
        if hasattr(event, key)
    }
    if isinstance(event, (ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent)):
        metadata["item_type"] = event.item.type
        metadata["item_id"] = getattr(event.item, "id", None)
    LOGGER.debug("Received Codex event metadata: %s", metadata)


async def _transform_stream(
    chat_log: conversation.ChatLog,
    stream: Any,
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
]:
    """Transform an OpenAI Responses stream into Home Assistant deltas."""
    tool_calls_by_item_id: dict[str, ResponseFunctionToolCall] = {}

    async for event in stream:
        _log_stream_event(event)

        if isinstance(event, ResponseOutputItemAddedEvent):
            if isinstance(event.item, ResponseFunctionToolCall):
                if event.item.id is not None:
                    tool_calls_by_item_id[event.item.id] = event.item
                tool_calls_by_item_id[event.item.call_id] = event.item
                yield {"role": "assistant"}
            elif isinstance(event.item, ResponseOutputMessage):
                yield {"role": "assistant"}
        elif isinstance(event, ResponseOutputItemDoneEvent):
            if isinstance(event.item, ResponseFunctionToolCall):
                if event.item.id is not None:
                    tool_calls_by_item_id[event.item.id] = event.item
                tool_calls_by_item_id[event.item.call_id] = event.item
        elif isinstance(event, ResponseTextDeltaEvent):
            if event.delta:
                yield {"content": event.delta}
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            continue
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            tool_call = tool_calls_by_item_id.get(event.item_id)
            call_id = tool_call.call_id if tool_call is not None else event.item_id
            tool_name = event.name or (
                tool_call.name if tool_call is not None else None
            )
            if not tool_name:
                raise HomeAssistantError(
                    "Codex emitted a tool call without a function name"
                )
            try:
                arguments = json.loads(event.arguments)
            except json.JSONDecodeError as err:
                raise HomeAssistantError(
                    f"Codex emitted invalid JSON arguments for {tool_name}: {err}"
                ) from err
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=call_id,
                        tool_name=tool_name,
                        tool_args=arguments,
                    )
                ]
            }
        elif isinstance(event, ResponseCompletedEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
        elif isinstance(event, ResponseIncompleteEvent):
            reason = "unknown reason"
            if (
                event.response.incomplete_details
                and event.response.incomplete_details.reason
            ):
                reason = event.response.incomplete_details.reason
            raise HomeAssistantError(f"Codex response incomplete: {reason}")
        elif isinstance(event, ResponseFailedEvent):
            reason = "unknown reason"
            if event.response.error is not None:
                reason = event.response.error.message
            raise HomeAssistantError(f"Codex response failed: {reason}")
        elif isinstance(event, ResponseErrorEvent):
            raise HomeAssistantError(f"Codex response error: {event.message}")


class OpenAICodexBaseLLMEntity(Entity):
    """Base entity for OpenAI Codex conversation entities."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(self, entry: OpenAICodexConfigEntry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="OpenAI",
            model=entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        """Generate an answer for the chat log."""
        messages = _convert_content_to_param(chat_log.content)

        try:
            resolved = await _resolve_request_options(self.entry)
        except ConfigEntryAuthFailed:
            self.entry.async_start_reauth(self.hass)
            raise HomeAssistantError("Authentication error") from None

        model = resolved.model.model
        options = resolved.options

        model_args: dict[str, Any] = {
            "model": model,
            "instructions": _extract_instructions(
                chat_log.content,
                include_entity_aliases=bool(chat_log.llm_api),
            ),
            "input": messages,
            "stream": True,
            "store": False,
        }

        runtime = self.entry.runtime_data
        tools: list[Any] = []
        if chat_log.llm_api:
            tools.extend(
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            )

        _apply_model_capability_options(model_args, tools, options, resolved.model)

        if tools:
            model_args["tools"] = tools
            model_args["tool_choice"] = "auto"
            model_args["parallel_tool_calls"] = False

        for _iteration in range(max_iterations):
            for attempt in range(1, MAX_TRANSIENT_API_ATTEMPTS + 1):
                content_count_before_request = len(chat_log.content)
                try:
                    stream: AsyncGenerator[ResponseStreamEvent] = (
                        await runtime.async_create_response(**model_args)
                    )
                    messages.extend(
                        _convert_content_to_param(
                            [
                                content
                                async for content in (
                                    chat_log.async_add_delta_content_stream(
                                        self.entity_id,
                                        _transform_stream(chat_log, stream),
                                    )
                                )
                            ]
                        )
                    )
                    break
                except ConfigEntryAuthFailed:
                    self.entry.async_start_reauth(self.hass)
                    raise HomeAssistantError("Authentication error") from None
                except openai.AuthenticationError as err:
                    self.entry.async_start_reauth(self.hass)
                    raise HomeAssistantError("Authentication error") from err
                except CodexAuthError as err:
                    LOGGER.error("Error refreshing Codex auth: %s", err)
                    raise HomeAssistantError("Error refreshing Codex auth") from err
                except openai.OpenAIError as err:
                    message = _openai_error_message(err)
                    error_context = _openai_error_context(err)
                    can_retry = (
                        attempt < MAX_TRANSIENT_API_ATTEMPTS
                        and _is_transient_openai_error(err)
                        and len(chat_log.content) == content_count_before_request
                    )
                    if can_retry:
                        LOGGER.warning(
                            "Transient Codex API error, retrying request: %s",
                            error_context,
                        )
                        await asyncio.sleep(attempt)
                        continue

                    LOGGER.error(
                        "Error talking to Codex: %s (%s)",
                        message,
                        error_context,
                    )
                    raise HomeAssistantError(
                        f"Error talking to Codex: {message}"
                    ) from err

            if not chat_log.unresponded_tool_results:
                return

        raise HomeAssistantError("Codex exceeded the maximum tool-call iterations")
