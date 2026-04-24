"""Config flow for OpenAI Codex."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    SOURCE_REAUTH,
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
)

from .auth import (
    CodexAuthClient,
    CodexAuthError,
    CodexAuthTimeout,
    CodexTokenData,
    DeviceCode,
)
from .client import (
    CodexModelsError,
    CodexModelOption,
    async_load_cached_model_options,
    configured_chat_model,
    normalize_model_options,
    selected_reasoning_effort,
)
from .const import (
    DEFAULT_NAME,
    CONF_ACCOUNT_ID,
    CONF_CHAT_MODEL,
    CONF_FAST_MODE,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_USER_ID,
    CONF_WEB_SEARCH,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CONVERSATION_OPTIONS,
    RECOMMENDED_WEB_SEARCH,
    RECOMMENDED_FAST_MODE,
)


class OpenAICodexConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle an OpenAI Codex config flow."""

    VERSION = 1
    MINOR_VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(
        _config_entry: ConfigEntry,
    ) -> OpenAICodexOptionsFlow:
        """Get the options flow for this handler."""
        return OpenAICodexOptionsFlow()

    def __init__(self) -> None:
        """Initialize the flow."""
        self._auth_client: CodexAuthClient | None = None
        self._device_code: DeviceCode | None = None
        self._login_task: asyncio.Task[CodexTokenData] | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Start device-code login."""
        if self._login_task is not None:
            if self._login_task.done():
                if self._login_task.cancelled():
                    self._reset_login_state()
                    return await self.async_step_user()
                if exception := self._login_task.exception():
                    if isinstance(exception, CodexAuthTimeout):
                        return self.async_show_progress_done(next_step_id="timeout")
                    return self.async_show_progress_done(
                        next_step_id="connection_error"
                    )
                return self.async_show_progress_done(next_step_id="finish_login")

            return self.async_show_progress(
                step_id="user",
                progress_action="wait_for_authorization",
                description_placeholders={
                    "verification_url": self._device_code.verification_url,
                    "user_code": self._device_code.user_code,
                },
                progress_task=self._login_task,
            )

        if self._auth_client is None:
            self._auth_client = CodexAuthClient(self.hass)

        if self._device_code is None:
            try:
                self._device_code = await self._auth_client.request_device_code()
            except CodexAuthError:
                LOGGER.exception("Failed to start Codex device-code login")
                return self.async_abort(reason="cannot_connect")

        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema({}),
                description_placeholders={
                    "verification_url": self._device_code.verification_url,
                    "user_code": self._device_code.user_code,
                },
            )

        self._login_task = self.hass.async_create_task(
            self._auth_client.complete_device_login(self._device_code),
            name="openai_codex_device_login",
        )

        return self.async_show_progress(
            step_id="user",
            progress_action="wait_for_authorization",
            description_placeholders={
                "verification_url": self._device_code.verification_url,
                "user_code": self._device_code.user_code,
            },
            progress_task=self._login_task,
        )

    async def async_step_finish_login(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Finish device-code login."""
        if self._login_task is None or not self._login_task.done():
            return await self.async_step_user()
        if self._login_task.cancelled():
            self._reset_login_state()
            return await self.async_step_user()

        token_data = self._login_task.result()
        unique_ids = _unique_ids_from_token_data(token_data)
        if not unique_ids:
            LOGGER.error("Codex login completed without a stable account identity")
            return self.async_abort(reason="missing_account_identity")
        unique_id = unique_ids[0]
        if self.source == SOURCE_REAUTH:
            reauth_entry = self._get_reauth_entry()
            if not _same_identity(reauth_entry.data, token_data):
                return self.async_abort(reason="wrong_account")
            if reauth_entry.unique_id:
                unique_id = reauth_entry.unique_id
            await self.async_set_unique_id(unique_id)
            return self.async_update_reload_and_abort(
                reauth_entry,
                data_updates=token_data.as_config_data(),
            )

        await self.async_set_unique_id(unique_id)

        existing_unique_ids = {
            entry.unique_id
            for entry in self._async_current_entries()
            if entry.unique_id
        }
        for candidate in unique_ids:
            if candidate in existing_unique_ids:
                await self.async_set_unique_id(candidate)
                self._abort_if_unique_id_configured()

        self._abort_if_unique_id_configured()
        return self.async_create_entry(
            title=token_data.email or DEFAULT_NAME,
            data=token_data.as_config_data(),
            options=RECOMMENDED_CONVERSATION_OPTIONS,
        )

    async def async_step_timeout(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle device-code timeout."""
        if user_input is None:
            return self.async_show_form(step_id="timeout")

        self._reset_login_state()
        return await self.async_step_user()

    async def async_step_connection_error(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle a connection error during login."""
        if user_input is None:
            return self.async_show_form(step_id="connection_error")

        self._reset_login_state()
        return await self.async_step_user()

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Start reauthentication."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm reauthentication."""
        if user_input is None:
            return self.async_show_form(
                step_id="reauth_confirm",
                data_schema=vol.Schema({}),
            )
        return await self.async_step_user()

    def _reset_login_state(self) -> None:
        """Reset transient device-code flow state."""
        if self._login_task is not None and not self._login_task.done():
            self._login_task.cancel()
        self._device_code = None
        self._login_task = None


class OpenAICodexOptionsFlow(OptionsFlowWithReload):
    """Handle OpenAI Codex options."""

    def __init__(self) -> None:
        """Initialize options flow."""
        self._models: list[CodexModelOption] | None = None
        self._selected_model: CodexModelOption | None = None
        self._pending_options: dict[str, Any] | None = None

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Select the Codex model."""
        errors: dict[str, str] = {}
        current_options = {**dict(self.config_entry.options), **(user_input or {})}
        models = await self._async_get_models(errors)
        selected_model = (
            configured_chat_model(current_options) or RECOMMENDED_CHAT_MODEL
        )
        model = next(
            (option for option in models if option.model == selected_model),
            None,
        )

        if user_input is not None and models:
            if model is None:
                errors[CONF_CHAT_MODEL] = "model_not_available"
            else:
                self._selected_model = model
                self._pending_options = normalize_model_options(
                    current_options, model
                )
                return await self.async_step_capabilities()

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_CHAT_MODEL,
                    default=selected_model,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=_model_selector_options(models, selected_model)
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
        )

    async def async_step_capabilities(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage options scoped to the selected model capabilities."""
        errors: dict[str, str] = {}
        model = self._selected_model
        current_options = dict(self._pending_options or self.config_entry.options)
        if model is None:
            models = await self._async_get_models(errors)
            selected_model = configured_chat_model(current_options)
            model = next(
                (option for option in models if option.model == selected_model),
                None,
            )
            if model is None:
                return await self.async_step_init()
            self._selected_model = model

        current_options.update(user_input or {})
        hass_apis = _llm_api_selector_options(self.hass)
        _normalize_llm_api_options(current_options, hass_apis)
        self._pending_options = current_options

        if user_input is not None:
            return self.async_create_entry(
                title="",
                data=normalize_model_options(current_options, model),
            )

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_PROMPT,
                    description={
                        "suggested_value": current_options.get(
                            CONF_PROMPT,
                            RECOMMENDED_CONVERSATION_OPTIONS[CONF_PROMPT],
                        )
                    },
                ): TemplateSelector(),
                vol.Optional(
                    CONF_LLM_HASS_API,
                    default=_llm_api_default(current_options),
                ): SelectSelector(
                    SelectSelectorConfig(options=hass_apis, multiple=True)
                ),
            }
        )
        schema = schema.extend(_capability_schema(model, current_options))

        return self.async_show_form(
            step_id="capabilities",
            data_schema=schema,
            errors=errors,
        )

    async def _async_get_models(
        self, errors: dict[str, str]
    ) -> list[CodexModelOption]:
        """Return model options from runtime or the last real HA cache."""
        if self._models is None:
            runtime = getattr(self.config_entry, "runtime_data", None)
            try:
                if runtime is not None:
                    models = await runtime.async_get_models(
                        allow_stale_on_error=False,
                        force_refresh=True,
                    )
                else:
                    models = await async_load_cached_model_options(
                        self.hass, self.config_entry.entry_id
                    )
            except ConfigEntryAuthFailed:
                self.config_entry.async_start_reauth(self.hass)
                errors["base"] = "reauth_required"
                return []
            except CodexModelsError:
                LOGGER.exception("Failed to load Codex model catalog")
                errors["base"] = "model_catalog_unavailable"
                return []
            if not models:
                errors.setdefault("base", "model_catalog_unavailable")
                return []
            self._models = models
        return self._models


def _unique_ids_from_token_data(token_data: CodexTokenData) -> list[str]:
    """Return stable unique ID candidates, preferred first."""
    unique_ids = [
        token_data.chatgpt_account_id,
        token_data.email.lower() if token_data.email else None,
        token_data.chatgpt_user_id,
    ]
    return [unique_id for unique_id in unique_ids if unique_id]


def _same_identity(
    entry_data: Mapping[str, Any], token_data: CodexTokenData
) -> bool:
    """Return whether new token data belongs to the existing entry identity."""
    old_account_id = entry_data.get(CONF_ACCOUNT_ID)
    if old_account_id:
        return token_data.chatgpt_account_id == old_account_id

    old_user_id = entry_data.get(CONF_USER_ID)
    if old_user_id:
        return token_data.chatgpt_user_id == old_user_id

    old_email = entry_data.get("email")
    return bool(
        old_email
        and token_data.email
        and token_data.email.lower() == str(old_email).lower()
    )


def _model_selector_options(
    models: list[CodexModelOption], selected_model: str
) -> list[SelectOptionDict]:
    """Return available Codex model selector options."""
    options = [
        SelectOptionDict(value=model.model, label=model.label) for model in models
    ]
    if selected_model not in {model.model for model in models}:
        options.append(
            SelectOptionDict(
                value=selected_model,
                label=f"{selected_model} (unavailable)",
            )
        )
    return options


def _capability_schema(
    model: CodexModelOption | None, options: dict[str, Any]
) -> dict[Any, Any]:
    """Return form fields for capabilities advertised by the selected model."""
    if model is None:
        return {}

    schema: dict[Any, Any] = {}
    if model.reasoning_efforts:
        schema[
            vol.Required(
                CONF_REASONING_EFFORT,
                default=selected_reasoning_effort(model, options),
            )
        ] = SelectSelector(
            SelectSelectorConfig(options=_reasoning_selector_options(model))
        )
    if model.supports_web_search:
        schema[
            vol.Required(
                CONF_WEB_SEARCH,
                default=bool(options.get(CONF_WEB_SEARCH, RECOMMENDED_WEB_SEARCH)),
            )
        ] = bool
    if model.supports_fast:
        schema[
            vol.Required(
                CONF_FAST_MODE,
                default=bool(options.get(CONF_FAST_MODE, RECOMMENDED_FAST_MODE)),
            )
        ] = bool
    return schema


def _reasoning_selector_options(model: CodexModelOption) -> list[SelectOptionDict]:
    """Return reasoning effort selector options."""
    return [
        SelectOptionDict(value=option.effort, label=option.label)
        for option in model.reasoning_efforts
    ]


def _llm_api_selector_options(hass: HomeAssistant) -> list[SelectOptionDict]:
    """Return available Home Assistant LLM API options."""
    return [
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    ]


def _llm_api_default(options: Mapping[str, Any]) -> Any:
    """Return the LLM API form default, preserving an explicit empty list."""
    llm_api_default = options.get(CONF_LLM_HASS_API)
    if llm_api_default is None:
        return RECOMMENDED_CONVERSATION_OPTIONS[CONF_LLM_HASS_API]
    return llm_api_default


def _normalize_llm_api_options(
    options: dict[str, Any], hass_apis: list[SelectOptionDict]
) -> None:
    """Drop LLM API selections that are not currently registered."""
    selected_apis = options.get(CONF_LLM_HASS_API)
    if isinstance(selected_apis, str):
        selected_apis = [selected_apis]
    if not selected_apis:
        return

    valid_api_ids = {api["value"] for api in hass_apis}
    options[CONF_LLM_HASS_API] = [
        api for api in selected_apis if api in valid_api_ids
    ]
