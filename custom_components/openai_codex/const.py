"""Constants for the OpenAI Codex integration."""

import logging

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

DOMAIN = "openai_codex"
LOGGER = logging.getLogger(__package__)

CONF_ACCESS_TOKEN = "access_token"
CONF_ACCOUNT_ID = "chatgpt_account_id"
CONF_ACCOUNT_IS_FEDRAMP = "chatgpt_account_is_fedramp"
CONF_CHAT_MODEL = "chat_model"
CONF_EXPIRES_AT = "expires_at"
CONF_FAST_MODE = "fast_mode"
CONF_ID_TOKEN = "id_token"
CONF_LAST_REFRESH = "last_refresh"
CONF_PROMPT = "prompt"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_REFRESH_TOKEN = "refresh_token"
CONF_USER_ID = "chatgpt_user_id"
CONF_WEB_SEARCH = "web_search"

DEFAULT_NAME = "OpenAI Codex"
RECOMMENDED_CHAT_MODEL = "gpt-5.4"
RECOMMENDED_FAST_MODE = False
RECOMMENDED_REASONING_EFFORT = "medium"
RECOMMENDED_WEB_SEARCH = False
CODEX_FAST_SERVICE_TIER = "priority"
CODEX_MODELS_CLIENT_VERSION = "0.124.0"
RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_ISSUER = "https://auth.openai.com"
CODEX_DEVICE_API_BASE = f"{CODEX_ISSUER}/api/accounts"
CODEX_DEVICE_VERIFICATION_URL = f"{CODEX_ISSUER}/codex/device"
CODEX_DEVICE_REDIRECT_URI = f"{CODEX_ISSUER}/deviceauth/callback"
CODEX_TOKEN_URL = f"{CODEX_ISSUER}/oauth/token"
CODEX_BACKEND_BASE_URL = "https://chatgpt.com/backend-api/codex"

USER_AGENT = "home-assistant-openai-codex/0.1"
ORIGINATOR = "homeassistant_openai_codex"
