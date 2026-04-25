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
CONF_STT_MODEL = "stt_model"
CONF_TTS_VOICE = "tts_voice"
CONF_USER_ID = "chatgpt_user_id"
CONF_WEB_SEARCH = "web_search"

DEFAULT_NAME = "OpenAI Codex"
DEFAULT_STT_NAME = "OpenAI Codex STT"
DEFAULT_TTS_NAME = "OpenAI Codex TTS"
EVENT_TTS_PREWARM = f"{DOMAIN}_tts_prewarm"
RECOMMENDED_CHAT_MODEL = "gpt-5.4"
RECOMMENDED_FAST_MODE = False
RECOMMENDED_REASONING_EFFORT = "medium"
RECOMMENDED_STT_MODEL = "gpt-4o-mini-transcribe"
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
CODEX_REALTIME_BASE_URL = "https://api.openai.com/v1"
CODEX_REALTIME_MODEL = "gpt-realtime-1.5"
CODEX_REALTIME_VERSION = "v2"
CODEX_REALTIME_AUDIO_RATE = 24_000
CODEX_REALTIME_AUDIO_CHANNELS = 1
CODEX_REALTIME_SAMPLE_WIDTH_BYTES = 2
CODEX_REALTIME_CALLS_PATH = "realtime/calls"
CODEX_REALTIME_WS_PATH = "realtime"

CODEX_TTS_DEFAULT_LANGUAGE = "en"
CODEX_TTS_SUPPORTED_LANGUAGES = (
    "ar",
    "bg",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "en-GB",
    "en-US",
    "es",
    "es-419",
    "fi",
    "fr",
    "he",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "nl",
    "no",
    "pl",
    "pt",
    "pt-BR",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sr-Latn",
    "sv",
    "th",
    "tr",
    "uk",
    "vi",
    "zh-CN",
    "zh-HK",
    "zh-TW",
)
CODEX_TTS_DEFAULT_VOICE = "marin"
CODEX_TTS_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
)
CODEX_TTS_SUPPORTED_OPTIONS = ("voice",)

CODEX_STT_SUPPORTED_LANGUAGES = (
    "af-ZA",
    "ar-SA",
    "hy-AM",
    "az-AZ",
    "be-BY",
    "bs-BA",
    "bg-BG",
    "ca-ES",
    "zh-CN",
    "hr-HR",
    "cs-CZ",
    "da-DK",
    "nl-NL",
    "en-US",
    "et-EE",
    "fi-FI",
    "fr-FR",
    "gl-ES",
    "de-DE",
    "el-GR",
    "he-IL",
    "hi-IN",
    "hu-HU",
    "is-IS",
    "id-ID",
    "it-IT",
    "ja-JP",
    "kn-IN",
    "kk-KZ",
    "ko-KR",
    "lv-LV",
    "lt-LT",
    "mk-MK",
    "ms-MY",
    "mr-IN",
    "mi-NZ",
    "ne-NP",
    "no-NO",
    "fa-IR",
    "pl-PL",
    "pt-PT",
    "ro-RO",
    "ru-RU",
    "sr-RS",
    "sk-SK",
    "sl-SI",
    "es-ES",
    "sw-KE",
    "sv-SE",
    "fil-PH",
    "ta-IN",
    "th-TH",
    "tr-TR",
    "uk-UA",
    "ur-PK",
    "vi-VN",
    "cy-GB",
)

USER_AGENT = "home-assistant-openai-codex/0.1"
ORIGINATOR = "homeassistant_openai_codex"
