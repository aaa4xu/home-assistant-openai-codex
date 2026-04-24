"""Authentication helpers for OpenAI Codex."""

from __future__ import annotations

import asyncio
import binascii
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import base64
import json
from typing import Any

import aiohttp

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CODEX_CLIENT_ID,
    CODEX_DEVICE_API_BASE,
    CODEX_DEVICE_REDIRECT_URI,
    CODEX_DEVICE_VERIFICATION_URL,
    CODEX_TOKEN_URL,
    CONF_ACCESS_TOKEN,
    CONF_ACCOUNT_ID,
    CONF_ACCOUNT_IS_FEDRAMP,
    CONF_EXPIRES_AT,
    CONF_ID_TOKEN,
    CONF_LAST_REFRESH,
    CONF_REFRESH_TOKEN,
    CONF_USER_ID,
    ORIGINATOR,
    USER_AGENT,
)

DEVICE_AUTH_TIMEOUT = timedelta(minutes=15)
TOKEN_REFRESH_INTERVAL = timedelta(days=8)
TOKEN_EXPIRY_SKEW = timedelta(minutes=5)
AUTH_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)


class CodexAuthError(Exception):
    """Base Codex auth error."""


class CodexAuthPending(CodexAuthError):
    """Device authorization is still pending."""


class CodexAuthTimeout(CodexAuthError):
    """Device authorization timed out."""


class CodexAuthPermanentError(CodexAuthError):
    """Credentials cannot be refreshed and require reauth."""


@dataclass(slots=True)
class DeviceCode:
    """Device-code authorization state."""

    verification_url: str
    user_code: str
    device_auth_id: str
    interval: int


@dataclass(slots=True)
class CodexTokenData:
    """Persistable ChatGPT OAuth token data."""

    id_token: str
    access_token: str
    refresh_token: str
    expires_at: float | None
    last_refresh: float
    email: str | None
    chatgpt_user_id: str | None
    chatgpt_account_id: str | None
    chatgpt_account_is_fedramp: bool

    def as_config_data(self) -> dict[str, Any]:
        """Return JSON-serializable config entry data."""
        data: dict[str, Any] = {
            CONF_ID_TOKEN: self.id_token,
            CONF_ACCESS_TOKEN: self.access_token,
            CONF_REFRESH_TOKEN: self.refresh_token,
            CONF_LAST_REFRESH: self.last_refresh,
            CONF_ACCOUNT_IS_FEDRAMP: self.chatgpt_account_is_fedramp,
        }
        if self.expires_at is not None:
            data[CONF_EXPIRES_AT] = self.expires_at
        if self.email:
            data["email"] = self.email
        if self.chatgpt_user_id:
            data[CONF_USER_ID] = self.chatgpt_user_id
        if self.chatgpt_account_id:
            data[CONF_ACCOUNT_ID] = self.chatgpt_account_id
        return data


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode a JWT payload without verifying its signature."""
    try:
        _header, payload, _signature = token.split(".", 2)
    except ValueError:
        return {}

    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(f"{payload}{padding}")
    except (ValueError, binascii.Error):
        return {}

    try:
        data = json.loads(decoded)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _expires_at(access_token: str) -> float | None:
    """Return access token expiry timestamp, if present."""
    exp = _decode_jwt_payload(access_token).get("exp")
    if isinstance(exp, int | float):
        return float(exp)
    return None


def _non_empty_string(value: Any) -> str | None:
    """Return a string claim only when it is usable as an identity value."""
    return value if isinstance(value, str) and value else None


def _metadata_from_id_token(id_token: str) -> dict[str, Any]:
    """Extract ChatGPT profile metadata from an ID token."""
    claims = _decode_jwt_payload(id_token)
    profile = claims.get("https://api.openai.com/profile") or {}
    auth = claims.get("https://api.openai.com/auth") or {}

    if not isinstance(profile, dict):
        profile = {}
    if not isinstance(auth, dict):
        auth = {}

    return {
        "email": _non_empty_string(claims.get("email"))
        or _non_empty_string(profile.get("email")),
        "chatgpt_user_id": _non_empty_string(auth.get("chatgpt_user_id"))
        or _non_empty_string(auth.get("user_id")),
        "chatgpt_account_id": _non_empty_string(auth.get("chatgpt_account_id")),
        "chatgpt_account_is_fedramp": bool(
            auth.get("chatgpt_account_is_fedramp", False)
        ),
    }


def token_data_from_config(data: dict[str, Any]) -> CodexTokenData:
    """Build token data from a config entry data mapping."""
    metadata = _metadata_from_id_token(data[CONF_ID_TOKEN])
    return CodexTokenData(
        id_token=data[CONF_ID_TOKEN],
        access_token=data[CONF_ACCESS_TOKEN],
        refresh_token=data[CONF_REFRESH_TOKEN],
        expires_at=data.get(CONF_EXPIRES_AT),
        last_refresh=data.get(CONF_LAST_REFRESH, 0),
        email=data.get("email") or metadata["email"],
        chatgpt_user_id=data.get(CONF_USER_ID) or metadata["chatgpt_user_id"],
        chatgpt_account_id=data.get(CONF_ACCOUNT_ID)
        or metadata["chatgpt_account_id"],
        chatgpt_account_is_fedramp=bool(
            data.get(
                CONF_ACCOUNT_IS_FEDRAMP,
                metadata["chatgpt_account_is_fedramp"],
            )
        ),
    )


def token_data_from_response(
    response: dict[str, Any],
    *,
    previous: CodexTokenData | None = None,
) -> CodexTokenData:
    """Build token data from a token endpoint response."""
    id_token = response.get(CONF_ID_TOKEN) or (previous.id_token if previous else None)
    access_token = response.get(CONF_ACCESS_TOKEN)
    refresh_token = response.get(CONF_REFRESH_TOKEN) or (
        previous.refresh_token if previous else None
    )

    if not isinstance(id_token, str) or not isinstance(access_token, str):
        raise CodexAuthError("Token response did not include usable tokens")
    if not isinstance(refresh_token, str):
        raise CodexAuthError("Token response did not include a refresh token")

    metadata = _metadata_from_id_token(id_token)
    return CodexTokenData(
        id_token=id_token,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=_expires_at(access_token),
        last_refresh=datetime.now(UTC).timestamp(),
        email=metadata["email"],
        chatgpt_user_id=metadata["chatgpt_user_id"],
        chatgpt_account_id=metadata["chatgpt_account_id"],
        chatgpt_account_is_fedramp=metadata["chatgpt_account_is_fedramp"],
    )


def token_refresh_needed(token_data: CodexTokenData) -> bool:
    """Return whether stored tokens should be refreshed."""
    now = datetime.now(UTC).timestamp()
    if token_data.expires_at is not None:
        return token_data.expires_at <= now + TOKEN_EXPIRY_SKEW.total_seconds()
    refresh_deadline = datetime.now(UTC) - TOKEN_REFRESH_INTERVAL
    return token_data.last_refresh < refresh_deadline.timestamp()


class CodexAuthClient:
    """Small async client for Codex OAuth endpoints."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the auth client."""
        self._session = async_get_clientsession(hass)

    async def request_device_code(self) -> DeviceCode:
        """Request a new device code."""
        status, data = await self._post_json(
            f"{CODEX_DEVICE_API_BASE}/deviceauth/usercode",
            error_context="Device-code request",
            json={"client_id": CODEX_CLIENT_ID},
            headers=self._headers(),
        )
        if status == 404:
            raise CodexAuthError("Device-code login is not enabled")
        if status >= 400:
            raise CodexAuthError(
                f"Device-code request failed with status {status}: "
                f"{self._error_message(data)}"
            )

        user_code = data.get("user_code")
        interval_raw = data.get("interval", "5")
        if not isinstance(user_code, str) or not isinstance(
            data.get("device_auth_id"), str
        ):
            raise CodexAuthError("Device-code response is missing required fields")
        try:
            interval = int(str(interval_raw).strip())
        except (TypeError, ValueError) as err:
            raise CodexAuthError("Device-code response has invalid interval") from err
        if interval <= 0:
            raise CodexAuthError("Device-code response has invalid interval")

        return DeviceCode(
            verification_url=CODEX_DEVICE_VERIFICATION_URL,
            user_code=user_code,
            device_auth_id=data["device_auth_id"],
            interval=interval,
        )

    async def complete_device_login(self, device_code: DeviceCode) -> CodexTokenData:
        """Poll until the device code is approved, then exchange it for tokens."""
        deadline = datetime.now(UTC) + DEVICE_AUTH_TIMEOUT

        while datetime.now(UTC) < deadline:
            try:
                code_response = await self._poll_device_token(device_code)
            except CodexAuthPending:
                await asyncio.sleep(device_code.interval)
                continue
            return await self.exchange_code_for_tokens(code_response)

        raise CodexAuthTimeout("Device authorization timed out")

    async def _poll_device_token(self, device_code: DeviceCode) -> dict[str, Any]:
        """Poll the device token endpoint once."""
        status, data = await self._post_json(
            f"{CODEX_DEVICE_API_BASE}/deviceauth/token",
            error_context="Device authorization",
            json={
                "device_auth_id": device_code.device_auth_id,
                "user_code": device_code.user_code,
            },
            headers=self._headers(),
        )
        if status in (403, 404):
            raise CodexAuthPending
        if status >= 400:
            raise CodexAuthError(
                f"Device authorization failed with status {status}: "
                f"{self._error_message(data)}"
            )
        return data

    async def exchange_code_for_tokens(
        self, code_response: dict[str, Any]
    ) -> CodexTokenData:
        """Exchange a device authorization code for OAuth tokens."""
        authorization_code = code_response.get("authorization_code")
        code_verifier = code_response.get("code_verifier")
        if not isinstance(authorization_code, str) or not isinstance(
            code_verifier, str
        ):
            raise CodexAuthError("Authorization response is missing required fields")

        status, data = await self._post_json(
            CODEX_TOKEN_URL,
            error_context="Token exchange",
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": CODEX_DEVICE_REDIRECT_URI,
                "client_id": CODEX_CLIENT_ID,
                "code_verifier": code_verifier,
            },
            headers=self._headers("application/x-www-form-urlencoded"),
        )
        if status >= 400:
            raise CodexAuthError(
                f"Token exchange failed with status {status}: "
                f"{self._error_message(data)}"
            )

        return token_data_from_response(data)

    async def refresh_tokens(self, token_data: CodexTokenData) -> CodexTokenData:
        """Refresh ChatGPT OAuth tokens."""
        status, data = await self._post_json(
            CODEX_TOKEN_URL,
            error_context="Token refresh",
            data={
                "client_id": CODEX_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": token_data.refresh_token,
            },
            headers=self._headers("application/x-www-form-urlencoded"),
        )
        if status in (400, 401) and self._error_code(data) == "invalid_grant":
            raise CodexAuthPermanentError(self._error_message(data))
        if status == 401:
            raise CodexAuthPermanentError(self._error_message(data))
        if status >= 400:
            raise CodexAuthError(
                f"Token refresh failed with status {status}: "
                f"{self._error_message(data)}"
            )

        refreshed = token_data_from_response(data, previous=token_data)
        if token_data.chatgpt_account_id:
            if not refreshed.chatgpt_account_id:
                raise CodexAuthPermanentError(
                    "Refreshed token is missing account identity"
                )
            if refreshed.chatgpt_account_id != token_data.chatgpt_account_id:
                raise CodexAuthPermanentError(
                    "Refreshed token belongs to another workspace"
                )
        return refreshed

    async def _post_json(
        self,
        url: str,
        *,
        error_context: str,
        **kwargs: Any,
    ) -> tuple[int, dict[str, Any]]:
        """POST to an auth endpoint and normalize transport/parsing errors."""
        try:
            kwargs.setdefault("timeout", AUTH_REQUEST_TIMEOUT)
            async with self._session.post(url, **kwargs) as response:
                status = response.status
                data = await self._json_response(response, error_context)
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            raise CodexAuthError(f"{error_context} failed: {err}") from err

        return status, data

    async def _json_response(
        self,
        response: aiohttp.ClientResponse,
        error_context: str,
    ) -> dict[str, Any]:
        """Read a JSON response body, preserving error text when possible."""
        try:
            data = await response.json(content_type=None)
        except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError) as err:
            text = await response.text()
            if response.status >= 400 and text:
                return {"error": text}
            raise CodexAuthError(
                f"{error_context} returned an invalid response"
            ) from err

        if not isinstance(data, dict):
            raise CodexAuthError(f"{error_context} returned an invalid response")
        return data

    @staticmethod
    def _error_code(data: dict[str, Any]) -> str | None:
        """Return a backend error code, if present."""
        error = data.get("error")
        if isinstance(error, dict):
            code = error.get("code") or error.get("error")
            return code if isinstance(code, str) else None
        if isinstance(error, str):
            return error
        return None

    @staticmethod
    def _error_message(data: dict[str, Any]) -> str:
        """Return a compact backend error message."""
        detail = data.get("detail")
        if isinstance(detail, str) and detail:
            return detail

        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code")
            return str(message or "Authentication failed")
        if isinstance(error, str):
            return error
        message = data.get("message")
        if isinstance(message, str) and message:
            return message
        return "Authentication failed"

    def _headers(self, content_type: str = "application/json") -> dict[str, str]:
        """Return standard Codex auth request headers."""
        return {
            "Content-Type": content_type,
            "User-Agent": USER_AGENT,
            "originator": ORIGINATOR,
        }
