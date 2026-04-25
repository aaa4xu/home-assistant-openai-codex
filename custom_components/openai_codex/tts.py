"""Text-to-speech support for OpenAI Codex."""

from __future__ import annotations

import asyncio
from array import array
import base64
import binascii
from collections.abc import Mapping
import contextlib
import io
import json
import wave
from typing import Any

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av.audio.resampler import AudioResampler

from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryAuthFailed, HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .client import OpenAICodexConfigEntry
from .const import (
    CODEX_REALTIME_AUDIO_CHANNELS,
    CODEX_REALTIME_AUDIO_RATE,
    CODEX_REALTIME_MODEL,
    CODEX_REALTIME_SAMPLE_WIDTH_BYTES,
    CODEX_TTS_DEFAULT_LANGUAGE,
    CODEX_TTS_DEFAULT_VOICE,
    CODEX_TTS_SUPPORTED_LANGUAGES,
    CODEX_TTS_SUPPORTED_OPTIONS,
    CODEX_TTS_VOICES,
    CONF_TTS_VOICE,
    DEFAULT_TTS_NAME,
    LOGGER,
)
from .entity import OpenAICodexBaseEntity

TTS_CONCURRENCY = 2
FIRST_AUDIO_TIMEOUT = 20.0
TOTAL_SYNTHESIS_TIMEOUT = 120.0
COMPLETION_AUDIO_GRACE = 2.0
AUDIO_IDLE_COMPLETION_TIMEOUT = 3.0
TRAILING_SILENCE_SECONDS = 1.0
PCM16_SILENCE_PEAK = 64

TTS_INSTRUCTIONS = (
    "You are a text-to-speech renderer. Read the user's text aloud exactly. "
    "Do not answer questions, do not explain, do not add words, and do not "
    "call tools."
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAICodexConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up TTS entities."""
    async_add_entities([OpenAICodexTTSEntity(config_entry)])


class OpenAICodexTTSEntity(TextToSpeechEntity, OpenAICodexBaseEntity):
    """OpenAI Codex TTS entity."""

    _attr_should_poll = False
    _attr_has_entity_name = False
    _attr_name = DEFAULT_TTS_NAME
    _attr_supported_languages = list(CODEX_TTS_SUPPORTED_LANGUAGES)
    _attr_default_language = CODEX_TTS_DEFAULT_LANGUAGE
    _attr_supported_options = list(CODEX_TTS_SUPPORTED_OPTIONS)
    _attr_default_options = {ATTR_VOICE: CODEX_TTS_DEFAULT_VOICE}

    _supported_voices = [Voice(voice, voice.title()) for voice in CODEX_TTS_VOICES]

    def __init__(self, entry: OpenAICodexConfigEntry) -> None:
        """Initialize the TTS entity."""
        super().__init__(entry, unique_id_suffix="tts")
        self._tts_semaphore = asyncio.Semaphore(TTS_CONCURRENCY)

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Return supported Codex realtime v2 voices."""
        return self._supported_voices

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any],
    ) -> TtsAudioType:
        """Generate TTS audio from Codex realtime."""
        stripped_message = message.strip()
        if not stripped_message:
            raise HomeAssistantError("Codex TTS message cannot be empty")

        voice = _resolve_voice(options, self.entry.options)
        session = _build_tts_session(voice, language)

        async with self._tts_semaphore:
            try:
                return await self._async_generate_tts_audio(
                    stripped_message,
                    session,
                    voice,
                )
            except ConfigEntryAuthFailed:
                self.entry.async_start_reauth(self.hass)
                raise

    async def _async_generate_tts_audio(
        self,
        message: str,
        session: dict[str, Any],
        voice: str,
    ) -> TtsAudioType:
        """Create a one-shot realtime session and collect audio."""
        runtime = self.entry.runtime_data
        peer_connection = RTCPeerConnection()
        audio_tracks: asyncio.Queue[MediaStreamTrack] = asyncio.Queue(maxsize=1)
        websocket: aiohttp.ClientWebSocketResponse | None = None

        @peer_connection.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            LOGGER.debug("Codex TTS WebRTC track received: %s", track.kind)
            if track.kind == "audio":
                with contextlib.suppress(asyncio.QueueFull):
                    audio_tracks.put_nowait(track)

        try:
            LOGGER.debug("Creating Codex TTS realtime call with voice %s", voice)
            peer_connection.addTransceiver("audio", direction="recvonly")
            offer = await peer_connection.createOffer()
            await peer_connection.setLocalDescription(offer)
            local_description = peer_connection.localDescription
            if local_description is None:
                raise HomeAssistantError("Codex TTS failed to create an SDP offer")

            call = await runtime.async_create_realtime_call(
                local_description.sdp,
                session,
            )
            await peer_connection.setRemoteDescription(
                RTCSessionDescription(sdp=call.answer_sdp, type="answer")
            )

            LOGGER.debug("Connecting Codex TTS realtime sideband")
            websocket = await runtime.async_connect_realtime_sideband(
                call.call_id,
            )
            await websocket.send_str(
                json.dumps({"type": "session.update", "session": session})
            )
            await websocket.send_str(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": message,
                                }
                            ],
                        },
                    }
                )
            )
            await websocket.send_str(json.dumps({"type": "response.create"}))
            LOGGER.debug("Codex TTS response.create sent")

            pcm, sample_rate, channels = await _collect_realtime_audio(
                websocket,
                audio_tracks,
            )
            return (
                "wav",
                _pcm16_to_wav(pcm, sample_rate=sample_rate, channels=channels),
            )
        finally:
            if websocket is not None:
                await websocket.close()
            await peer_connection.close()


def _resolve_voice(
    options: dict[str, Any],
    entry_options: Mapping[str, Any],
) -> str:
    """Resolve and validate the requested TTS voice."""
    voice = (
        options.get(ATTR_VOICE)
        or options.get(CONF_TTS_VOICE)
        or entry_options.get(CONF_TTS_VOICE)
        or CODEX_TTS_DEFAULT_VOICE
    )
    if not isinstance(voice, str) or voice not in CODEX_TTS_VOICES:
        raise HomeAssistantError(f"Unsupported Codex TTS voice: {voice}")
    return voice


def _build_tts_session(voice: str, language: str) -> dict[str, Any]:
    """Build the Codex realtime session payload for TTS."""
    instructions = (
        f"{TTS_INSTRUCTIONS}\n"
        f"The requested language tag is {language}; preserve the language of "
        "the input text and pronounce it naturally."
    )
    return {
        "type": "realtime",
        "model": CODEX_REALTIME_MODEL,
        "instructions": instructions,
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": CODEX_REALTIME_AUDIO_RATE,
                }
            },
            "output": {
                "format": {
                    "type": "audio/pcm",
                    "rate": CODEX_REALTIME_AUDIO_RATE,
                },
                "voice": voice,
            },
        },
    }


async def _collect_realtime_audio(
    websocket: aiohttp.ClientWebSocketResponse,
    audio_tracks: asyncio.Queue[MediaStreamTrack],
) -> tuple[bytes, int, int]:
    """Collect PCM16 audio deltas from the realtime sideband websocket."""
    audio_chunks: list[bytes] = []
    sample_rate = CODEX_REALTIME_AUDIO_RATE
    channels = CODEX_REALTIME_AUDIO_CHANNELS
    response_id: str | None = None
    first_audio_received = False
    speech_started = False
    trailing_silence_samples = 0
    response_done = False
    completion_deadline: float | None = None
    track_ended = False
    loop = asyncio.get_running_loop()
    first_audio_deadline = loop.time() + FIRST_AUDIO_TIMEOUT
    last_audio_at: float | None = None
    deadline = loop.time() + TOTAL_SYNTHESIS_TIMEOUT
    stop_audio = asyncio.Event()
    webrtc_audio: asyncio.Queue[
        tuple[bytes, int, int] | BaseException | None
    ] = asyncio.Queue()
    track_reader = asyncio.create_task(
        _read_webrtc_audio(audio_tracks, webrtc_audio, stop_audio)
    )
    websocket_receive = asyncio.create_task(websocket.receive())
    audio_receive = asyncio.create_task(webrtc_audio.get())

    try:
        while True:
            now = loop.time()
            remaining = deadline - now
            if remaining <= 0:
                raise HomeAssistantError("Timed out waiting for Codex TTS audio")
            if not first_audio_received:
                first_audio_remaining = first_audio_deadline - now
                if first_audio_remaining <= 0:
                    raise HomeAssistantError(
                        "Timed out waiting for first Codex TTS audio"
                    )
                remaining = min(remaining, first_audio_remaining)
            elif not speech_started and now >= first_audio_deadline:
                raise HomeAssistantError(
                    "Timed out waiting for audible Codex TTS audio"
                )
            elif not response_done and last_audio_at is not None:
                idle_remaining = (
                    AUDIO_IDLE_COMPLETION_TIMEOUT - (now - last_audio_at)
                )
                if idle_remaining <= 0:
                    return b"".join(audio_chunks), sample_rate, channels
                remaining = min(remaining, idle_remaining)
            if response_done and completion_deadline is not None:
                if now >= completion_deadline:
                    if not audio_chunks:
                        raise HomeAssistantError(
                            "Codex TTS response completed without audio"
                        )
                    return b"".join(audio_chunks), sample_rate, channels
                remaining = min(remaining, completion_deadline - now)

            timeout = remaining
            done, _ = await asyncio.wait(
                {websocket_receive, audio_receive},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                if response_done:
                    if not audio_chunks:
                        raise HomeAssistantError(
                            "Codex TTS response completed without audio"
                        )
                    return b"".join(audio_chunks), sample_rate, channels
                if not first_audio_received:
                    raise HomeAssistantError(
                        "Timed out waiting for first Codex TTS audio"
                    )
                raise HomeAssistantError("Timed out waiting for Codex TTS completion")

            if audio_receive in done:
                audio_item = audio_receive.result()
                audio_receive = asyncio.create_task(webrtc_audio.get())
                if audio_item is None:
                    LOGGER.debug("Codex TTS WebRTC audio track ended")
                    track_ended = True
                    if audio_chunks:
                        return b"".join(audio_chunks), sample_rate, channels
                elif isinstance(audio_item, BaseException):
                    raise HomeAssistantError("Codex TTS WebRTC audio failed") from audio_item
                else:
                    chunk, chunk_sample_rate, chunk_channels = audio_item
                    (
                        sample_rate,
                        channels,
                        first_audio_received,
                        speech_started,
                        trailing_silence_samples,
                    ) = _append_audio_chunk(
                        audio_chunks,
                        chunk,
                        chunk_sample_rate,
                        chunk_channels,
                        sample_rate,
                        channels,
                        first_audio_received,
                        speech_started,
                        trailing_silence_samples,
                    )
                    last_audio_at = loop.time()
                    if (
                        speech_started
                        and trailing_silence_samples
                        >= int(sample_rate * TRAILING_SILENCE_SECONDS)
                    ):
                        return b"".join(audio_chunks), sample_rate, channels
                    if response_done:
                        completion_deadline = loop.time() + COMPLETION_AUDIO_GRACE
                    if track_ended:
                        return b"".join(audio_chunks), sample_rate, channels

            if websocket_receive in done:
                message = websocket_receive.result()
                websocket_receive = asyncio.create_task(websocket.receive())

                if message.type == aiohttp.WSMsgType.TEXT:
                    event = _decode_event(message.data)
                elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                    raise HomeAssistantError(
                        "Codex TTS websocket closed before completion"
                    )
                elif message.type == aiohttp.WSMsgType.ERROR:
                    raise HomeAssistantError("Codex TTS websocket failed")
                else:
                    continue

                event_type = event.get("type")
                LOGGER.debug("Codex TTS realtime event: %s", event_type)
                if event_type == "response.created":
                    response_id = _event_response_id(event) or response_id
                    LOGGER.debug("Codex TTS response created")
                    continue

                if event_type in ("response.output_audio.delta", "response.audio.delta"):
                    if response_id and (event_id := _event_response_id(event)):
                        if event_id != response_id:
                            continue
                    chunk, chunk_sample_rate, chunk_channels = _audio_chunk_from_event(event)
                    (
                        sample_rate,
                        channels,
                        first_audio_received,
                        speech_started,
                        trailing_silence_samples,
                    ) = _append_audio_chunk(
                        audio_chunks,
                        chunk,
                        chunk_sample_rate,
                        chunk_channels,
                        sample_rate,
                        channels,
                        first_audio_received,
                        speech_started,
                        trailing_silence_samples,
                    )
                    last_audio_at = loop.time()
                    continue

                if event_type in (
                    "response.output_audio_transcript.delta",
                    "response.output_audio_transcript.done",
                ):
                    LOGGER.debug("Received Codex TTS transcript event: %s", event_type)
                    continue

                if event_type == "response.done":
                    if response_id and (event_id := _event_response_id(event)):
                        if event_id != response_id:
                            continue
                    LOGGER.debug("Codex TTS response done")
                    response_done = True
                    completion_deadline = loop.time() + COMPLETION_AUDIO_GRACE
                    if track_ended and audio_chunks:
                        return b"".join(audio_chunks), sample_rate, channels
                    continue

                if event_type == "response.cancelled":
                    raise HomeAssistantError("Codex TTS response was cancelled")

                if event_type == "error":
                    raise HomeAssistantError(
                        f"Codex TTS error: {_event_error_message(event)}"
                    )
    finally:
        stop_audio.set()
        for task in (websocket_receive, audio_receive, track_reader):
            task.cancel()
        for task in (websocket_receive, audio_receive, track_reader):
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def _read_webrtc_audio(
    audio_tracks: asyncio.Queue[MediaStreamTrack],
    audio_chunks: asyncio.Queue[tuple[bytes, int, int] | BaseException | None],
    stop_audio: asyncio.Event,
) -> None:
    """Read WebRTC output audio frames into PCM16 chunks."""
    get_track = asyncio.create_task(audio_tracks.get())
    stop_wait = asyncio.create_task(stop_audio.wait())
    try:
        done, _ = await asyncio.wait(
            {get_track, stop_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_wait in done:
            return
        track = get_track.result()
        LOGGER.debug("Codex TTS reading WebRTC audio track")
        resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=CODEX_REALTIME_AUDIO_RATE,
        )
        while not stop_audio.is_set():
            frame = await track.recv()
            for resampled in resampler.resample(frame):
                channels = len(resampled.layout.channels)
                expected_size = (
                    resampled.samples * channels * CODEX_REALTIME_SAMPLE_WIDTH_BYTES
                )
                pcm = bytes(resampled.planes[0])[:expected_size]
                if pcm:
                    await audio_chunks.put((pcm, resampled.sample_rate, channels))
    except MediaStreamError:
        await audio_chunks.put(None)
    except (Exception, asyncio.CancelledError) as err:
        if not isinstance(err, asyncio.CancelledError):
            await audio_chunks.put(err)
        raise
    finally:
        get_track.cancel()
        stop_wait.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await get_track
        with contextlib.suppress(asyncio.CancelledError):
            await stop_wait


def _append_audio_chunk(
    audio_chunks: list[bytes],
    chunk: bytes,
    chunk_sample_rate: int,
    chunk_channels: int,
    sample_rate: int,
    channels: int,
    first_audio_received: bool,
    speech_started: bool,
    trailing_silence_samples: int,
) -> tuple[int, int, bool, bool, int]:
    """Append an audio chunk after validating the stream shape."""
    if audio_chunks and (
        chunk_sample_rate != sample_rate or chunk_channels != channels
    ):
        raise HomeAssistantError(
            "Codex TTS returned mixed audio sample rates or channels"
        )

    chunk_is_silent = _pcm16_is_silent(chunk)
    if not speech_started and chunk_is_silent:
        if not first_audio_received:
            LOGGER.debug("Received first silent Codex TTS audio chunk")
        return chunk_sample_rate, chunk_channels, True, False, 0

    audio_chunks.append(chunk)
    if chunk_is_silent:
        trailing_silence_samples += len(chunk) // (
            chunk_channels * CODEX_REALTIME_SAMPLE_WIDTH_BYTES
        )
    else:
        speech_started = True
        trailing_silence_samples = 0

    if not first_audio_received:
        LOGGER.debug("Received first Codex TTS audio chunk")
    return (
        chunk_sample_rate,
        chunk_channels,
        True,
        speech_started,
        trailing_silence_samples,
    )


def _pcm16_is_silent(pcm: bytes) -> bool:
    """Return whether a PCM16 little-endian chunk is effectively silent."""
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % CODEX_REALTIME_SAMPLE_WIDTH_BYTES)])
    return not samples or max(abs(sample) for sample in samples) <= PCM16_SILENCE_PEAK


def _decode_event(data: str) -> dict[str, Any]:
    """Decode a realtime websocket event."""
    try:
        event = json.loads(data)
    except json.JSONDecodeError as err:
        raise HomeAssistantError("Codex TTS emitted invalid JSON") from err
    if not isinstance(event, dict):
        raise HomeAssistantError("Codex TTS emitted a non-object event")
    return event


def _event_response_id(event: dict[str, Any]) -> str | None:
    """Return response id from a realtime event when present."""
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return response["id"]
    response_id = event.get("response_id")
    return response_id if isinstance(response_id, str) else None


def _audio_chunk_from_event(event: dict[str, Any]) -> tuple[bytes, int, int]:
    """Decode one realtime audio delta event."""
    delta = event.get("delta")
    if not isinstance(delta, str) or not delta:
        raise HomeAssistantError("Codex TTS audio event was missing delta")

    try:
        pcm = base64.b64decode(delta, validate=True)
    except (binascii.Error, ValueError) as err:
        raise HomeAssistantError("Codex TTS emitted invalid audio base64") from err

    sample_rate = _positive_int(
        event.get("sample_rate"),
        CODEX_REALTIME_AUDIO_RATE,
    )
    channels = _positive_int(
        event.get("channels", event.get("num_channels")),
        CODEX_REALTIME_AUDIO_CHANNELS,
    )
    frame_width = channels * CODEX_REALTIME_SAMPLE_WIDTH_BYTES
    if len(pcm) % frame_width:
        raise HomeAssistantError("Codex TTS emitted malformed PCM16 audio")
    return pcm, sample_rate, channels


def _positive_int(value: Any, default: int) -> int:
    """Return a positive int or a default."""
    if isinstance(value, int) and value > 0:
        return value
    return default


def _event_error_message(event: dict[str, Any]) -> str:
    """Extract a provider error message from a realtime error event."""
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code")
        if isinstance(message, str) and message:
            return message
    if isinstance(error, str) and error:
        return error
    return "unknown error"


def _pcm16_to_wav(pcm: bytes, *, sample_rate: int, channels: int) -> bytes:
    """Encode little-endian PCM16 data as a WAV file."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(CODEX_REALTIME_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return wav_buffer.getvalue()
