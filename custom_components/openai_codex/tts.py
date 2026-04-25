"""Text-to-speech support for OpenAI Codex."""

from __future__ import annotations

import asyncio
from array import array
from collections.abc import AsyncGenerator, Mapping
import contextlib
import json
from typing import Any

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av.audio.resampler import AudioResampler

from homeassistant.components import ffmpeg
from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TTSAudioRequest,
    TTSAudioResponse,
    Voice,
)
from homeassistant.core import Event, HomeAssistant, callback
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
    EVENT_TTS_PREWARM,
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
FFMPEG_CHUNK_SIZE = 32_768
TTS_SESSION_IDLE_TIMEOUT = 120.0

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
        self._session_lock = asyncio.Lock()
        self._warm_session: ReusableTtsSession | None = None
        self._prewarm_task: asyncio.Task[None] | None = None
        self._idle_close_task: asyncio.Task[None] | None = None

    async def async_added_to_hass(self) -> None:
        """Register prewarm hooks."""
        await super().async_added_to_hass()
        self.async_on_remove(
            self.hass.bus.async_listen(EVENT_TTS_PREWARM, self._handle_prewarm_event)
        )
        self._schedule_prewarm(
            CODEX_TTS_DEFAULT_VOICE,
            CODEX_TTS_DEFAULT_LANGUAGE,
            "entity added",
        )

    async def async_will_remove_from_hass(self) -> None:
        """Close warm realtime resources."""
        if self._prewarm_task is not None:
            self._prewarm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._prewarm_task
        if self._idle_close_task is not None:
            self._idle_close_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._idle_close_task
        async with self._session_lock:
            await self._async_close_warm_session_locked()
        await super().async_will_remove_from_hass()

    @callback
    def _handle_prewarm_event(self, event: Event) -> None:
        """Handle integration-local TTS prewarm event."""
        if event.data.get("entry_id") not in (None, self.entry.entry_id):
            return
        voice = event.data.get("voice", CODEX_TTS_DEFAULT_VOICE)
        language = event.data.get("language", CODEX_TTS_DEFAULT_LANGUAGE)
        if not isinstance(voice, str) or voice not in CODEX_TTS_VOICES:
            voice = CODEX_TTS_DEFAULT_VOICE
        if (
            not isinstance(language, str)
            or language not in CODEX_TTS_SUPPORTED_LANGUAGES
        ):
            language = CODEX_TTS_DEFAULT_LANGUAGE
        self._schedule_prewarm(voice, language, "event")

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Return supported Codex realtime v2 voices."""
        return self._supported_voices

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Stream Codex realtime TTS as MP3."""
        message = "".join([chunk async for chunk in request.message_gen])
        stripped_message = message.strip()
        if not stripped_message:
            raise HomeAssistantError("Codex TTS message cannot be empty")

        voice = _resolve_voice(request.options, self.entry.options)
        session = _build_tts_session(voice, request.language)

        return TTSAudioResponse(
            "mp3",
            self._async_stream_tts_mp3(stripped_message, session, voice),
        )

    async def _async_stream_tts_mp3(
        self,
        message: str,
        session: dict[str, Any],
        voice: str,
    ) -> AsyncGenerator[bytes]:
        """Create a one-shot realtime session and stream encoded MP3 audio."""
        async with self._tts_semaphore:
            try:
                async for chunk in self._async_generate_tts_mp3(
                    message,
                    session,
                    voice,
                ):
                    yield chunk
            except ConfigEntryAuthFailed:
                self.entry.async_start_reauth(self.hass)
                raise

    async def _async_generate_tts_mp3(
        self,
        message: str,
        session: dict[str, Any],
        voice: str,
    ) -> AsyncGenerator[bytes]:
        """Create a realtime session and stream MP3 audio from WebRTC PCM."""
        started_at = asyncio.get_running_loop().time()
        _log_timing(started_at, "request start")
        reusable_session = await self._async_get_warm_session(
            voice,
            session,
            started_at,
        )
        completed = False
        try:
            pcm_stream = reusable_session.stream_pcm(message, session, started_at)
            async for mp3_chunk in _stream_pcm_as_mp3(
                self.hass,
                pcm_stream,
                started_at,
            ):
                yield mp3_chunk
            completed = True
        except BaseException:
            async with self._session_lock:
                if self._warm_session is reusable_session:
                    await self._async_close_warm_session_locked()
            raise
        finally:
            if completed:
                self._schedule_idle_close()

    @callback
    def _schedule_prewarm(self, voice: str, language: str, reason: str) -> None:
        """Schedule a background realtime session prewarm."""
        if self._prewarm_task is not None and not self._prewarm_task.done():
            return
        self._prewarm_task = self.hass.async_create_task(
            self._async_prewarm(voice, language, reason),
            "openai_codex_tts_prewarm",
        )

    async def _async_prewarm(self, voice: str, language: str, reason: str) -> None:
        """Prepare a reusable realtime session before TTS playback needs it."""
        session = _build_tts_session(voice, language)
        started_at = asyncio.get_running_loop().time()
        try:
            _log_timing(started_at, f"prewarm start ({reason})")
            await self._async_get_warm_session(voice, session, started_at)
            _log_timing(started_at, "prewarm ready")
            self._schedule_idle_close()
        except ConfigEntryAuthFailed:
            self.entry.async_start_reauth(self.hass)
        except Exception:
            LOGGER.debug("Codex TTS prewarm failed", exc_info=True)

    async def _async_get_warm_session(
        self,
        voice: str,
        session: dict[str, Any],
        started_at: float,
    ) -> "ReusableTtsSession":
        """Return a connected reusable realtime session."""
        async with self._session_lock:
            if self._idle_close_task is not None:
                self._idle_close_task.cancel()
                self._idle_close_task = None

            if (
                self._warm_session is not None
                and self._warm_session.voice == voice
                and not self._warm_session.closed
            ):
                _log_timing(started_at, "warm realtime session reused")
                return self._warm_session

            await self._async_close_warm_session_locked()
            warm_session = ReusableTtsSession(self.entry, voice)
            try:
                await warm_session.connect(session, started_at)
            except Exception:
                await warm_session.close()
                raise
            self._warm_session = warm_session
            return warm_session

    @callback
    def _schedule_idle_close(self) -> None:
        """Close the warm realtime session after a short idle period."""
        if self._idle_close_task is not None:
            self._idle_close_task.cancel()
        self._idle_close_task = self.hass.async_create_task(
            self._async_idle_close(),
            "openai_codex_tts_idle_close",
        )

    async def _async_idle_close(self) -> None:
        """Close the reusable realtime session after it has been idle."""
        try:
            await asyncio.sleep(TTS_SESSION_IDLE_TIMEOUT)
            async with self._session_lock:
                await self._async_close_warm_session_locked()
        except asyncio.CancelledError:
            raise

    async def _async_close_warm_session_locked(self) -> None:
        """Close the current warm session while holding the session lock."""
        if self._warm_session is None:
            return
        session = self._warm_session
        self._warm_session = None
        await session.close()


class ReusableTtsSession:
    """Reusable Codex realtime WebRTC session for TTS."""

    def __init__(self, entry: OpenAICodexConfigEntry, voice: str) -> None:
        """Initialize the reusable session."""
        self.entry = entry
        self.voice = voice
        self.peer_connection: RTCPeerConnection | None = None
        self.websocket: aiohttp.ClientWebSocketResponse | None = None
        self.audio_tracks: asyncio.Queue[MediaStreamTrack] = asyncio.Queue(maxsize=1)
        self.webrtc_audio: asyncio.Queue[
            tuple[bytes, int, int] | BaseException | None
        ] = asyncio.Queue(maxsize=400)
        self.stop_audio = asyncio.Event()
        self.accept_audio = asyncio.Event()
        self.track_reader: asyncio.Task[None] | None = None
        self.closed = False

    async def connect(self, session: dict[str, Any], started_at: float) -> None:
        """Connect the WebRTC and sideband transports."""
        runtime = self.entry.runtime_data
        peer_connection = RTCPeerConnection()
        self.peer_connection = peer_connection

        @peer_connection.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            LOGGER.debug("Codex TTS WebRTC track received: %s", track.kind)
            if track.kind == "audio":
                with contextlib.suppress(asyncio.QueueFull):
                    self.audio_tracks.put_nowait(track)

        LOGGER.debug("Creating Codex TTS realtime call with voice %s", self.voice)
        peer_connection.addTransceiver("audio", direction="recvonly")
        offer = await peer_connection.createOffer()
        await peer_connection.setLocalDescription(offer)
        local_description = peer_connection.localDescription
        if local_description is None:
            raise HomeAssistantError("Codex TTS failed to create an SDP offer")
        _log_timing(started_at, "local SDP offer ready")

        call = await runtime.async_create_realtime_call(
            local_description.sdp,
            session,
        )
        _log_timing(started_at, "realtime call created")
        self.track_reader = asyncio.create_task(
            _read_webrtc_audio(
                self.audio_tracks,
                self.webrtc_audio,
                self.stop_audio,
                self.accept_audio,
            )
        )
        await peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=call.answer_sdp, type="answer")
        )
        _log_timing(started_at, "remote SDP answer applied")

        LOGGER.debug("Connecting Codex TTS realtime sideband")
        self.websocket = await runtime.async_connect_realtime_sideband(call.call_id)
        _log_timing(started_at, "sideband connected")
        await self.websocket.send_str(
            json.dumps({"type": "session.update", "session": session})
        )

    async def stream_pcm(
        self,
        message: str,
        session: dict[str, Any],
        started_at: float,
    ) -> AsyncGenerator[bytes]:
        """Send one TTS request and stream its PCM response."""
        if self.websocket is None:
            raise HomeAssistantError("Codex TTS session is not connected")
        if self.track_reader is not None and self.track_reader.done():
            raise HomeAssistantError("Codex TTS WebRTC audio track is closed")
        _clear_queue(self.webrtc_audio)
        self.accept_audio.set()
        try:
            await self.websocket.send_str(
                json.dumps({"type": "session.update", "session": session})
            )
            await self.websocket.send_str(
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
            await self.websocket.send_str(json.dumps({"type": "response.create"}))
            LOGGER.debug("Codex TTS response.create sent")
            _log_timing(started_at, "response.create sent")

            async for chunk in _stream_realtime_pcm(
                self.websocket,
                self.webrtc_audio,
                started_at,
            ):
                yield chunk
        finally:
            self.accept_audio.clear()
            _clear_queue(self.webrtc_audio)

    async def close(self) -> None:
        """Close realtime resources."""
        self.closed = True
        self.stop_audio.set()
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None
        if self.peer_connection is not None:
            await self.peer_connection.close()
            self.peer_connection = None
        if self.track_reader is not None:
            self.track_reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.track_reader
            self.track_reader = None


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


async def _stream_realtime_pcm(
    websocket: aiohttp.ClientWebSocketResponse,
    webrtc_audio: asyncio.Queue[tuple[bytes, int, int] | BaseException | None],
    started_at: float,
) -> AsyncGenerator[bytes]:
    """Stream PCM16 audio chunks from the realtime WebRTC audio track."""
    sample_rate = CODEX_REALTIME_AUDIO_RATE
    channels = CODEX_REALTIME_AUDIO_CHANNELS
    response_id: str | None = None
    first_audio_received = False
    first_audio_logged = False
    speech_started = False
    trailing_silence_samples = 0
    response_done = False
    completion_deadline: float | None = None
    track_ended = False
    pcm_bytes = 0
    loop = asyncio.get_running_loop()
    first_audio_deadline = loop.time() + FIRST_AUDIO_TIMEOUT
    last_audio_at: float | None = None
    deadline = loop.time() + TOTAL_SYNTHESIS_TIMEOUT
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
            elif not response_done and last_audio_at is not None:
                idle_remaining = (
                    AUDIO_IDLE_COMPLETION_TIMEOUT - (now - last_audio_at)
                )
                if idle_remaining <= 0:
                    _log_timing(started_at, "audio idle completion")
                    return
                remaining = min(remaining, idle_remaining)
            if response_done and completion_deadline is not None:
                if now >= completion_deadline:
                    if not first_audio_received:
                        raise HomeAssistantError(
                            "Codex TTS response completed without audio"
                        )
                    _log_timing(
                        started_at,
                        "PCM stream complete",
                        "bytes=%s",
                        pcm_bytes,
                    )
                    return
                remaining = min(remaining, completion_deadline - now)

            timeout = remaining
            done, _ = await asyncio.wait(
                {websocket_receive, audio_receive},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                if response_done:
                    if not first_audio_received:
                        raise HomeAssistantError(
                            "Codex TTS response completed without audio"
                        )
                    _log_timing(
                        started_at,
                        "PCM stream complete",
                        "bytes=%s",
                        pcm_bytes,
                    )
                    return
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
                    if first_audio_received:
                        _log_timing(
                            started_at,
                            "WebRTC audio track ended",
                            "bytes=%s",
                            pcm_bytes,
                        )
                        return
                elif isinstance(audio_item, BaseException):
                    raise HomeAssistantError(
                        "Codex TTS WebRTC audio failed"
                    ) from audio_item
                else:
                    chunk, chunk_sample_rate, chunk_channels = audio_item
                    (
                        sample_rate,
                        channels,
                        first_audio_received,
                        speech_started,
                        trailing_silence_samples,
                    ) = _process_audio_chunk(
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
                    if not first_audio_logged:
                        _log_timing(started_at, "first WebRTC PCM chunk")
                        first_audio_logged = True
                    pcm_bytes += len(chunk)
                    yield chunk
                    if (
                        speech_started
                        and trailing_silence_samples
                        >= int(sample_rate * TRAILING_SILENCE_SECONDS)
                    ):
                        _log_timing(
                            started_at,
                            "trailing silence completion",
                            "bytes=%s",
                            pcm_bytes,
                        )
                        return
                    if response_done:
                        completion_deadline = loop.time() + COMPLETION_AUDIO_GRACE
                    if track_ended:
                        _log_timing(
                            started_at,
                            "WebRTC audio track ended",
                            "bytes=%s",
                            pcm_bytes,
                        )
                        return

            if websocket_receive in done:
                message = websocket_receive.result()
                websocket_receive = asyncio.create_task(websocket.receive())

                if message.type == aiohttp.WSMsgType.TEXT:
                    event = _decode_event(message.data)
                elif message.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                ):
                    if response_done and first_audio_received:
                        _log_timing(
                            started_at,
                            "sideband closed after response.done",
                            "bytes=%s",
                            pcm_bytes,
                        )
                        return
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
                    _log_timing(started_at, "response.created")
                    continue

                if event_type in (
                    "response.output_audio.delta",
                    "response.audio.delta",
                ):
                    LOGGER.debug("Ignoring sideband Codex TTS audio delta")
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
                    _log_timing(started_at, "response.done")
                    response_done = True
                    completion_deadline = loop.time() + COMPLETION_AUDIO_GRACE
                    if track_ended and first_audio_received:
                        _log_timing(
                            started_at,
                            "PCM stream complete",
                            "bytes=%s",
                            pcm_bytes,
                        )
                        return
                    continue

                if event_type == "response.cancelled":
                    raise HomeAssistantError("Codex TTS response was cancelled")

                if event_type == "error":
                    raise HomeAssistantError(
                        f"Codex TTS error: {_event_error_message(event)}"
                    )
    finally:
        for task in (websocket_receive, audio_receive):
            task.cancel()
        for task in (websocket_receive, audio_receive):
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def _stream_pcm_as_mp3(
    hass: HomeAssistant,
    pcm_stream: AsyncGenerator[bytes],
    started_at: float,
) -> AsyncGenerator[bytes]:
    """Encode PCM16 24 kHz mono audio to MP3 while streaming."""
    manager = ffmpeg.get_ffmpeg_manager(hass)
    command = [
        manager.binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(CODEX_REALTIME_AUDIO_RATE),
        "-ac",
        str(CODEX_REALTIME_AUDIO_CHANNELS),
        "-i",
        "pipe:0",
        "-f",
        "mp3",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "0",
        "-flush_packets",
        "1",
        "pipe:1",
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    _log_timing(started_at, "ffmpeg MP3 encoder started")
    writer_task = asyncio.create_task(_write_pcm_to_ffmpeg(process, pcm_stream))
    first_mp3_logged = False
    mp3_bytes = 0

    try:
        while chunk := await process.stdout.read(FFMPEG_CHUNK_SIZE):
            if not first_mp3_logged:
                _log_timing(started_at, "first MP3 chunk")
                first_mp3_logged = True
            mp3_bytes += len(chunk)
            yield chunk

        await writer_task
        return_code = await process.wait()
        if return_code != 0:
            stderr = (await process.stderr.read()).decode(errors="replace").strip()
            raise HomeAssistantError(
                f"Codex TTS MP3 encoding failed with status {return_code}: {stderr}"
            )
        _log_timing(started_at, "MP3 stream complete", "bytes=%s", mp3_bytes)
    finally:
        if not writer_task.done():
            writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await writer_task
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            with contextlib.suppress(ProcessLookupError, asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=2)
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(ProcessLookupError):
                await process.wait()


async def _write_pcm_to_ffmpeg(
    process: asyncio.subprocess.Process,
    pcm_stream: AsyncGenerator[bytes],
) -> None:
    """Write realtime PCM chunks into ffmpeg stdin."""
    assert process.stdin is not None
    try:
        async for chunk in pcm_stream:
            process.stdin.write(chunk)
            await process.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        if process.returncode not in (0, None):
            raise
    finally:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            process.stdin.close()
            await process.stdin.wait_closed()


def _clear_queue(queue: asyncio.Queue[Any]) -> None:
    """Remove queued items without waiting."""
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


async def _put_audio_chunk(
    queue: asyncio.Queue[tuple[bytes, int, int] | BaseException | None],
    item: tuple[bytes, int, int] | BaseException | None,
) -> None:
    """Put an audio item, dropping the oldest item if the queue is full."""
    if queue.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    await queue.put(item)


async def _read_webrtc_audio(
    audio_tracks: asyncio.Queue[MediaStreamTrack],
    audio_chunks: asyncio.Queue[tuple[bytes, int, int] | BaseException | None],
    stop_audio: asyncio.Event,
    accept_audio: asyncio.Event,
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
                if pcm and accept_audio.is_set():
                    await _put_audio_chunk(
                        audio_chunks,
                        (pcm, resampled.sample_rate, channels),
                    )
    except MediaStreamError:
        await _put_audio_chunk(audio_chunks, None)
    except (Exception, asyncio.CancelledError) as err:
        if not isinstance(err, asyncio.CancelledError):
            await _put_audio_chunk(audio_chunks, err)
        raise
    finally:
        get_track.cancel()
        stop_wait.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await get_track
        with contextlib.suppress(asyncio.CancelledError):
            await stop_wait


def _process_audio_chunk(
    chunk: bytes,
    chunk_sample_rate: int,
    chunk_channels: int,
    sample_rate: int,
    channels: int,
    first_audio_received: bool,
    speech_started: bool,
    trailing_silence_samples: int,
) -> tuple[int, int, bool, bool, int]:
    """Validate an audio chunk and update stream state."""
    if first_audio_received and (
        chunk_sample_rate != sample_rate or chunk_channels != channels
    ):
        raise HomeAssistantError(
            "Codex TTS returned mixed audio sample rates or channels"
        )

    chunk_is_silent = _pcm16_is_silent(chunk)
    if not chunk_is_silent:
        speech_started = True
        trailing_silence_samples = 0
    elif speech_started:
        trailing_silence_samples += len(chunk) // (
            chunk_channels * CODEX_REALTIME_SAMPLE_WIDTH_BYTES
        )

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


def _log_timing(
    started_at: float,
    label: str,
    message: str = "",
    *args: Any,
) -> None:
    """Log a TTS timing marker relative to request start."""
    elapsed = asyncio.get_running_loop().time() - started_at
    if message:
        LOGGER.debug("Codex TTS timing %.3fs: %s; " + message, elapsed, label, *args)
    else:
        LOGGER.debug("Codex TTS timing %.3fs: %s", elapsed, label)
