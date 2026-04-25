"""Speech-to-text support for OpenAI Codex."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
import contextlib
from fractions import Fraction
import io
import json
import wave
from typing import Any

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler

from homeassistant.components import stt
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .client import OpenAICodexConfigEntry
from .const import (
    CODEX_REALTIME_AUDIO_CHANNELS,
    CODEX_REALTIME_AUDIO_RATE,
    CODEX_REALTIME_SAMPLE_WIDTH_BYTES,
    CODEX_STT_SUPPORTED_LANGUAGES,
    DEFAULT_STT_NAME,
    EVENT_TTS_PREWARM,
    LOGGER,
    RECOMMENDED_STT_MODEL,
)
from .entity import OpenAICodexBaseEntity

STT_FRAME_DURATION_SECONDS = 0.02
STT_FRAME_SAMPLES = int(CODEX_REALTIME_AUDIO_RATE * STT_FRAME_DURATION_SECONDS)
STT_TRAILING_SILENCE_SECONDS = 0.8
STT_TOTAL_TIMEOUT = 45.0
STT_COMPLETION_TIMEOUT_AFTER_AUDIO = 15.0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAICodexConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up STT entities."""
    async_add_entities([OpenAICodexSTTEntity(config_entry)])


class OpenAICodexSTTEntity(stt.SpeechToTextEntity, OpenAICodexBaseEntity):
    """OpenAI Codex speech-to-text entity."""

    _attr_should_poll = False
    _attr_has_entity_name = False
    _attr_name = DEFAULT_STT_NAME

    def __init__(self, entry: OpenAICodexConfigEntry) -> None:
        """Initialize the STT entity."""
        super().__init__(entry, unique_id_suffix="stt")

    @property
    def supported_languages(self) -> list[str]:
        """Return supported transcription languages."""
        return list(CODEX_STT_SUPPORTED_LANGUAGES)

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return supported audio formats."""
        return [stt.AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return supported audio codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return supported bit rates."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return supported sample rates."""
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return supported channel counts."""
        return [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self,
        metadata: stt.SpeechMetadata,
        stream: AsyncIterable[bytes],
    ) -> stt.SpeechResult:
        """Process an audio stream through Codex realtime transcription."""
        self.hass.bus.async_fire(
            EVENT_TTS_PREWARM,
            {"entry_id": self.entry.entry_id},
        )
        if not self.check_metadata(metadata):
            LOGGER.warning(
                "Codex STT received unsupported audio metadata: %s",
                metadata,
            )
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        try:
            audio = _extract_pcm16_mono(await _read_audio_stream(stream), metadata)
            if not audio:
                return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

            transcript = await self._async_transcribe_audio(audio, metadata)
        except ConfigEntryAuthFailed:
            self.entry.async_start_reauth(self.hass)
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
        except Exception:
            LOGGER.exception("Error during Codex realtime STT")
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        if transcript:
            return stt.SpeechResult(transcript, stt.SpeechResultState.SUCCESS)
        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

    async def _async_transcribe_audio(
        self,
        pcm16_mono: bytes,
        metadata: stt.SpeechMetadata,
    ) -> str | None:
        """Create a one-shot realtime transcription session."""
        session = _build_transcription_session()
        runtime = self.entry.runtime_data
        source_rate = int(metadata.sample_rate.value)
        audio = _resample_pcm16_mono(
            pcm16_mono,
            source_rate=source_rate,
            target_rate=CODEX_REALTIME_AUDIO_RATE,
        )
        audio += _silence_pcm16(STT_TRAILING_SILENCE_SECONDS)

        peer_connection = RTCPeerConnection()
        websocket: aiohttp.ClientWebSocketResponse | None = None
        track = BufferedPCM16AudioTrack(audio)

        try:
            peer_connection.addTrack(track)
            offer = await peer_connection.createOffer()
            await peer_connection.setLocalDescription(offer)
            local_description = peer_connection.localDescription
            if local_description is None:
                raise HomeAssistantError("Codex STT failed to create an SDP offer")

            call = await runtime.async_create_realtime_call(
                local_description.sdp,
                session,
            )
            await peer_connection.setRemoteDescription(
                RTCSessionDescription(sdp=call.answer_sdp, type="answer")
            )

            websocket = await runtime.async_connect_realtime_sideband(call.call_id)
            await websocket.send_str(
                json.dumps({"type": "session.update", "session": session})
            )
            return await _collect_transcript(websocket, track)
        finally:
            if websocket is not None:
                await websocket.close()
            await peer_connection.close()


class BufferedPCM16AudioTrack(MediaStreamTrack):
    """Replay buffered PCM16 mono audio as a paced WebRTC audio track."""

    kind = "audio"

    def __init__(self, pcm16_mono: bytes) -> None:
        """Initialize the audio track."""
        super().__init__()
        self._pcm = pcm16_mono
        self._offset = 0
        self._pts = 0
        self._started_at: float | None = None
        self.done = asyncio.Event()

    async def recv(self) -> AudioFrame:
        """Return the next PCM16 audio frame."""
        if self.readyState != "live":
            self.done.set()
            raise MediaStreamError

        if self._offset >= len(self._pcm):
            self.done.set()
            self.stop()
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        if self._started_at is None:
            self._started_at = loop.time()
        else:
            target_time = self._started_at + (
                self._pts / CODEX_REALTIME_AUDIO_RATE
            )
            delay = target_time - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

        frame_size = STT_FRAME_SAMPLES * CODEX_REALTIME_SAMPLE_WIDTH_BYTES
        chunk = self._pcm[self._offset : self._offset + frame_size]
        self._offset += len(chunk)
        if len(chunk) < frame_size:
            chunk += b"\0" * (frame_size - len(chunk))

        frame = AudioFrame(format="s16", layout="mono", samples=STT_FRAME_SAMPLES)
        frame.planes[0].update(chunk)
        frame.sample_rate = CODEX_REALTIME_AUDIO_RATE
        frame.time_base = Fraction(1, CODEX_REALTIME_AUDIO_RATE)
        frame.pts = self._pts
        self._pts += STT_FRAME_SAMPLES
        return frame


async def _read_audio_stream(stream: AsyncIterable[bytes]) -> bytes:
    """Read an HA STT audio stream into bytes."""
    audio = bytearray()
    async for chunk in stream:
        audio.extend(chunk)
    return bytes(audio)


def _extract_pcm16_mono(audio: bytes, metadata: stt.SpeechMetadata) -> bytes:
    """Return raw PCM16 mono audio, stripping a WAV container when present."""
    if not audio:
        return b""
    if audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            if wav_file.getnchannels() != metadata.channel.value:
                raise HomeAssistantError("Codex STT WAV channel count mismatch")
            if wav_file.getsampwidth() != metadata.bit_rate.value // 8:
                raise HomeAssistantError("Codex STT WAV sample width mismatch")
            if wav_file.getframerate() != metadata.sample_rate.value:
                raise HomeAssistantError("Codex STT WAV sample rate mismatch")
            return wav_file.readframes(wav_file.getnframes())
    return audio


def _resample_pcm16_mono(
    pcm16_mono: bytes,
    *,
    source_rate: int,
    target_rate: int,
) -> bytes:
    """Resample PCM16 mono audio to the realtime input rate."""
    if source_rate == target_rate:
        return pcm16_mono

    source_frame_samples = max(1, int(source_rate * STT_FRAME_DURATION_SECONDS))
    source_frame_size = source_frame_samples * CODEX_REALTIME_SAMPLE_WIDTH_BYTES
    resampler = AudioResampler(format="s16", layout="mono", rate=target_rate)
    chunks: list[bytes] = []

    for offset in range(0, len(pcm16_mono), source_frame_size):
        chunk = pcm16_mono[offset : offset + source_frame_size]
        if not chunk:
            continue
        if len(chunk) % CODEX_REALTIME_SAMPLE_WIDTH_BYTES:
            chunk = chunk[:-1]
        if not chunk:
            continue
        samples = len(chunk) // CODEX_REALTIME_SAMPLE_WIDTH_BYTES
        frame = AudioFrame(format="s16", layout="mono", samples=samples)
        frame.planes[0].update(chunk)
        frame.sample_rate = source_rate
        frame.time_base = Fraction(1, source_rate)
        for resampled in resampler.resample(frame):
            chunks.append(_audio_frame_to_pcm16(resampled))

    with contextlib.suppress(Exception):
        for resampled in resampler.resample(None):
            chunks.append(_audio_frame_to_pcm16(resampled))
    return b"".join(chunks)


def _audio_frame_to_pcm16(frame: AudioFrame) -> bytes:
    """Return packed PCM16 bytes from a mono PyAV frame."""
    expected_size = (
        frame.samples
        * len(frame.layout.channels)
        * CODEX_REALTIME_SAMPLE_WIDTH_BYTES
    )
    return bytes(frame.planes[0])[:expected_size]


def _silence_pcm16(seconds: float) -> bytes:
    """Return PCM16 mono silence at the realtime input rate."""
    samples = int(CODEX_REALTIME_AUDIO_RATE * seconds)
    return b"\0" * (
        samples * CODEX_REALTIME_AUDIO_CHANNELS * CODEX_REALTIME_SAMPLE_WIDTH_BYTES
    )


def _build_transcription_session() -> dict[str, Any]:
    """Build a realtime transcription session payload."""
    return {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": CODEX_REALTIME_AUDIO_RATE,
                },
                "transcription": {
                    "model": RECOMMENDED_STT_MODEL,
                },
            },
        },
    }


async def _collect_transcript(
    websocket: aiohttp.ClientWebSocketResponse,
    track: BufferedPCM16AudioTrack,
) -> str | None:
    """Collect realtime transcription events until the final transcript arrives."""
    transcript_parts: list[str] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + STT_TOTAL_TIMEOUT
    audio_done_at: float | None = None
    receive = asyncio.create_task(websocket.receive())
    audio_done: asyncio.Task[bool] | None = asyncio.create_task(track.done.wait())

    try:
        while True:
            now = loop.time()
            if audio_done_at is not None:
                deadline = min(
                    deadline,
                    audio_done_at + STT_COMPLETION_TIMEOUT_AFTER_AUDIO,
                )
            remaining = deadline - now
            if remaining <= 0:
                LOGGER.warning("Timed out waiting for Codex STT transcript")
                return _joined_transcript(transcript_parts)

            wait_tasks = {receive}
            if audio_done is not None:
                wait_tasks.add(audio_done)
            done, _ = await asyncio.wait(
                wait_tasks,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                LOGGER.warning("Timed out waiting for Codex STT transcript")
                return _joined_transcript(transcript_parts)

            if audio_done is not None and audio_done in done:
                audio_done_at = loop.time()
                audio_done = None

            if receive in done:
                message = receive.result()
                receive = asyncio.create_task(websocket.receive())
                if message.type == aiohttp.WSMsgType.TEXT:
                    event = _decode_event(message.data)
                elif message.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                ):
                    LOGGER.warning("Codex STT websocket closed before completion")
                    return _joined_transcript(transcript_parts)
                elif message.type == aiohttp.WSMsgType.ERROR:
                    LOGGER.warning("Codex STT websocket failed")
                    return _joined_transcript(transcript_parts)
                else:
                    continue

                event_type = event.get("type")
                LOGGER.debug("Codex STT realtime event: %s", event_type)
                if event_type == "conversation.item.input_audio_transcription.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        transcript_parts.append(delta)
                    continue
                if (
                    event_type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    transcript = event.get("transcript")
                    if isinstance(transcript, str) and transcript.strip():
                        return transcript.strip()
                    return _joined_transcript(transcript_parts)
                if event_type == "error":
                    raise HomeAssistantError(
                        f"Codex STT error: {_event_error_message(event)}"
                    )
    finally:
        receive.cancel()
        if audio_done is not None:
            audio_done.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receive
        if audio_done is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await audio_done


def _decode_event(data: str) -> dict[str, Any]:
    """Decode a realtime websocket event."""
    try:
        event = json.loads(data)
    except json.JSONDecodeError as err:
        raise HomeAssistantError("Codex STT emitted invalid JSON") from err
    if not isinstance(event, dict):
        raise HomeAssistantError("Codex STT emitted a non-object event")
    return event


def _joined_transcript(parts: list[str]) -> str | None:
    """Return accumulated transcript deltas when available."""
    transcript = "".join(parts).strip()
    return transcript or None


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
