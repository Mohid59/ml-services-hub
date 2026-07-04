"""Audio helpers: ASR (Whisper) + text-to-speech (gTTS).

Heavy imports (transformers/torch) are done lazily inside functions so the
app boots without loading them. Full wiring lands in milestone 5.
"""
import base64
import io

_asr_pipe = None  # module-level singleton cache


def get_asr():
    """Lazily build and cache the Whisper ASR pipeline."""
    global _asr_pipe
    if _asr_pipe is None:
        from transformers import pipeline

        import config
        _asr_pipe = pipeline(
            "automatic-speech-recognition", model=config.WHISPER_MODEL
        )
    return _asr_pipe


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file to text using Whisper."""
    result = get_asr()(audio_path)
    return (result.get("text") or "").strip()


def tts_base64(text: str, lang: str = "en") -> str:
    """Synthesize speech with gTTS, return base64 MP3 (no data prefix).

    Embed as: <audio controls src="data:audio/mp3;base64,{{ value }}">
    """
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
