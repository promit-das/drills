"""
CLI tool to generate minutes of meeting (MoM) from audio or transcript files.

Transcription can be done via:
- OpenAI audio transcription (default)
- Hugging Face ASR pipeline
- Existing transcript files (.txt/.md/.docx/.pdf)

MoM generation can be done via:
- OpenAI chat models
- Ollama (local, e.g., llama3.2)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional
from openai import OpenAI
try:
    import ollama
except ImportError:
    ollama = None
from dotenv import load_dotenv


DEFAULT_SYSTEM_MESSAGE = (
    "You produce minutes of meetings from transcripts, with summary, key discussion "
    "points, takeaways and action items with owners, in markdown format without code blocks."
)

DEFAULT_USER_PROMPT = (
    "Below is a transcript of a meeting. Please write minutes in markdown without code blocks, "
    "including:\n"
    "- a summary with attendees, location and date\n"
    "- discussion points\n"
    "- takeaways\n"
    "- action items with owners\n"
)


def _get_openai_client(api_key: Optional[str]) -> OpenAI:
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Set it in your environment or pass "
            "--openai-api-key to use OpenAI services."
        )
    return OpenAI(api_key=api_key)


def transcribe_with_openai(
    audio_path: Path, model: str, api_key: Optional[str]
) -> str:
    client = _get_openai_client(api_key)
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model, file=audio_file, response_format="text"
        )
    if isinstance(response, str):
        return response
    return getattr(response, "text", str(response))


def transcribe_with_hf(audio_path: Path, model: str) -> str:
    from transformers import pipeline
    import torch
    import shutil

    if shutil.which("ffmpeg") is None:
        raise ValueError(
            "ffmpeg is required for Hugging Face transcription. "
            "Install it (e.g., `brew install ffmpeg`) or use OpenAI transcription."
        )

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if device == 0 else torch.float32

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        device=device,
        dtype=dtype,
        return_timestamps=True,
    )

    result = pipe(str(audio_path))
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    return str(result)


def build_user_message(prompt: str, transcript: str) -> str:
    if "{transcript}" in prompt:
        return prompt.format(transcript=transcript)
    return f"{prompt.strip()}\n\nTranscription:\n{transcript}"


def _downloads_dir() -> Path:
    return Path.home() / "Downloads"


def save_transcription_to_downloads(audio_path: Path, transcript: str) -> Path:
    downloads_dir = _downloads_dir()
    downloads_dir.mkdir(parents=True, exist_ok=True)
    output_path = downloads_dir / f"{audio_path.stem}_transcription.txt"
    output_path.write_text(transcript.strip() + "\n", encoding="utf-8")
    return output_path


def save_mom_to_downloads(audio_path: Path, mom: str) -> Path:
    downloads_dir = _downloads_dir()
    downloads_dir.mkdir(parents=True, exist_ok=True)
    output_path = downloads_dir / f"{audio_path.stem}_mom.txt"
    output_path.write_text(mom.strip() + "\n", encoding="utf-8")
    return output_path


def generate_mom_with_openai(
    transcript: str, prompt: str, model: str, api_key: Optional[str]
) -> str:
    client = _get_openai_client(api_key)

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": build_user_message(prompt, transcript)},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    content = response.choices[0].message.content or ""
    return content.strip()


def generate_mom_with_ollama(transcript: str, prompt: str, model: str) -> str:
    if ollama is None:
        raise ImportError(
            "ollama Python package is not installed. Install it or use --mom-provider openai."
        )

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": build_user_message(prompt, transcript)},
    ]

    response = ollama.chat(model=model, messages=messages)
    content = response.get("message", {}).get("content", "")
    return (content or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio and generate minutes of meeting (MoM)."
    )
    parser.add_argument(
        "--audio",
        help="Path to an audio file (.mp3/.wav/...) or transcript (.txt/.md/.docx/.pdf).",
    )
    parser.add_argument(
        "--transcriber",
        choices=["openai", "hf"],
        default=None,
        help="Transcription provider (if omitted, you will be prompted).",
    )
    parser.add_argument(
        "--transcribe-model",
        default="gpt-4o-mini-transcribe",
        help="OpenAI transcription model (used when --transcriber openai).",
    )
    parser.add_argument(
        "--hf-asr-model",
        default="openai/whisper-medium.en",
        help="Hugging Face ASR model (used when --transcriber hf).",
    )
    parser.add_argument(
        "--mom-model",
        default="gpt-4o-mini",
        help="OpenAI model to generate the MoM.",
    )
    parser.add_argument(
        "--mom-provider",
        choices=["openai", "ollama"],
        default=None,
        help="MoM generation provider (if omitted, you will be prompted).",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model to generate the MoM when --mom-provider ollama.",
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (overrides OPENAI_API_KEY from environment).",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        help=(
            "Custom prompt text. If it contains {transcript}, it will be replaced. "
            "Otherwise the transcript is appended."
        ),
    )
    prompt_group.add_argument(
        "--prompt-file",
        help="Path to a text file containing the prompt.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the MoM markdown.",
    )

    return parser.parse_args()


def prompt_for_audio_path() -> str:
    while True:
        user_input = input(
            "Enter path to the audio file (.mp3/.wav) or transcript (.txt/.md/.docx/.pdf): "
        ).strip()
        if user_input:
            return user_input
        print("Audio path cannot be empty. Please try again.")


def _is_audio_path(path: Path) -> bool:
    return path.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


def _is_text_path(path: Path) -> bool:
    return path.suffix.lower() in {".txt", ".md", ".docx", ".pdf"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_docx_file(path: Path) -> str:
    try:
        import docx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "python-docx is required to read .docx files. Install it or provide a .txt file."
        ) from exc
    document = docx.Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text]
    return "\n".join(paragraphs)


def _read_pdf_file(path: Path) -> str:
    try:
        import pypdf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pypdf is required to read .pdf files. Install it or provide a .txt file."
        ) from exc
    reader = pypdf.PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            pages_text.append(page_text)
    return "\n".join(pages_text).strip()


def read_transcript_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return _read_text_file(path)
    if ext == ".docx":
        return _read_docx_file(path)
    if ext == ".pdf":
        return _read_pdf_file(path)
    raise ValueError(
        "Unsupported transcript file type. Use .txt/.md/.docx/.pdf or provide audio."
    )


def _prompt_yes_no(question: str, default: bool = False) -> bool:
    prompt = " [Y/n]: " if default else " [y/N]: "
    while True:
        user_input = input(question + prompt).strip().lower()
        if not user_input:
            return default
        if user_input in {"y", "yes"}:
            return True
        if user_input in {"n", "no"}:
            return False
        print("Please enter y or n.")


def prompt_for_transcriber() -> str:
    while True:
        user_input = input(
            "Select transcription provider: [1] OpenAI  [2] Hugging Face: "
        ).strip().lower()
        if user_input in {"1", "openai", "o"}:
            return "openai"
        if user_input in {"2", "hf", "huggingface", "h"}:
            return "hf"
        print("Please enter 1 for OpenAI or 2 for Hugging Face.")


def prompt_for_mom_provider() -> str:
    while True:
        user_input = input("Select MoM provider: [1] OpenAI  [2] Ollama: ").strip().lower()
        if user_input in {"1", "openai", "o"}:
            return "openai"
        if user_input in {"2", "ollama", "ol"}:
            return "ollama"
        print("Please enter 1 for OpenAI or 2 for Ollama.")


def main() -> None:
    args = parse_args()

    if load_dotenv is not None:
        load_dotenv()

    audio_input = args.audio or prompt_for_audio_path()
    audio_path = Path(audio_input).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")

    prompt = DEFAULT_USER_PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt:
        prompt = args.prompt

    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")

    transcriber = args.transcriber or prompt_for_transcriber()

    transcript: str
    if _is_text_path(audio_path):
        if _prompt_yes_no(
            "You selected a text file. Use it as the transcript and skip transcription?",
            default=True,
        ):
            transcript = read_transcript_file(audio_path)
        else:
            raise ValueError(
                "Please provide an audio file (.mp3/.wav/.m4a/.flac/.ogg/.webm)."
            )
    else:
        if not _is_audio_path(audio_path):
            raise ValueError(
                "Unsupported file type. Please provide an audio file "
                "(.mp3/.wav/.m4a/.flac/.ogg/.webm) or a transcript "
                "(.txt/.md/.docx/.pdf)."
            )
        if transcriber == "openai":
            transcript = transcribe_with_openai(
                audio_path=audio_path,
                model=args.transcribe_model,
                api_key=openai_api_key,
            )
        else:
            transcript = transcribe_with_hf(
                audio_path=audio_path, model=args.hf_asr_model
            )

    save_transcription_to_downloads(audio_path, transcript)

    mom_provider = args.mom_provider or prompt_for_mom_provider()

    if mom_provider == "ollama":
        mom = generate_mom_with_ollama(
            transcript=transcript,
            prompt=prompt,
            model=args.ollama_model,
        )
    else:
        mom = generate_mom_with_openai(
            transcript=transcript,
            prompt=prompt,
            model=args.mom_model,
            api_key=openai_api_key,
        )

    save_mom_to_downloads(audio_path, mom)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.write_text(mom + "\n", encoding="utf-8")

    print(mom)


if __name__ == "__main__":
    main()
