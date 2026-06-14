# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This repository is a Python-based pipeline that automates the creation of a daily AI news digest. It pulls information from various sources, uses an LLM to select and summarize relevant articles, and optionally generates audio versions of those summaries using Spark-TTS.

### Architecture
- **Configuration**: Managed via `config.yaml`, defining LLM parameters, fetch settings, and the list of news sources.
- **Digest Generation (`ai_digest.py`)**:
    - Fetches content from sources defined in `config.yaml`.
    - For `entry_point` types: Scrapes links, uses LLM to pick top $N$ AI-relevant links.
    - For `article` types: Directly summarizes the content.
    - Outputs a nested directory structure: `./digests/<date>/<source_name>/<article_id>.md`.
    - Creates an `index.md` in each date folder to link all articles.
- **Audio Synthesis (`tts_digest.py`)**:
    - Scans the `./digests/` directory for markdown files.
    - Strips markdown formatting (links, headers, bullets, etc.) to prepare plain text for TTS.
    - Uses the Spark-TTS-0.5B model to generate `.wav` files for each article.

## Development Commands

### Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
# Edit .env to include OPENAI_BASE_URL, OPENAI_API_KEY, and HF_TOKEN
```

### Running the Pipeline
```bash
# Run the news digest generation
python ai_digest.py

# Run the audio synthesis (today's digest)
python tts_digest.py

# Run audio synthesis for a specific date
python tts_digest.py 2026-06-11

# Run audio synthesis for a specific directory
python tts_digest.py --root ./digests/2026-06-11

# Force re-generation of audio files
python tts_digest.py --force
```

## Key Files
- `ai_digest.py`: Main logic for fetching, LLM summarization, and markdown generation.
- `tts_digest.py`: Logic for markdown-to-speech conversion using Spark-TTS.
- `config.yaml`: Source list and LLM configuration.
- `requirements.txt`: Core dependencies.
- `requirements-tts.txt`: Additional dependencies for the TTS component.
