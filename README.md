# AI Digest

A small Python pipeline that pulls AI/LLM news from a list of sources,
summarizes each item via an OpenAI-compatible LLM (e.g. a BedRocks proxy),
and writes one markdown file per article. A second script optionally
narrates each article to audio using
[Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B).

## Setup

```bash
cd ai-digest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# then edit .env and fill in OPENAI_BASE_URL, OPENAI_API_KEY AND HF_TOKEN
```

## Run

```bash
python ai_digest.py
```

Output layout (relative to the `output.dir` in `config.yaml`, default
`./digests/`):

```
digests/
└── 2026-06-11/
    ├── index.md                       # links to every article
    ├── hacker-news-front-page/
    │   ├── 01-some-article.md
    │   ├── 02-another-article.md
    │   └── ...
    ├── anthropic-news/
    │   └── 01-…md
    └── …
```

## Configure

Edit `config.yaml`:

- Each source has `enabled: true|false`.
- `type: entry_point` means the page is a list of links — the script will
  fetch the page, ask the LLM to pick the top N AI-relevant links, then
  fetch and summarize each.
- `type: article` means the URL is a single article and gets summarized
  directly.
- `max_articles_per_entry_point` caps how many articles are followed per
  entry-point source.
- `output.dir` is the root; each run creates a subfolder named after the
  date.

## Audio digest (Spark-TTS)

`tts_digest.py` walks the date folder produced by `ai_digest.py` and
writes a `.wav` next to every article markdown.

### One-time setup

```bash
# 1. Clone Spark-TTS somewhere stable
git clone https://github.com/SparkAudio/Spark-TTS.git ./Spark-TTS

# 2. Download the 0.5B model weights
cd ./Spark-TTS
mkdir -p pretrained_models
huggingface-cli download SparkAudio/Spark-TTS-0.5B \
    --local-dir pretrained_models/Spark-TTS-0.5B

# 3. Install Spark-TTS deps (large — torch, transformers, etc.) and ours
pip install -r requirements.txt
cd -   # back to ai-digest
pip install -r requirements-tts.txt

# 4. Tell tts_digest.py where the repo and model live
export SPARK_TTS_PATH=./Spark-TTS
export SPARK_TTS_MODEL_DIR=$SPARK_TTS_PATH/pretrained_models/Spark-TTS-0.5B
```

### Run

```bash
# Today's digest
python tts_digest.py

# A specific date
python tts_digest.py 2026-06-10

# Or point at any folder of digest markdown
python tts_digest.py --root ./digests/2026-06-10

# Re-generate even if .wav already exists
python tts_digest.py --force
```

Audio lands as `<article>.wav` next to each `<article>.md`. `index.md` is
skipped.

Notes:

- `pip install -r requirements-tts.txt` only installs the small extras
  this script needs (`torch`, `soundfile`, `huggingface_hub`); the heavy
  Spark-TTS deps come from `~/Spark-TTS/requirements.txt`.
- Apple Silicon: the script picks MPS automatically. CUDA is used when
  available; otherwise CPU (slow but functional).
