#!/usr/bin/env python3
"""Daily AI news digest.

Reads sources from config.yaml, fetches each enabled source, optionally picks
top links via the LLM (for entry_point sources), summarizes each article, and
writes one markdown file per article under <output_dir>/<date>/<source>/.
An index.md per date links to every article.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

CONFIG_PATH = Path(__file__).parent / "config.yaml"
ARTICLE_TEXT_CHAR_LIMIT = 8000
LINK_LIST_CHAR_LIMIT = 12000


@dataclass
class FetchedArticle:
    url: str
    title: str
    summary: str
    key_points: list[str]


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def http_get(url: str, *, user_agent: str, timeout: int) -> str:
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]
    ):
        tag.decompose()
    container = soup.find("article") or soup.find("main") or soup.body or soup
    text = container.get_text(separator="\n", strip=True)
    return text[:ARTICLE_TEXT_CHAR_LIMIT]


def extract_links(html: str, base_url: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    links: list[dict[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        title = a.get_text(separator=" ", strip=True)
        if not title or len(title) < 4:
            continue
        links.append({"title": title[:200], "url": absolute})
    return links


def llm_pick_links(
    client: OpenAI,
    *,
    model: str,
    source_name: str,
    links: list[dict[str, str]],
    max_picks: int,
    timeout: int,
) -> list[dict[str, str]]:
    listing = "\n".join(f"- {l['title']} :: {l['url']}" for l in links)[
        :LINK_LIST_CHAR_LIMIT
    ]
    system = (
        "You select the top news/article links most relevant to AI, LLMs, or ML "
        "from a scraped link list. Skip navigation, login, tag pages, and unrelated content. "
        "Avoid paywalled sites (e.g., WSJ, NYT, etc.) and prioritize freely accessible news sources. "
        "Respond with strict JSON only."
    )
    user = (
        f"Source: {source_name}\n"
        f"Pick up to {max_picks} items most relevant to AI/LLM/ML news.\n"
        f'Return JSON of the form: {{"picks": [{{"title": "...", "url": "..."}}]}}\n\n'
        f"Links:\n{listing}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        timeout=timeout,
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    picks = data.get("picks", [])
    return [p for p in picks if isinstance(p, dict) and p.get("url")][:max_picks]


def llm_summarize_article(
    client: OpenAI,
    *,
    model: str,
    url: str,
    text: str,
    max_tokens: int,
    timeout: int,
) -> FetchedArticle:
    system = (
        "You summarize articles for an AI-engineering audience. "
        "Write 2-3 sentence summaries. Respond with strict JSON only."
    )
    user = (
        f"URL: {url}\n\n"
        "Summarize this article in 2-3 sentences, then list 3-5 concise key bullets.\n"
        'Return JSON: {"title": "...", "summary": "...", "key_points": ["...", "..."]}\n\n'
        f"Article text:\n{text}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        timeout=timeout,
    )

    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    return FetchedArticle(
        url=url,
        title=data.get("title") or url,
        summary=data.get("summary") or "(no summary returned)",
        key_points=[str(p) for p in data.get("key_points", []) if p],
    )


def process_source(
    client: OpenAI,
    *,
    source: dict[str, Any],
    cfg: dict[str, Any],
) -> list[FetchedArticle]:
    name = source["name"]
    url = source["url"]
    stype = source.get("type", "entry_point")
    user_agent = cfg["fetch"]["user_agent"]
    http_timeout = cfg["fetch"]["http_timeout_s"]
    llm_timeout = cfg["llm"]["request_timeout_s"]
    model = cfg["llm"]["model"]
    max_tokens = cfg["llm"]["max_tokens_per_summary"]

    print(f"[{name}] fetching {url}", file=sys.stderr)
    html = http_get(url, user_agent=user_agent, timeout=http_timeout)

    if stype == "article":
        text = extract_main_text(html)
        article = llm_summarize_article(
            client,
            model=model,
            url=url,
            text=text,
            max_tokens=max_tokens,
            timeout=llm_timeout,
        )
        return [article]

    # entry_point
    links = extract_links(html, base_url=url)
    if not links:
        print(f"[{name}] no usable links found", file=sys.stderr)
        return []
    print(
        f"[{name}] {len(links)} candidate links — asking LLM to pick top {cfg['fetch']['max_articles_per_entry_point']}",
        file=sys.stderr,
    )
    picks = llm_pick_links(
        client,
        model=model,
        source_name=name,
        links=links,
        max_picks=cfg["fetch"]["max_articles_per_entry_point"],
        timeout=llm_timeout,
    )
    print(f"[{name}] LLM picked {len(picks)} links", file=sys.stderr)

    articles: list[FetchedArticle] = []
    for pick in picks:
        purl = pick["url"]
        try:
            print(f"[{name}] fetching article {purl}", file=sys.stderr)
            ahtml = http_get(purl, user_agent=user_agent, timeout=http_timeout)
            atext = extract_main_text(ahtml)
            article = llm_summarize_article(
                client,
                model=model,
                url=purl,
                text=atext,
                max_tokens=max_tokens,
                timeout=llm_timeout,
            )
            articles.append(article)
        except Exception as e:
            print(f"[{name}] failed on {purl}: {e}", file=sys.stderr)
    return articles


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_len: int = 60) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return (s or "item")[:max_len].rstrip("-")


def render_article_markdown(source_name: str, article: FetchedArticle) -> str:
    out: list[str] = [
        f"# {article.title}",
        "",
        f"_Source: {source_name}_",
        "",
        article.summary,
        "",
    ]
    for kp in article.key_points:
        out.append(f"- {kp}")
    if article.key_points:
        out.append("")
    out.append(f"[Read more]({article.url})")
    out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_index(
    today: str,
    written: list[tuple[str, list[tuple[FetchedArticle, Path]]]],
    date_root: Path,
) -> str:
    out: list[str] = [f"# AI Digest — {today}", ""]
    for source_name, items in written:
        out.append(f"## {source_name}")
        out.append("")
        if not items:
            out.append("_No items collected._")
            out.append("")
            continue
        for article, path in items:
            rel = path.relative_to(date_root).as_posix()
            out.append(f"- [{article.title}]({rel}) — [original]({article.url})")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    load_dotenv()
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        print(
            "ERROR: set OPENAI_BASE_URL and OPENAI_API_KEY (in env or .env)",
            file=sys.stderr,
        )
        return 2

    cfg = load_config()
    client = OpenAI(base_url=base_url, api_key=api_key)

    enabled_sources = [s for s in cfg.get("sources", []) if s.get("enabled")]
    if not enabled_sources:
        print("No enabled sources in config.yaml", file=sys.stderr)
        return 1

    today = date.today().isoformat()
    out_root = Path(cfg["output"]["dir"]).expanduser().resolve()
    date_root = out_root / today / "md"
    date_root.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, list[tuple[FetchedArticle, Path]]]] = []
    for source in enabled_sources:
        name = source["name"]
        try:
            articles = process_source(client, source=source, cfg=cfg)
        except Exception as e:
            print(f"[{name}] source failed: {e}", file=sys.stderr)
            articles = []

        source_dir = date_root / slugify(name)
        source_dir.mkdir(parents=True, exist_ok=True)
        items: list[tuple[FetchedArticle, Path]] = []
        for idx, article in enumerate(articles, start=1):
            filename = f"{idx:02d}-{slugify(article.title)}.md"
            article_path = source_dir / filename
            article_path.write_text(render_article_markdown(name, article))
            print(f"Wrote {article_path}")
            items.append((article, article_path))
        written.append((name, items))

    index_path = date_root / "index.md"
    index_path.write_text(render_index(today, written, date_root))
    print(f"Wrote {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
