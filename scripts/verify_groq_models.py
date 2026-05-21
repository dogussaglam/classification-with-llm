"""Pre-flight check: confirm all 4 Phase 3 model slugs are available on Groq."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import groq  # noqa: E402

from llm_textcls.io import load_env  # noqa: E402

REQUIRED_SLUGS = (
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Groq model slugs are live.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Also issue a 1-token smoke call to GPT-OSS with reasoning_effort=low.",
    )
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Put it in .env or export it.")
        return 2

    client = groq.Groq(api_key=api_key)
    try:
        models = client.models.list()
    except groq.APIError as e:
        print(f"ERROR: Groq models.list() failed: {e}")
        return 2

    available = {m.id for m in models.data}
    missing = [s for s in REQUIRED_SLUGS if s not in available]

    print(f"Available Groq models ({len(available)}): {sorted(available)}")
    print()
    if missing:
        print(f"ERROR: missing required slug(s): {missing}")
        return 2

    print(f"OK: all {len(REQUIRED_SLUGS)} required slugs are available.")

    if args.smoke:
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                temperature=0.0,
                max_tokens=8,
                reasoning_effort="low",
                reasoning_format="hidden",
            )
            print(f"Smoke call OK: {resp.choices[0].message.content!r}")
        except TypeError as e:
            print(
                "WARNING: SDK didn't accept reasoning_effort/reasoning_format as direct "
                f"kwargs ({e}). GroqLabeler will fall back to extra_body."
            )
        except groq.APIError as e:
            print(f"WARNING: smoke call failed (non-fatal): {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
