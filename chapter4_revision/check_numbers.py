"""Did the rewrite change any number?

Run this on whatever comes back from the rewrite. It compares every numeric
value in the new text against the original, by value and by count, and reports
anything added, dropped, or altered.

    python check_numbers.py rewritten.tex
    python check_numbers.py rewritten.tex --against 02_CONCLUZII.tex

A rewrite that passes this has changed only the prose.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent

# Figures written as words that must survive as words.
SPELLED = ["eleven million", "twenty-two million", "fifty-four percent",
           "fifty-seven percent", "six hundred", "nineteen thousand"]


def strip_comments(text: str) -> str:
    return "\n".join(l for l in text.splitlines()
                     if not l.lstrip().startswith("%"))


def numbers(text: str) -> Counter:
    """Every numeric literal, however it is written."""
    body = strip_comments(text)
    found: list[str] = []
    # inside \( ... \)
    found += re.findall(r"\\\((-?\d+(?:[.,]\d+)?)\\\)", body)
    # bare, outside maths -- catches a value someone reformatted out of \(...\)
    found += re.findall(r"(?<![\d.\\])(\d+\.\d+)(?![\d])", body)
    return Counter(found)


def spelled(text: str) -> Counter:
    low = strip_comments(text).lower()
    return Counter({s: low.count(s) for s in SPELLED if low.count(s)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rewritten", help="the file that came back")
    ap.add_argument("--against", default=str(HERE / "02_CONCLUZII.tex"))
    args = ap.parse_args()

    old = Path(args.against).read_text(encoding="utf-8")
    new = Path(args.rewritten).read_text(encoding="utf-8")

    a, b = numbers(old), numbers(new)
    sa, sb = spelled(old), spelled(new)

    print("=" * 70)
    print("NUMERIC VALUES")
    print("=" * 70)
    problems = 0
    for v in sorted(set(a) | set(b), key=lambda x: -a.get(x, 0)):
        if a.get(v, 0) == b.get(v, 0):
            continue
        problems += 1
        if v not in b:
            print(f"  DROPPED   {v}  (appeared {a[v]}x)")
        elif v not in a:
            print(f"  ADDED     {v}  (appears {b[v]}x) -- not in the original")
        else:
            print(f"  COUNT     {v}  was {a[v]}x, now {b[v]}x")
    if not problems:
        print(f"  every one of the {len(a)} values appears exactly as before")

    print()
    print("=" * 70)
    print("FIGURES WRITTEN AS WORDS")
    print("=" * 70)
    wp = 0
    for s in SPELLED:
        if sa.get(s, 0) != sb.get(s, 0):
            wp += 1
            print(f"  {s!r}: was {sa.get(s,0)}x, now {sb.get(s,0)}x")
    if not wp:
        print("  unchanged")

    print()
    print("=" * 70)
    total = problems + wp
    if total:
        print(f"VERDICT: {total} numeric differences -- do not use as-is")
    else:
        print("VERDICT: no number was touched; only the prose changed")
    print("=" * 70)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
