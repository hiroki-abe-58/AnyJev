"""sokudan's bench_ja (Japanese, 300 items) and bench_en (English, 290 items): support messages with three
typed questions each, the same three schemas in both languages. One task per (set, question):
`sokudan_ja.department`, `sokudan_ja.urgency`, `sokudan_ja.churn`, and the same three under `sokudan_en.`.

Mapping from sokudan's types (question text, options and their order exactly as
`sokudan/eval/bench_ja.py` and `bench_en.py` define them at the pinned commit):
- choice (department, 4 options) -> `Question.choice`, options "key: description" as in typed_decisions;
  label = index of the gold key.
- score (urgency, 3 levels) -> `Question.score(levels=...)`, least urgent first; label = the gold level
  index, which the source stores as an int.
- bool (churn) -> `Question.noul` with the source's question as its text; label 0 (Yes) is true.

Gold is the condition each message was generated from (label-conditioned generation by one local LLM,
qwen3:30b-a3b-instruct-2507), not a human judgment, so accuracy is agreement with that condition. bench_en rows
also carry a blind LLM verifier's verdicts; the `verified` count goes into `meta`, it does not filter.
The files are fetched at run time from the pinned commit and checked against their SHA-256, never vendored.
Both sets are CC BY 4.0. They are meant for evaluation: the source asks that neither be used as training data
(a request, not a licence term).

    python -m bench.run --model Qwen/Qwen3-8B --n 200 --calib 100 \
        --tasks sokudan_ja.department,sokudan_ja.urgency,sokudan_ja.churn
    python -m bench.run --model Qwen/Qwen3-8B --n 190 --calib 100 \
        --tasks sokudan_en.department,sokudan_en.urgency,sokudan_en.churn
"""
from __future__ import annotations

import functools
import hashlib
import json
from typing import Any, Dict, List

from anyjev.question import Question
from bench.tasks.base import Task, register

COMMIT = "788306c844866bef9f83c4865e5bb1cb3f73e334"
URL = "https://raw.githubusercontent.com/hiroki-abe-58/sokudan/{commit}/data/{name}.jsonl"
SHA256 = {"bench_ja": "08ed6d1d2c86a30eff4cd13ced6627210631c878526e478b2e7c6ad3d91f74f5",
          "bench_en": "dcf62c360ceef05d199efd5a96cf5838caff583d42e3ddf8797708659e2b51c0"}
LICENSE = {"bench_ja": "CC-BY-4.0", "bench_en": "CC-BY-4.0"}

# set -> (file, {question: (sokudan type, text, criteria)}), from sokudan/eval/bench_{ja,en}.py
SETS: Dict[str, tuple] = {
    "ja": ("bench_ja", {
        "department": ("choice", "この問い合わせはどの部署が担当すべきか", {
            "請求": "支払い・返金・請求書・料金の二重引き落としなど金銭処理に関するもの",
            "技術": "不具合・障害・エラー・ログイン不能など製品が動かないことに関するもの",
            "営業": "料金プラン・新規契約・見積もり・増席など購入判断に関するもの",
            "その他": "上のどれにも当てはまらない一般的な連絡"}),
        "urgency": ("score", "この依頼の緊急度は", ["急がない", "早めに", "業務が止まっている"]),
        "churn": ("bool", "送信者は解約・契約終了を示唆しているか", None)}),
    "en": ("bench_en", {
        "department": ("choice", "Which team should handle this message?", {
            "Billing": "payments, refunds, invoices, double charges -- anything about money",
            "Technical": "bugs, outages, errors, being unable to log in -- the product not working",
            "Sales": "pricing plans, new contracts, quotes, adding seats -- a purchase decision",
            "Other": "general correspondence that fits none of the above"}),
        "urgency": ("score", "How urgent is this request?", ["Not urgent", "Soon", "Work is blocked"]),
        "churn": ("bool", "Is the sender hinting that they may stop using the service?", None)}),
}


@functools.lru_cache(maxsize=None)
def fetch(name: str) -> List[Dict[str, Any]]:
    """The rows of `data/<name>.jsonl` at the pinned commit; refuses a file whose hash moved."""
    import urllib.request

    with urllib.request.urlopen(URL.format(commit=COMMIT, name=name), timeout=60) as r:
        raw = r.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != SHA256[name]:
        raise ValueError(f"{name}.jsonl: sha256 {got}, pinned {SHA256[name]}")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def build_question(qname: str, kind: str, text: str, criteria: Any) -> Question:
    if kind == "choice":
        return Question.choice(text, [f"{k}: {v}" for k, v in criteria.items()], name=qname)
    if kind == "score":
        return Question.score(text, levels=criteria, name=qname)
    return Question.noul(text, name=qname)


def gold_index(kind: str, criteria: Any, value: Any) -> int:
    if kind == "choice":
        return list(criteria).index(value)
    if kind == "score":
        return int(value)
    return 0 if value else 1          # noul: option 0 is Yes


def load(lang: str, qname: str) -> Task:
    name, questions = SETS[lang]
    kind, text, criteria = questions[qname]
    rows = fetch(name)
    items = [(row["state"], gold_index(kind, criteria, row[qname])) for row in rows]
    verified = sum(1 for row in rows if row.get("verified")) if "verified" in rows[0] else None
    return Task(f"sokudan_{lang}.{qname}", build_question(qname, kind, text, criteria), items,
                license=LICENSE[name], source=URL.format(commit=COMMIT, name=name),
                notes=f"{len(items)} items; gold is the generation condition, not a human label",
                meta={"commit": COMMIT, "sha256": SHA256[name], "sokudan_type": kind, "verified": verified})


for _lang, (_, _questions) in SETS.items():
    for _qname in _questions:
        register(f"sokudan_{_lang}.{_qname}")(functools.partial(load, _lang, _qname))
