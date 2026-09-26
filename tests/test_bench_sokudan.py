"""bench/tasks/sokudan.py on synthetic rows: the type mapping, the gold indices, one bench.run pass per
question type on the fake backend, and the hash check. No download: `fetch` is replaced."""
import io

import pytest

from anyjev import Decider
from anyjev.backends.fake import FakeBackend
from bench.run import run_task
from bench.tasks import TASKS, get_task, sokudan


def rows(lang, n=24):
    """Made-up rows in the source's format; the state spells out its own gold so the fake can read it."""
    _, qs = sokudan.SETS[lang]
    depts, levels = list(qs["department"][2]), qs["urgency"][2]
    out = []
    for i in range(n):
        d, u, c = depts[i % 4], i % 3, i % 5 == 0
        out.append({"item_id": f"t-{i}", "state": f"{i};={d};={levels[u]};churn={'yes' if c else 'no'};",
                    "department": d, "urgency": u, "churn": c, **({"verified": i % 2 == 0} if lang == "en" else {})})
    return out


@pytest.fixture
def fake_fetch(monkeypatch):
    monkeypatch.setattr(sokudan, "fetch", lambda name: rows("ja" if name == "bench_ja" else "en"))


def content(state, option):
    if option in ("Yes", "No"):
        return 3.0 if (option == "Yes") == ("churn=yes" in state) else 0.0
    return 3.0 if f"={option.split(': ')[0]};" in state else 0.0


def test_six_tasks_are_registered():
    assert {f"sokudan_{lang}.{q}" for lang in ("ja", "en") for q in ("department", "urgency", "churn")} <= set(TASKS)


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_types_map_to_choice_score_noul_with_the_source_order(fake_fetch, lang):
    _, qs = sokudan.SETS[lang]
    src = rows(lang)
    dep = get_task(f"sokudan_{lang}.department")
    assert dep.question.kind == "choice" and dep.question.k == 4
    assert [o.split(": ")[0] for o in dep.question.options] == list(qs["department"][2])
    assert [y for _, y in dep.items] == [list(qs["department"][2]).index(r["department"]) for r in src]
    urg = get_task(f"sokudan_{lang}.urgency")
    assert urg.question.kind == "score" and urg.question.ordered
    assert list(urg.question.options) == qs["urgency"][2]          # least urgent first
    assert urg.question.centers == (0.0, 1.0, 2.0)
    assert [y for _, y in urg.items] == [r["urgency"] for r in src]
    churn = get_task(f"sokudan_{lang}.churn")
    assert churn.question.kind == "noul" and churn.question.options == ("Yes", "No")
    assert churn.question.text == qs["churn"][1]
    assert [y for _, y in churn.items] == [0 if r["churn"] else 1 for r in src]
    assert dep.meta["commit"] == sokudan.COMMIT and dep.license == sokudan.LICENSE[sokudan.SETS[lang][0]]
    assert dep.meta["verified"] == (12 if lang == "en" else None)


@pytest.mark.parametrize("qname", ["department", "urgency", "churn"])
def test_bench_run_reports_every_level_and_the_flip_rate_on_the_fake(fake_fetch, qname):
    be = FakeBackend(content, position_bias=[4.0, 0.0, 0.0, 0.0], label_prior={"Yes": 2.0})
    out = run_task(Decider(be), f"sokudan_ja.{qname}", n_test=12, n_calib=12, seed=0, levels=["raw", "L0", "L1"])
    assert out["n_test"] == 12 and out["n_calib"] == 12
    assert {"raw", "L0", "L1"} <= set(out["levels"])
    assert all("flip" in m for m in out["levels"].values())
    # the planted bias fools raw; zero-label debiasing reads the planted content
    assert out["levels"]["raw"]["acc"] < 1.0
    assert out["levels"]["L0"]["acc"] == 1.0


def test_fetch_refuses_a_file_whose_hash_moved(monkeypatch):
    import urllib.request

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: Resp(b'{"state": "x"}\n'))
    sokudan.fetch.cache_clear()
    try:
        with pytest.raises(ValueError, match="sha256"):
            sokudan.fetch("bench_ja")
    finally:
        sokudan.fetch.cache_clear()
