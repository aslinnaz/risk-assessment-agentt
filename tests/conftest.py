import json, re, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
import rag


@pytest.fixture(scope="session")
def index():
    return rag.get_index()


class FakeClient:
    """Scripted fake client used only to test the application's own flow (produces no real result)."""
    def __init__(self, scripts):
        self.scripts, self.calls, self.last_user = list(scripts), 0, ''
        outer = self
        class _C:
            def create(self, **kw):
                outer.calls += 1
                user = kw["messages"][1]["content"]
                topics = re.findall(r"ADAY konu (\w+)", user)
                outer.last_user = user
                upd = outer.scripts.pop(0) if outer.scripts else {}
                m = re.search(r"SON CEVAP \(\w+\): (.*?)\n\n", user, re.S)
                ans = m.group(1) if m else ""
                ev = upd.get("evidence", {})
                res = {"slot_updates": [{"slot": k, "value": v, "evidence": ev.get(k, ans)} for k, v in upd.get("slots", {}).items()],
                       "control_updates": upd.get("controls", []), "basis": upd.get("basis", []),
                       "contradiction": upd.get("contradiction", ""),
                       "next": {"topic_id": upd.get("topic") or (topics[0] if topics else "purpose"), "is_clarification": bool(upd.get("contradiction")),
                                "question": upd.get("question", "Test sorusu burada?"), "options": ["Evet", "Hayır"], "missing": "Eksik test bilgisi.", "clarifies": "Test netleştirmesi.", "source_note": "", "passage_ids": [], "adapted": False}}
                class M: content = json.dumps(res)
                class Ch: message = M
                class R: choices = [Ch]
                return R
        class _Chat: completions = _C()
        self.chat = _Chat()
        class _E:
            def create(self, **kw): raise RuntimeError("no embed")
        self.embeddings = _E()
