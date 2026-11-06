"""Programmatically verifies catalog citations against the actual passages (existence + keyword support)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config, rag

def check_ref(idx, r):
    ch = idx.lookup(r["source_id"], r["ref"])
    if not ch:
        return False, "pasaj bulunamadı"
    txt = " ".join(c["text"].lower() for c in ch)
    hit = [k for k in r["keywords"] if k.lower() in txt]
    return (bool(hit), "ok" if hit else "anahtar kelime yok")

def main():
    idx = rag.get_index()
    cat = json.loads((config.CATALOG_DIR / "controls.json").read_text(encoding="utf-8"))
    bad = 0
    for c in cat["controls"]:
        for r in c["sources"]:
            ok, why = check_ref(idx, r)
            if not ok:
                bad += 1
                print(f"DESTEKLENMİYOR {c['id']} {r['source_id']} {r['ref']}: {why}")
    print(f"{sum(len(c['sources']) for c in cat['controls'])} eşleştirme, {bad} sorunlu")
    return bad

if __name__ == "__main__":
    sys.exit(1 if main() else 0)
