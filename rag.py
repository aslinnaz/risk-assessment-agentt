"""Local RAG: PDF extraction, article/section-bounded chunking, BM25 + persistent embedding index.

- Cache by file hash: the same file is not re-read / re-embedded.
- Embeddings are produced only by explicit user action (build_embeddings).
- Sources with api_transfer_allowed=false (ISO) do not enter the index.
- Source texts are data; instructions inside them are not applied.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

import config

_HDR = re.compile(r"^(OJ L, .*EN|EN OJ L, .*|ELI: http\S+.*)$")


# --------------------------------------------------------------------------- manifest
def load_manifest() -> dict:
    return json.loads((config.CATALOG_DIR / "sources.json").read_text(encoding="utf-8"))


def source_meta(source_id: str) -> dict:
    for s in load_manifest()["sources"]:
        if s["source_id"] == source_id:
            return s
    raise KeyError(source_id)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- extraction
def extract_pages(path: Path) -> list[str]:
    import logging
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    from pypdf import PdfReader
    return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- chunking
def _mk(src, ref, title, page, text, kind, section=""):
    return {"source_id": src["source_id"], "ref": ref, "title": title, "section": section,
            "version": src["version"], "page": page, "text": text.strip(), "kind": kind,
            "url": src.get("url"), "authority": src["authority"]}


def _split_long(text: str, limit: int = 1800) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.;:])\s+", text):
        if len(cur) + len(sent) > limit and cur:
            parts.append(cur)
            cur = ""
        cur += (" " if cur else "") + sent
    if cur:
        parts.append(cur)
    return parts


def chunk_aia(src, pages):
    lines = []
    for i, t in enumerate(pages, 1):
        for ln in t.split("\n"):
            ln = ln.rstrip()
            if ln and not _HDR.match(ln.strip()):
                lines.append((i, ln))
    chunks = []
    # --- recitals / body split
    idx = next(i for i, (_, l) in enumerate(lines) if l.strip().startswith("HAVE ADOPTED THIS REGULATION"))
    rec, body = lines[:idx], lines[idx + 1:]
    cur, n = None, 1
    for pg, ln in rec:
        m = re.match(r"^\((\d{1,3})\) ", ln)
        if m and int(m.group(1)) == n:
            if cur:
                chunks.append(cur)
            cur = [n, pg, [ln]]
            n += 1
        elif cur:
            cur[2].append(ln)
    if cur:
        chunks.append(cur)
    out = []
    for n_, pg, ls in chunks:
        for j, part in enumerate(_split_long(_flat(" ".join(ls)), 2200)):
            out.append(_mk(src, f"Recital ({n_})" + (f" #{j+1}" if j else ""), "Recital", pg, part, "recital"))
    # --- articles and annexes
    blocks, cur, art_n = [], None, 1
    skip_next = False
    for pg, ln in body:
        s = ln.strip()
        if skip_next:
            skip_next = False
            if s.isupper():
                continue
        if re.match(r"^(CHAPTER|SECTION) [IVX\d]+$", s):
            skip_next = True
            continue
        ma = re.match(r"^Article (\d+)$", s)
        mx = re.match(r"^ANNEX ([IVX]+)$", s)
        if cur is None or cur["type"] == "art":
            if ma and int(ma.group(1)) == art_n:
                cur = {"type": "art", "id": art_n, "page": pg, "lines": []}
                blocks.append(cur)
                art_n += 1
                continue
        if mx and (cur is None or cur["id"] != mx.group(1)):
            cur = {"type": "annex", "id": mx.group(1), "page": pg, "lines": []}
            blocks.append(cur)
            continue
        if cur:
            cur["lines"].append((pg, ln))
    for b in blocks:
        label = f"Art. {b['id']}" if b["type"] == "art" else f"Annex {b['id']}"
        ls = b["lines"]
        title = ls[0][1].strip() if ls else ""
        ls = ls[1:] if b["type"] == "art" else ls
        if b["type"] == "annex":
            title = ls[0][1].strip() if ls else ""
            ls = ls[1:]
        if b["type"] == "art" and b["id"] == 3:  # definitions: (1) ‘AI system’ … (N) — one passage per numbered definition
            defs, curd, nd = [[None, ls[0][0] if ls else b["page"], []]], None, 1
            for pg, ln in ls:
                m = re.match(r"^\((\d{1,2})\) ", ln)
                if m and int(m.group(1)) == nd:
                    defs.append([str(nd), pg, [ln]])
                    nd += 1
                else:
                    defs[-1][2].append(ln)
            for lab, pg, pl in defs:
                txt = _flat(" ".join(pl))
                if not txt:
                    continue
                rf = "Art. 3 (giriş)" if lab is None else f"Art. 3({lab})"
                for j, part in enumerate(_split_long(txt, 2200)):
                    out.append(_mk(src, rf + (f" #{j+1}" if j else ""), f"Art. 3 — {title} (tanım {lab})" if lab else f"Art. 3 — {title}", pg, part, "article", title))
            continue
        # paragraph split
        paras, curp, nxt = [], [None, ls[0][0] if ls else b["page"], []], 1
        for pg, ln in ls:
            m = re.match(r"^(\d+)\. ", ln)
            if m and int(m.group(1)) == nxt:
                paras.append(curp)
                curp = [str(nxt), pg, [ln]]
                nxt += 1
            else:
                curp[2].append(ln)
        paras.append(curp)
        for lab, pg, pl in paras:
            if not pl:
                continue
            txt = _flat(" ".join(pl))
            ref = label if lab is None else f"{label}({lab})" if b["type"] == "art" else f"{label}, {lab}."
            if lab is None and len(paras) > 1:
                ref = label + " (giriş)"
            # sub-point split
            if len(txt) > 1800 and b["type"] == "art":
                items, curi, nl = [], [None, pg, []], 0
                for l2 in pl:
                    m2 = re.match(r"^\(([a-z])\) ", l2)
                    if m2 and ord(m2.group(1)) - 96 == nl + 1:
                        items.append(curi)
                        curi = [m2.group(1), pg, [l2]]
                        nl += 1
                    else:
                        curi[2].append(l2)
                items.append(curi)
                if len(items) > 2:
                    head = _flat(" ".join(items[0][2]))
                    if len(head) >= 300:  # a long introduction is not discarded
                        for j, part in enumerate(_split_long(head, 2200)):
                            out.append(_mk(src, ref + " (giriş)" + (f" #{j+1}" if j else ""), f"{label} — {title}", items[0][1], part, "article", title))
                    for il, ipg, ill in items:
                        if il is None:
                            continue
                        pg = ipg
                        t2 = _flat(" ".join(ill))
                        t2 = (head[:300] + " … " + t2) if len(head) < 300 and head else t2
                        for j, part in enumerate(_split_long(t2, 2200)):
                            out.append(_mk(src, f"{ref}({il})" + (f" #{j+1}" if j else ""), f"{label} — {title}", pg, part, "article" if b["type"] == "art" else "annex", title))
                    continue
            for j, part in enumerate(_split_long(txt, 2200)):
                out.append(_mk(src, ref + (f" #{j+1}" if j else ""), f"{label} — {title}", pg, part, "article" if b["type"] == "art" else "annex", title))
    return out


def chunk_decision(src, pages):
    out = []
    for i, t in enumerate(pages, 1):
        t = _flat("\n".join(l for l in t.split("\n") if not _HDR.match(l.strip())))
        if i == 1:
            out.append(_mk(src, "Recitals", "Commission Implementing Decision (EU) 2026/2274", i, t, "recital"))
        elif i == 2:
            out.append(_mk(src, "Art. 1–2", "Articles", i, t, "article"))
        else:
            out.append(_mk(src, "Annex", "Annex — harmonised standard", i, t, "annex"))
    return out


PLAYBOOK_SECTIONS = {
    "TAI-AG": [(3, "01 What Makes Agents Different"), (5, "02 The Anatomy of an Agent"), (7, "03 Where Things Go Wrong"),
               (10, "04 Designing for Oversight"), (12, "05 The Logging Problem"), (14, "Multi-Agent Systems"),
               (19, "Deploying Responsibly"), (22, None), (28, "Glossary"), (32, "References")],
    "TAI-GOV": [(3, "01 The Case for Governance"), (4, "02 Transparency in AI Governance"), (6, "03 EU AI Act Compliance"),
                (7, "04 Agentic AI"), (9, "05 Explainability in Practice"), (11, None)],
}


def _section(source_id, page):
    cur = None
    for start, name in PLAYBOOK_SECTIONS.get(source_id, []):
        if page >= start:
            cur = name
    return cur


def chunk_playbook(src, pages):
    out = []
    for i, t in enumerate(pages, 1):
        t = _flat(t)
        t = re.sub(r"^TrustAI (Agentic )?AI (Governance )?Playbook trustai\.com\.tr ", "", t)
        t = re.sub(r" 2026 \d+$", "", t)
        if len(t) < 80:
            continue
        for j, part in enumerate(_split_long(t, 1500)):
            sec = _section(src["source_id"], i)
            out.append(_mk(src, f"s. {i}" + (f".{j+1}" if j else ""), src["short"] + (f" — {sec}" if sec else ""), i, part, "guidance", sec or ""))
    return out


def chunk_nist(src, pages):
    lines = []
    for i, t in enumerate(pages, 1):
        if i < 6:
            continue
        for ln in t.split("\n"):
            if ln.strip():
                lines.append((i, ln.strip()))
    out, cur = [], None
    pat = re.compile(r"^(GOVERN|MAP|MEASURE|MANAGE) \d+\.\d+$")
    for pg, ln in lines:
        if pat.match(ln):
            cur = [ln, pg, []]
            out.append(cur)
        elif cur:
            cur[2].append(ln)
    res = []
    for ref, pg, ls in out:
        txt = _flat(" ".join(ls))
        for j, part in enumerate(_split_long(txt, 2000)):
            res.append(_mk(src, ref + (f" #{j+1}" if j else ""), f"NIST AI RMF Playbook — {ref}", pg, part, "guidance"))
    return res


CHUNKERS = {"AIA": chunk_aia, "DEC2274": chunk_decision, "TAI-AG": chunk_playbook,
            "TAI-GOV": chunk_playbook, "NIST-PB": chunk_nist}


def build_chunks() -> list[dict]:
    """Chunks the permitted sources; caches by file hash."""
    man = load_manifest()
    key = hashlib.sha256()
    todo = []
    for s in man["sources"]:
        if not s["rag"] or not s["api_transfer_allowed"]:
            continue
        p = config.SOURCES_DIR / s["file"]
        if not p.exists():
            continue
        key.update(file_hash(p).encode())
        todo.append((s, p))
    key.update(Path(__file__).read_bytes())
    cache = config.CACHE_DIR / f"chunks_{key.hexdigest()[:16]}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    chunks = []
    for s, p in todo:
        chunks += CHUNKERS[s["source_id"]](s, extract_pages(p))
    for c in chunks:  # after Art. 5(1)(h), '(i)' is not a letter but the exception sub-points (i)-(iii)
        if c["source_id"] == "AIA" and c["ref"] == "Art. 5(1)(i)":
            c["ref"] = "Art. 5(1)(h) istisnaları (i)–(iii)"
    seen = Counter()
    for c in chunks:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", c["ref"]).strip("-")
        cid = f"{c['source_id']}:{slug}"
        seen[cid] += 1
        c["chunk_id"] = cid if seen[cid] == 1 else f"{cid}~{seen[cid]}"
    cache.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return chunks


# --------------------------------------------------------------------------- index
class Index:
    def __init__(self):
        self.chunks = build_chunks()
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self._tok = [self._tokens(c["title"] + " " + c["ref"] + " " + c["text"]) for c in self.chunks]
        self.df = Counter(t for toks in self._tok for t in set(toks))
        self.avg = sum(len(t) for t in self._tok) / max(1, len(self._tok))
        self.emb = self._load_emb()

    @staticmethod
    def _tokens(s: str) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2]

    # ----- embeddings (persistent cache; aligned by passage id + text digest, can be updated piece by piece)
    @staticmethod
    def _key(c: dict) -> str:
        return c["chunk_id"] + "|" + hashlib.sha1(c["text"].encode()).hexdigest()[:10]

    def _emb_path(self) -> Path:
        return config.CACHE_DIR / f"emb_{config.embed_model().replace('/', '_')}.npz"

    def _migrate_legacy(self):
        """Converts an old-format (.npy, order-based) index into an id-based format using the matching old chunk list. No API call."""
        for npy in config.CACHE_DIR.glob(f"emb_{config.embed_model().replace('/', '_')}_*.npy"):
            h = npy.stem.rsplit("_", 1)[1]
            for f in config.CACHE_DIR.glob("chunks_*.json"):
                ch = json.loads(f.read_text(encoding="utf-8"))
                if hashlib.sha256("".join(c["chunk_id"] + c["text"] for c in ch).encode()).hexdigest()[:16] == h:
                    arr = np.load(npy)
                    if len(arr) == len(ch):
                        np.savez(self._emb_path(), keys=np.array([self._key(c) for c in ch]), vecs=arr)
                        return True
        return False

    def _load_emb(self):
        p = self._emb_path()
        if not p.exists() and not self._migrate_legacy():
            self.missing = len(self.chunks)
            return None
        z = np.load(p)
        have = dict(zip(z["keys"].tolist(), range(len(z["keys"]))))
        dim = z["vecs"].shape[1]
        arr = np.zeros((len(self.chunks), dim), dtype=np.float32)
        miss = 0
        for i, c in enumerate(self.chunks):
            j = have.get(self._key(c))
            if j is None:
                miss += 1
            else:
                arr[i] = z["vecs"][j]
        self.missing = miss
        return arr if miss < len(self.chunks) else None

    def has_embeddings(self) -> bool:
        return self.emb is not None

    def refresh_embeddings(self) -> bool:
        """Loads the persistent index from disk (if present); makes no API call."""
        if self.emb is None:
            self.emb = self._load_emb()
        return self.emb is not None

    def status(self) -> dict:
        return {"chunks": len(self.chunks), "embeddings": self.emb is not None, "missing": getattr(self, "missing", len(self.chunks)),
                "embed_model": config.embed_model(), "sources": sorted({c["source_id"] for c in self.chunks})}

    def build_embeddings(self, client, batch=96, usage=None) -> int:
        """Called by explicit user action; embeds only the passages that are not in the index."""
        todo = [i for i in range(len(self.chunks)) if self.emb is None or not np.any(self.emb[i])]
        if not todo:
            return 0
        vecs = {}
        for k in range(0, len(todo), batch):
            part = todo[k:k + batch]
            r = client.embeddings.create(model=config.embed_model(), input=[(self.chunks[i]["title"] + "\n" + self.chunks[i]["text"])[:6000] for i in part])
            for i, d in zip(part, r.data):
                vecs[i] = d.embedding
            if usage is not None and getattr(r, "usage", None):
                usage.append({"kind": "embedding_index", "prompt": r.usage.prompt_tokens, "completion": 0})
        dim = len(next(iter(vecs.values())))
        arr = self.emb if self.emb is not None else np.zeros((len(self.chunks), dim), dtype=np.float32)
        for i, v in vecs.items():
            x = np.asarray(v, dtype=np.float32)
            arr[i] = x / (np.linalg.norm(x) + 1e-9)
        np.savez(self._emb_path(), keys=np.array([self._key(c) for c in self.chunks]), vecs=arr)
        self.emb, self.missing = arr, 0
        return len(todo)

    # ----- search
    def bm25(self, query: str, k1=1.4, b=0.75) -> np.ndarray:
        q = self._tokens(query)
        n = len(self.chunks)
        scores = np.zeros(n)
        for i, toks in enumerate(self._tok):
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self.avg))
            scores[i] = s
        return scores

    def lookup(self, source_id: str, ref: str) -> list[dict]:
        """Direct retrieval by article/sub-point. 'Art. 5(1)' -> Art. 5(1) and its sub-points (a), (b)…"""
        out = []
        for c in self.chunks:
            if c["source_id"] != source_id:
                continue
            r = re.sub(r" #\d+$", "", c["ref"])
            if r == ref or r.startswith(ref + "(") or r.startswith(ref + " ") or r.startswith(ref + ",") or r.startswith(ref + "."):
                out.append(c)
        return out

    def retrieve(self, query: str, anchors: list[str] | None = None, k: int = 4,
                 query_vec=None, max_anchor: int = 8, source_ids: list[str] | None = None) -> list[dict]:
        """First direct citation (anchor), then hybrid (BM25 + embedding) search.
        Different sources/versions are not merged; each passage carries its own source id."""
        res, seen = [], set()
        for a in anchors or []:
            sid, ref = a.split("|", 1)
            for c in self.lookup(sid, ref)[:max_anchor]:
                if c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    res.append(c)
        if query.strip():
            s = self.bm25(query)
            s = s / (s.max() + 1e-9)
            if self.emb is not None and query_vec is not None:
                qv = np.asarray(query_vec, dtype=np.float32)
                qv /= np.linalg.norm(qv) + 1e-9
                s = 0.5 * s + 0.5 * (self.emb @ qv)
            # authority: legislation > guidance; recitals slightly lower
            pen = np.array([0.0 if c["kind"] != "recital" else 0.08 for c in self.chunks])
            auth = np.array([(5 - c["authority"]) * 0.01 for c in self.chunks])
            s = s + auth - pen
            if source_ids:  # search only in the requested collections
                s = np.where(np.array([c["source_id"] in source_ids for c in self.chunks]), s, -1.0)
            for i in np.argsort(-s)[:k * 3]:
                c = self.chunks[i]
                if c["chunk_id"] not in seen and s[i] > 0.05:
                    seen.add(c["chunk_id"])
                    res.append(c)
                    if len(res) >= len(anchors or []) + k:
                        break
        return res


def retrieve_topic(index: "Index", topic: dict, qvec=None, max_total: int = 3) -> list[dict]:
    """Topic-specific collections. Fixed anchors help find the relevant article but do not replace choosing a passage that fits the scenario:
    order within each source = first anchor, the passage best matching the scenario (embedding/keyword), remaining anchors. Sources are traversed in the order of topic['sources'];
    at least one passage remains from each source (e.g. for oversight topics a playbook passage is not excluded by legislation anchors)."""
    per_source, seen = [], set()
    anchors = topic.get("anchors") or []
    for sid in topic.get("sources") or ["AIA"]:
        mine = [a for a in anchors if a.startswith(sid + "|")]
        anc = [c for c in index.retrieve("", anchors=mine, k=0, source_ids=[sid], max_anchor=2)] if mine else []
        hyb = index.retrieve(topic["hint"], anchors=[], k=1, query_vec=qvec, source_ids=[sid]) if topic.get("hint") else []
        order = []
        for c in ([anc[0]] if anc else []) + hyb[:1] + anc[1:]:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                order.append(c)
        per_source.append(order)
    out = []
    for rnd in range(max((len(o) for o in per_source), default=0)):  # round-robin across sources: diversity comes in turn, not at the cost of irrelevant citations
        for o in per_source:
            if rnd < len(o) and len(out) < max_total:
                out.append(o[rnd])
    return out


@lru_cache(maxsize=1)
def get_index() -> Index:
    return Index()
