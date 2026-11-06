"""Derivation from the interview state and deterministic classification (no numeric formula; versioned decision table)."""
from __future__ import annotations

import json
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def policy() -> dict:
    return json.loads((config.CATALOG_DIR / "policy.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def interview_cfg() -> dict:
    return json.loads((config.CATALOG_DIR / "interview.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def controls_cfg() -> dict:
    return json.loads((config.CATALOG_DIR / "controls.json").read_text(encoding="utf-8"))


TOPICS = lambda: {t["id"]: t for t in interview_cfg()["topics"]}  # noqa: E731
STATUS_TR = {"declared_present": "Var olduğu beyan edildi", "declared_absent": "Yok olduğu beyan edildi",
             "unknown": "Bilinmiyor", "not_relevant": "Bu kullanımda ilgili değil",
             "unasked": "Sorulmadı", "covered": "Bu görüşmede ele alındı"}
ORDER = ["LOW", "MEDIUM", "HIGH"]


COUNTED_KINDS = ("question", "followup", "conflict")  # answered assessment questions


def is_counted(t: dict) -> bool:
    return t.get("answer") is not None and t["question"].get("kind") in COUNTED_KINDS


def answered_count(state: dict) -> int:
    return sum(1 for t in state["turns"] if is_counted(t))


def _definite(v) -> bool:
    return v not in (None, "unknown")


# ----------------------------------------------------------------------- derivation
def derive(state: dict) -> dict:
    """Sequential fold over the turns.
    - 'unknown' NEVER overwrites a definite statement.
    - If a different definite value arrives (except p_* refinements), a conflict is recorded and only that item is asked.
    - Equivalent/linked fields are closed by a single answer (equivalences); skipped branches are kept in derived together with their reasons."""
    cfg = interview_cfg()
    spec = cfg["slots"]
    slots, src, controls, csrc, basis = {}, {}, {}, {}, {}
    asked, followups, conflict_asked, conflicts, unresolved_conflicts = {}, {}, {}, {}, {}
    st_eu = (state.get("settings") or {}).get("eu_link", "unknown")
    if st_eu in ("yes", "no"):  # assessment setting
        slots["eu_link"], src["eu_link"] = st_eu, "AYAR"
    for t in state["turns"]:
        if t.get("answer") is None:
            continue
        aid, q = t["id"], t["question"]
        kind = q.get("kind")
        u = t.get("updates") or {}
        for k, v in (u.get("slots") or {}).items():
            old = slots.get(k)
            if v == "unknown" and _definite(old):
                continue
            if (kind != "conflict" and spec.get(k) != "text" and not k.startswith("p_")
                    and _definite(old) and _definite(v) and v != old):
                conflicts[k] = {"old": old, "new": v, "old_src": src[k], "new_src": aid, "old_slot": k}
            slots[k], src[k] = v, aid
        if kind == "conflict" and q.get("fact"):
            f = q["fact"]
            conflict_asked[f] = conflict_asked.get(f, 0) + 1
            conflicts.pop(f, None)
            if not _definite((u.get("slots") or {}).get(f)):
                slots[f], src[f] = "unknown", aid  # conflict could not be resolved: recorded as unknown
                unresolved_conflicts[f] = aid     # unresolved conflict: no certain finding is produced
            else:
                unresolved_conflicts.pop(f, None)
        for cid, cv in (u.get("controls") or {}).items():
            old = controls.get(cid)
            if cv.get("status") == "unknown" and old and old.get("status") in ("declared_present", "declared_absent"):
                continue  # 'unknown' does not overwrite a definite statement
            controls[cid], csrc[cid] = cv, aid
        for b in u.get("basis") or []:
            basis[b["key"]] = {**b, "answer_id": aid}
        if kind == "followup" and q.get("fact"):
            followups[q["fact"]] = followups.get(q["fact"], 0) + 1
        if kind in ("question", "followup") and q.get("topic") in TOPICS():
            asked[q["topic"]] = asked.get(q["topic"], 0) + 1
            top = TOPICS()[q["topic"]]
            for sl in required_slots(top):
                if sl not in slots:  # asked but nothing could be extracted: unknown
                    slots[sl], src[sl] = "unknown", aid
            for cid in ([] if top["slots"] else top["controls"]):
                if cid not in controls and cid not in ("C01", "C02"):
                    controls[cid], csrc[cid] = {"status": "unknown", "note": ""}, aid
    derived = {}
    for r in cfg.get("equivalences", []):
        if all(slots.get(k) == v for k, v in r["if"].items()):
            for k, v in r["then"].items():
                if slots.get(k) is None:
                    first = next(iter(r["if"]))
                    slots[k], src[k], derived[k] = v, src[first], r["reason"]
    for gid, g in cfg.get("prohibition_groups", {}).items():
        # if one precondition is explicitly 'no', the sub-prohibition screens are skipped; the reason is recorded.
        if any(slots.get(k) == "no" for k in g["requires"]):
            for dep in g["deps"]:
                if slots.get(dep) is None:
                    slots[dep], src[dep], derived[dep] = "none", src[next(k for k in g["requires"] if slots.get(k) == "no")], g["skip_tr"]
    for r in cfg.get("exclusions", []):
        if all(slots.get(k) == v for k, v in r["if"].items()):
            for k, bad in r["excludes"].items():
                if slots.get(k) in bad:
                    ifk = next(iter(r["if"]))
                    if conflict_asked.get(k, 0) == 0:
                        conflicts[k] = {"old": slots[ifk], "new": slots[k], "old_src": src[ifk], "new_src": src[k], "old_slot": ifk,
                                        "question": r.get("question"), "resolve": r.get("resolve")}
                    else:  # asked and still conflicting
                        unresolved_conflicts[k] = src.get(k)
    def _decl(cid, slot_names, yes, no):
        vals = [(k, slots.get(k)) for k in slot_names if slots.get(k) is not None]
        if cid in controls or not vals:
            return
        k, v = vals[0]
        if v in yes:
            controls[cid], csrc[cid] = {"status": "declared_present", "note": "beyan edildi"}, src.get(k)
        elif v in no:
            controls[cid], csrc[cid] = {"status": "declared_absent", "note": "yok beyanı"}, src.get(k)
        else:  # partial / unknown: recorded as unknown
            controls[cid], csrc[cid] = {"status": "unknown", "note": "kısmen veya bilinmiyor"}, src.get(k)
    _decl("C09", ["approval_bound", "human_review"], ("yes",), ("no",))
    _decl("C11", ["limits_enforced"], ("technical",), ("policy_only", "none"))
    return {"slots": slots, "slots_src": src, "controls": controls, "controls_src": csrc, "basis": basis, "asked": asked,
            "derived": derived, "followups": followups, "conflicts": conflicts, "conflict_asked": conflict_asked,
            "unresolved_conflicts": unresolved_conflicts}


def control_gaps(d: dict, legal_code: str) -> dict:
    """Controls relevant to the result and their states: {cid: {"why","state"}}; state: addressed | unknown | unasked.
    An unasked control is not treated as 'not relevant'; 'unasked' lowers result sufficiency, 'unknown' is listed as a critical gap."""
    out, s = {}, d["slots"]
    for r in controls_cfg().get("result_relevance", []):
        if "when_legal" in r:
            ok = legal_code in r["when_legal"]
        elif "when_any" in r:
            ok = any(s.get(k) in v for k, v in r["when_any"].items())
        else:
            ok = all(s.get(k) in v for k, v in r["when"].items())
        if not ok:
            continue
        for cid in r["controls"]:
            st = d["controls"].get(cid, {}).get("status")
            state = "unasked" if st is None else "unknown" if st == "unknown" else "addressed"
            out[cid] = {"why": r["why_tr"], "state": state}
    return out


def control_scope(d: dict, legal_code: str) -> dict:
    """Reasoned 'not relevant for this use' scope rules (from the catalog) for controls that have not been declared."""
    out, s = {}, d["slots"]
    for r in controls_cfg().get("scope_rules", []):
        if "when_legal" in r:
            ok = legal_code in r["when_legal"]
        else:
            tests = [s.get(k) in v for k, v in r["when"].items()]
            ok = all(tests) if r.get("all") else any(tests) if len(tests) == 1 else all(tests)
        if ok:
            for cid in r["controls"]:
                if d["controls"].get(cid, {}).get("status") in (None, "unknown"):
                    out[cid] = {"status": "not_relevant", "note": r["reason_tr"]}
    return out


def group_of(dep: str) -> str | None:
    for gid, g in interview_cfg().get("prohibition_groups", {}).items():
        if dep in g["deps"]:
            return gid
    return None


def preconditions_met(slots: dict, gid: str) -> bool:
    """For a detail branch, ALL preconditions must be 'yes' by user declaration."""
    return all(slots.get(k) == v for k, v in interview_cfg()["prohibition_groups"][gid]["requires"].items())


def detail_groups(slots: dict) -> list[str]:
    """Prohibition groups that need a detail question: all preconditions are 'yes' and the sub-field is unresolved."""
    return [gid for gid, g in interview_cfg().get("prohibition_groups", {}).items()
            if preconditions_met(slots, gid) and any(slots.get(x) in (None, "possible") for x in g["deps"])]


def topic_active(top: dict, slots: dict) -> bool:
    c = top.get("condition")
    if not c:
        return True
    eu_out = slots.get("eu_link") == "no"
    mode = slots.get("action_mode")
    if c == "approval_exists_yes":
        return slots.get("approval_exists") == "yes"
    if c == "person_eval_yes":
        return not eu_out and slots.get("g_person_eval") == "yes"
    if c == "annex3_area_set":
        return not eu_out and slots.get("annex3_area") not in (None, "none", "unknown")
    if c == "acts":
        return mode in ("approved_action", "envelope_action", "self_expanding", "unknown", None)
    if c == "acts_or_reads":
        return mode != "info_only"
    if c == "recommends":
        return mode in ("recommend", "draft")
    if c == "harm_known":
        return slots.get("max_harm") in ("limited", "serious", "critical", "unknown")
    if c == "personal_data_set":
        return slots.get("personal_data") in ("ordinary", "special", "unknown")
    if c == "synthetic_yes":
        return slots.get("t_synthetic_content") == "yes"
    if c == "legal_high_or_unknown":
        return not eu_out and slots.get("annex3_area") not in (None, "none")
    return True


def required_slots(top: dict) -> list[str]:
    """Slots required to resolve a topic: free-text fields (summary/description) are optional, except 'purpose'."""
    spec = interview_cfg()["slots"]
    return [sl for sl in top["slots"] if spec[sl] != "text" or sl == "purpose"]


def option_sets(top: dict) -> dict:
    """Option label → {slot: value}. A choice is stored together with the answered field and value (the label is not left to re-interpretation).
    Without an explicit mapping (option_map), yes/no/unknown questions with a single required slot are mapped automatically."""
    if top.get("option_map"):
        return top["option_map"]
    req = required_slots(top)
    spec = interview_cfg()["slots"]
    if len(req) == 1 and spec[req[0]] == ["yes", "no", "unknown"]:
        return {"Evet": {req[0]: "yes"}, "Hayır": {req[0]: "no"}, "Bilmiyorum": {req[0]: "unknown"}}
    return {}


def control_option_sets(top: dict) -> dict:
    """Option label → {control: status} for topics without slots. Labels starting with 'Evet/Hayır/Bilmiyorum/Kısmen' are recorded deterministically."""
    if top["slots"] or not top.get("options"):
        return {}
    out = {}
    for o in top["options"]:
        low = o.strip().lower()
        st, note = (("declared_present", "beyan edildi") if low.startswith("evet") else ("declared_absent", "yok beyanı") if low.startswith("hayır")
                    else ("unknown", "") if low.startswith("bilmiyorum") else ("unknown", "kısmen / belirsiz") if low.startswith("kısmen") else (None, None))
        if st:
            out[o] = {cid: {"status": st, "note": note} for cid in top["controls"] if cid not in ("C01", "C02")}
    return out


def topic_resolved(top: dict, d: dict) -> bool:
    n = d["asked"].get(top["id"], 0)
    if top["id"] == "prohibited_detail":
        return (not topic_active(top, d["slots"])) or n >= 2
    req = required_slots(top)
    if req:
        return all(d["slots"].get(sl) is not None for sl in req)
    return n >= 1


OPTIMISTIC = {"approval_exists_yes", "person_eval_yes", "annex3_area_set", "acts", "acts_or_reads", "recommends", "harm_known", "personal_data_set",
              "synthetic_yes", "legal_high_or_unknown"}


def open_topics(state: dict, optimistic: bool = False) -> list[dict]:
    """Open and valid topics in order of priority. optimistic=True: topics whose condition may change with this answer
    before it is processed are also treated as candidates."""
    d = derive(state)
    out = []
    for top in interview_cfg()["topics"]:
        if topic_resolved(top, d):
            continue
        if topic_active(top, d["slots"]) or (optimistic and top.get("condition") in OPTIMISTIC and d["slots"].get("eu_link") != "no"):
            out.append(top)
    return sorted(out, key=lambda t: -t.get("priority", 0))


# ----------------------------------------------------------------------- classification
def forbidden_findings(d: dict) -> list[dict]:
    """Only if the conditions of the prohibition are supported and the exception/scope/date assessment is supported."""
    pol = policy()["prohibition_refs"]
    out = []
    if d["slots"].get("eu_link") == "no":
        return out  # if it is declared that there is no EU link
    for k, (ref, name) in pol.items():
        gid = group_of(k)
        if d["slots"].get(k) == "supported" and gid and preconditions_met(d["slots"], gid):
            b = d["basis"].get(k, {})
            if b.get("conditions_confirmed") and b.get("exception_considered") and b.get("scope_date_considered"):
                out.append({"key": k, "ref": ref, "name": name, "answer_id": b["answer_id"],
                            "slot_answer_id": d["slots_src"].get(k)})
    return out


AUTONOMY_SLOTS = {"action_mode", "approval_exists", "approval_bound", "limits_enforced", "authority_expansion"}


def autonomy_conflict(d: dict) -> bool:
    """Is there an unresolved (unasked or unanswered) conflict in the approval/autonomy fields?"""
    return bool((set(d["conflicts"]) | set(d["unresolved_conflicts"])) & AUTONOMY_SLOTS)


def levels(d: dict) -> tuple[str | None, str | None, list[str]]:
    s, notes = d["slots"], []
    P = policy()
    mode = s.get("action_mode")
    if autonomy_conflict(d):  # no certain L is produced from conflicting approval/autonomy statements
        return None, P["impact_map"].get(s.get("max_harm")), notes
    L = P["autonomy_map"].get(mode)
    if L == "L3" and s.get("approval_bound") != "yes":
        L = "L4"
        notes.append("Onay işleme özel olduğu beyan edilmediği için L4 olarak ele alındı (L3 için onayın o eyleme bağlı olması gerekir).")
    if L == "L4" and s.get("limits_enforced") in ("policy_only", "none"):
        L = "L5"
        notes.append("Sınırlar teknik olarak uygulanmadığı beyan edildiği için ulaşılabilir yetki sınırsız kabul edilip L5 olarak ele alındı.")
    if L in ("L3", "L4") and s.get("authority_expansion") == "yes":
        L = "L5"
        notes.append("Yetki genişletme beyan edildiği için L5.")
    I = P["impact_map"].get(s.get("max_harm"))
    return L, I, notes


def legal_path(d: dict) -> dict:
    s, P = d["slots"], policy()
    area, a1, eu = s.get("annex3_area"), s.get("annex1_product"), s.get("eu_link")
    res = {"code": "undetermined", "high": False, "uncertain": [], "refs": [("AIA", "Art. 6")], "answers": []}
    if eu == "no":
        res.update(code="out_of_scope_declared", refs=[("AIA", "Art. 2(1)")], answers=[d["slots_src"].get("eu_link")])
        res["scope_note"] = ("AB bağlantısı olmadığı beyan edildi; Tüzük kapsamı dışında kalabilir (Art. 2). Bu beyan hukuki kapsam analizi olarak doğrulanmadı. "
                             "Görüşmenin geri kalanı AB yükümlülüğü değil, genel TrustAI risk değerlendirmesidir.")
        return res
    elif eu in (None, "unknown"):
        res["applicability"] = "unknown"  
    if a1 == "yes":
        res.update(code="annex1_high", high=True)
        res["refs"] = [("AIA", "Art. 6(1)"), ("AIA", "Annex I (giriş)")]
        res["answers"] = [d["slots_src"].get("annex1_product")]
        return res
    if area in P["annex3_areas_tr"]:
        res["refs"] = [("AIA", "Art. 6(2)"), ("AIA", P["annex3_refs"][area])]
        res["answers"] = [d["slots_src"].get("annex3_area")]
        prof, exc, infl = s.get("profiling"), s.get("art6_3_exception"), s.get("influences_decision")
        res["refs"] += [("AIA", "Art. 6(3)")]
        if prof == "yes":
            res.update(code="annex3_high_profiling", high=True)
        elif exc in ("narrow_procedural", "improve_prior_human_work", "detect_patterns", "preparatory") and infl == "no" and prof == "no":
            res.update(code="annex3_exception_claimed", high=False)
            res["refs"].append(("AIA", "Art. 6(4)"))
            res["uncertain"].append("Art. 6(3) istisnası beyana dayanır; sağlayıcının değerlendirmeyi belgelemesi ve kayıt yükümlülüğü (Art. 6(4), 49(2)) ayrıca gerekir.")
        else:
            res.update(code="annex3_high", high=True)
            if exc in (None, "unknown") or prof in (None, "unknown"):
                res["uncertain"].append("Art. 6(3) istisnasının ve profillemenin uygulanıp uygulanmadığı kesinleşmedi; yüksek riskli yol temkinli biçimde kabul edildi.")
        res["answers"] += [d["slots_src"].get(k) for k in ("profiling", "art6_3_exception", "influences_decision") if d["slots_src"].get(k)]
    elif area == "none":
        res.update(code="not_high_by_declaration")
        res["answers"] = [d["slots_src"].get("annex3_area")]
        res["refs"].append(("AIA", "Annex III (giriş)"))
    return res


def classify(state: dict) -> dict:
    d = derive(state)
    s, P = d["slots"], policy()
    forb = forbidden_findings(d)
    L, I, lnotes = levels(d)
    legal = legal_path(d)
    transp = [{"slot": k, "ref": ref, "name": name, "answer_id": d["slots_src"].get(k)}
              for k, (ref, name) in P["transparency_refs"].items() if s.get(k) == "yes"]
    unresolved = []
    eu_out = s.get("eu_link") == "no"
    for k in P["critical_slots"]:
        if eu_out and (k.startswith("p_") or k == "annex3_area"):
            continue  # with a declaration of no EU link, the legal screening is not counted as a critical gap
        v = s.get(k)
        bad = v in (None, "unknown", "") or (k.startswith("p_") and v == "possible")
        if bad:
            unresolved.append(k)
    unc = list(legal["uncertain"])
    if autonomy_conflict(d):
        unc.append("Onay/otonomi beyanları çelişiyor; kesin bir onay/otonomi bulgusu üretilmedi.")
        if "action_mode" not in unresolved:
            unresolved.append("action_mode")
    for k in unresolved:
        unc.append(f"Kritik bilgi eksik: {P['critical_labels_tr'].get(k, k)}.")
    unc += lnotes
    eu_s = s.get("eu_link") if s.get("eu_link") in ("yes", "no") else "unknown"
    out = {"applicability": eu_s, "legal": legal, "transparency": transp, "L": L, "I": I, "level_notes": lnotes,
           "assurance": None, "uncertainties": unc, "forbidden": forb, "unresolved": unresolved,
           "escalation": [], "risk_label": None, "assessment_status": "incomplete",
           "label_basis": [], "policy_version": P["policy_version"]}
    out["control_gaps"] = control_gaps(d, legal["code"])
    if L and I:
        out["assurance"] = P["assurance_matrix"][I][L]
    if forb:
        out.update(risk_label="FORBIDDEN", assessment_status="complete")
        out["label_basis"] = [("forbidden", forb)]
        return out
    prohib_open = [k for k in unresolved if k.startswith("p_")]
    if prohib_open:
        return out  # no label is given before the prohibition screening is complete
    op = P["label_table"][I][L] if (L and I) else None
    if I == "I3" and not L:
        op = "HIGH"
    if op and I != "I0":
        fac = []
        if s.get("reach") == "wide":
            fac.append("yayılım geniş")
        if s.get("reversibility") == "irreversible":
            fac.append("zarar geri alınamaz")
        if s.get("vulnerable_groups") == "yes":
            fac.append("korunmasız gruplar etkileniyor")
        if s.get("personal_data") == "special":
            fac.append("hassas veri kullanılıyor")
        if fac and op != "HIGH":
            op = ORDER[ORDER.index(op) + 1]
            out["escalation"] = fac
    out["operational_label"] = op
    label = None
    if legal["high"]:
        label = "HIGH"
    elif op == "HIGH":
        label = "HIGH"
    elif op and not unresolved:
        label = op
    if label:
        out.update(risk_label=label, assessment_status="complete")
        titles = {c["id"]: c["title_tr"] for c in controls_cfg()["controls"]}
        for cid, g in out["control_gaps"].items():
            if g["state"] == "unknown":
                out["uncertainties"].append(f"Sonuca etki eden kontrol bilinmiyor: {cid} {titles[cid]}.")
            elif g["state"] == "unasked" and label in ("LOW", "MEDIUM"):
                out["assessment_status"] = "partial"  # a label exists but a relevant control affecting the result has not been assessed
                out["uncertainties"].append(f"Sonuca etki eden kontrol henüz değerlendirilmedi: {cid} {titles[cid]} ({g['why']})")
        if unresolved and label == "HIGH":
            out["uncertainties"].append("Etiket mevcut bilgilerle HIGH; eksik alanlar etiketi düşürmez ancak ayrıntıyı değiştirebilir.")
    return out
