import classify as C, interview as iv, planner as P, report as rp, rag, llm
from conftest import FakeClient

GATES_NO = {"g_biometric": "no", "g_influence": "no", "g_person_eval": "no"}
EXAMPLE = ("Agent, bir e-ticaret şirketinde müşterilerin iade taleplerini inceleyip destek çalışanına öneri sunuyor. Son kararı çalışan veriyor; "
           "agent ödeme veya iade işlemi yapmıyor ve müşterilerle doğrudan konuşmuyor.")
EX_SLOTS = {"purpose": "Müşteri iade taleplerini inceleyip destek çalışanına öneri sunmak", "affected_decisions": "iade kararı",
            "action_mode": "recommend", "t_interacts_people": "no"}
TRUTH = {"eu_link": "yes", "org_role": "deployer", "use_status": "live", **GATES_NO, "annex3_area": "none", "annex1_product": "no", "max_harm": "limited",
         "personal_data": "ordinary", "vulnerable_groups": "no", "t_synthetic_content": "no", "reversibility": "reversible", "reach": "single",
         "human_review": "yes", "art6_3_exception": "none"}


def start(index):
    s = iv.new_state(); iv.add_first(s, index); return s


def run_truth(s, index, truth, limit=40):
    """Answers each question with 'truth' for the slots of the asked topic (a scripted loop; it only tests our own flow)."""
    topics = []
    while s["phase"] == "interview" and C.answered_count(s) < limit:
        q = s["turns"][-1]["question"]
        topics.append((q["kind"], q["topic"], q.get("fact")))
        top = C.TOPICS()[q["topic"]]
        upd = {sl: truth[sl] for sl in C.required_slots(top) if sl in truth}
        if q["kind"] == "followup" and q.get("fact") in truth:
            upd = {q["fact"]: truth[q["fact"]]}
        iv.process(s, len(s["turns"]) - 1, "cevap", index, FakeClient([{"slots": upd}]))
    return topics


def test_sources_metadata(index):
    man = rag.load_manifest()["sources"]
    assert {m["source_id"] for m in man if not m["api_transfer_allowed"]} == {"ISO42001"}
    assert all(c["source_id"] != "ISO42001" for c in index.chunks)
    assert all(c["version"] and c["page"] for c in index.chunks)
    assert index.lookup("AIA", "Art. 6(3)") and index.lookup("AIA", "Art. 113")


def test_catalog_citations():
    import tools.verify_catalog as v
    assert v.main() == 0


# ---------------------------------------------------------------- causes of repetition
def test_unknown_never_overwrites_definite(index):
    """Old bug: a later 'unknown' overwrote a definite 'no' answer and reopened the detail branch."""
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "eu_link": "yes", "g_biometric": "no"}}]))
    iv.process(s, 1, "y", index, FakeClient([{"slots": {"annex3_area": "none", "g_biometric": "unknown"}}]))
    d = C.derive(s)
    assert d["slots"]["g_biometric"] == "no" and d["slots"]["p_rbi_law_enforcement"] == "none"
    assert not [c for c in P.candidates(s) if c["topic"]["id"] in ("g_biometric", "prohibited_detail") or c.get("gate") == "g_biometric"]


def test_conflict_asks_only_the_conflicting_fact_once(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "eu_link": "yes", "g_biometric": "no"}}]))
    iv.process(s, 1, "y", index, FakeClient([{"slots": {"annex3_area": "biometrics"}}]))
    q = s["turns"][-1]["question"]
    assert q["kind"] == "conflict" and q["fact"] == "annex3_area" and not iv.is_vague(q["text"])
    assert "biyometrik" in q["text"].lower() and "hassas kullanım alanı" in q["text"] and len(q["options"]) == 3
    iv.process(s, 2, "bilmiyorum", index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert d["slots"]["annex3_area"] == "unknown" and not d["conflicts"]  # could not be resolved → unknown, not 'no'
    assert all(t["question"]["kind"] != "conflict" for t in s["turns"][3:])  # not asked a second time
    assert C.answered_count(s) == 3


def test_equivalent_facts_close_each_other(index):
    s = {"turns": [{"id": "A1", "question": {"topic": "annex3", "kind": "question"}, "answer": "x",
                    "updates": {"slots": {"annex3_area": "biometrics"}, "controls": {}, "basis": []}}]}
    d = C.derive(s)
    assert d["slots"]["g_biometric"] == "yes" and "g_biometric" in d["derived"]
    ids = [t["id"] for t in C.open_topics(s)]
    assert "g_biometric" not in ids


def test_dont_know_gets_one_simple_followup_then_moves_on(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "g_influence": "no", "g_person_eval": "no", "annex3_area": "none"}}]))
    s["turns"][-1]["question"] = iv.make_question(C.TOPICS()["g_biometric"], index, s)  # the user was asked the biometrics question
    iv.process(s, len(s["turns"]) - 1, "bilmiyorum", index, FakeClient([{"slots": {"g_biometric": "unknown"}}]))
    q = s["turns"][-1]["question"]
    assert q["kind"] == "followup" and q["fact"] == "g_biometric" and "Örneğin" in q["text"]
    iv.process(s, len(s["turns"]) - 1, "hâlâ bilmiyorum", index, FakeClient([{"slots": {"g_biometric": "unknown"}}]))
    assert all(t_["question"].get("fact") != "g_biometric" for t_ in s["turns"][len(s["turns"]) - 1:])
    d = C.derive(s)
    assert d["slots"]["g_biometric"] == "unknown" and d["slots"].get("p_facial_scraping") is None  # was not counted as 'no'
    assert not [c for c in P.candidates(s) if c["kind"] == "followup"]
    cls = C.classify(s)
    assert cls["risk_label"] is None and any("yasak" in x.lower() for x in cls["uncertainties"])


def test_no_unrequested_defaults(index):
    s = start(index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    d = C.derive(s)
    for k in ("g_biometric", "g_influence", "g_person_eval", "annex3_area", "max_harm", "personal_data", "approval_bound", "human_review", "eu_link"):
        assert d["slots"].get(k) is None, k  # 'no' was not derived from the scenario


# ---------------------------------------------------------------- refund example
def test_refund_example_flow(index):
    s = start(index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    d = C.derive(s)
    assert d["slots"]["action_mode"] == "recommend" and d["slots"]["t_interacts_people"] == "no" and d["slots_src"]["action_mode"] == "A1"
    ids = [c["topic"]["id"] for c in P.candidates(s)]
    for known in ("purpose", "action_mode", "t_interacts", "approval_bound", "limits_enforced", "authority_expansion", "stop_recover"):
        assert known not in ids, known
    assert {"annex3", "max_harm", "g_biometric"} <= set(ids) and "eu_link" not in ids  # the EU link is not asked (it comes from the settings)
    topics = run_truth(s, index, TRUTH)
    n = C.answered_count(s)
    assert s["phase"] == "summary" and 8 <= n <= 15, (n, topics)
    asked = [t for _, t, _ in topics]
    assert len(asked) == len(set(asked)), asked  # no topic was asked again
    assert "action_mode" not in asked and "t_interacts" not in asked and "approval_bound" not in asked
    cls = C.classify(s)
    assert cls["risk_label"] == "LOW" and cls["L"] == "L1" and cls["I"] == "I1"
    rep = rp.build_report(s, index)
    st = {c["id"]: c["status"] for c in rep["controls"]}
    assert st["C16"] == "not_relevant" and st["C20"] == "not_relevant"
    assert st["C09"] == "declared_present" and cls["assessment_status"] == "complete"  # C09, which affects the result, was addressed by declaration
    assert st["C08"] == "unasked" and st["C11"] == "unasked" and st["C12"] == "unasked"  # an unasked control was automatically counted neither as 'not relevant' nor as 'present'
    assert all(c["note"] for c in rep["controls"] if c["status"] == "not_relevant")
    assert rep["verification"]["all_ok"] and rp.pdf_bytes(rep)[:4] == b"%PDF"


def test_ends_when_enough_and_context_is_limited(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {**EX_SLOTS, **TRUTH}}]))  # everything was given in the first answer
    assert s["phase"] == "summary" or all(c["topic"].get("context") for c in P.candidates(s))
    ctx = 0
    while s["phase"] == "interview":
        ctx += 1
        iv.process(s, len(s["turns"]) - 1, "x", index, FakeClient([{"slots": {}}]))
    assert ctx <= 3 and C.answered_count(s) <= 4


def test_hard_limit_20_and_incomplete(index):
    s = start(index)
    while s["phase"] == "interview" and C.answered_count(s) < 40:
        iv.process(s, len(s["turns"]) - 1, "bilmiyorum", index, FakeClient([{"slots": {}}]))
    assert C.answered_count(s) <= 20 and s["phase"] == "summary"
    facts = [t["question"].get("fact") for t in s["turns"] if t["question"].get("kind") == "followup"]
    assert len(facts) == len(set(facts))  # at most one follow-up for the same item
    cls = C.classify(s)
    assert cls["risk_label"] is None and cls["assessment_status"] == "incomplete" and cls["unresolved"]
    rep = rp.build_report(s, index)
    assert any("Kritik bilgi eksik" in g or "tamamlanamadı" in g for g in rep["critical_gaps"])


def test_progress_summary_from_state_only(index):
    s = start(index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    pg = P.progress(s)
    assert "kullanım amacı" in pg["known"] and "agentın yetkisi" in pg["known"]
    assert "AB bağlantısı" not in pg["need"] and "yasak kullanım taraması" in pg["need"]


# ---------------------------------------------------------------- question clarity / options
def test_mixed_topic_model_question_is_replaced(index):
    s = start(index)
    mixed = "Agent biyometrik veri kullanıyor mu, ve işe alım gibi hassas bir alanda mı çalışıyor?"
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "eu_link": "yes"}, "topic": "g_biometric", "question": mixed}]))
    q = s["turns"][-1]["question"]
    if q["topic"] == "g_biometric":
        assert q["text"] == C.TOPICS()["g_biometric"]["default_q"] and "işe alım" not in q["text"] and q["options"] == ["Evet", "Hayır", "Bilmiyorum"]
    assert not iv.mixes_topics(C.TOPICS()["g_biometric"]["default_q"], C.TOPICS()["g_biometric"])
    assert "biyomet" not in C.TOPICS()["personal_data"]["default_q"].lower()
    assert "biyomet" not in C.TOPICS()["annex3"]["default_q"].lower() and "Biyometri" not in " ".join(C.TOPICS()["annex3"]["options"])


def test_oversight_retrieves_playbook(index):
    for tid in ("review_basis", "approval_bound", "stop_ability", "rollback_ability", "limits_enforced", "authority_expansion"):
        ps = rag.retrieve_topic(index, C.TOPICS()[tid], max_total=3)
        assert {p["source_id"] for p in ps} & {"TAI-AG", "NIST-PB"}, tid
    assert {p["source_id"] for p in rag.retrieve_topic(index, C.TOPICS()["annex3"], max_total=2)} == {"AIA"}
    assert all(p["source_id"] != "ISO42001" for t in C.TOPICS().values() for p in rag.retrieve_topic(index, t))


def test_legal_passage_keeps_conditions(index):
    ps = index.lookup("AIA", "Art. 6(3)")[0]["text"]
    for needle in ("narrow procedural task", "profiling of natural persons"):
        assert needle in ps
    ids = [p["chunk_id"] for p in rag.retrieve_topic(index, C.TOPICS()["art6_role"], max_total=3)]
    assert "AIA:Art-6-3" in ids and "AIA:Art-6-4" in ids


def test_humanize_and_why_storage(index):
    assert "provider" not in iv.humanize("provider ve importer_distributor rolü approved_action").lower()
    s = start(index)
    fc = FakeClient([{"slots": {"purpose": "p"}, "topic": "eu_link", "question": "Rolünüz provider mı?"}])
    iv.process(s, 0, "x", index, fc)
    q = s["turns"][-1]["question"]
    assert "provider" not in q["text"] and isinstance(q["why"], dict) and q["why"]["missing"] and "passage_ids" in q
    assert fc.calls == 1


def test_index_status_reflects_disk(index):
    st = index.status()
    assert st["chunks"] > 1000 and st["embeddings"] == index.refresh_embeddings()


def test_out_of_scope_is_general(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "eu_link": "no"}, "topic": "g_biometric"}]))
    q = s["turns"][-1]["question"]
    assert q["framing"] == "general" or q["topic"] not in iv.LEGAL_ONLY
    assert C.classify(s)["legal"]["code"] == "out_of_scope_declared"


def test_edit_invalidates_later_turns(index):
    s = start(index)
    fc = FakeClient([{"slots": {"purpose": "p", "eu_link": "yes"}}, {"slots": {"g_influence": "no"}}])
    iv.process(s, 0, "a", index, fc)
    iv.process(s, 1, "b", index, fc)
    assert len(s["turns"]) == 3 and C.classify(s)["assessment_status"] == "incomplete"
    iv.process(s, 0, "düzeltilmiş", index, FakeClient([{"slots": {"purpose": "q"}}]))
    assert len(s["turns"]) == 2 and C.derive(s)["slots"]["purpose"] == "q" and "g_influence" not in C.derive(s)["slots"]


def test_first_question_has_source_fields(index):
    q = iv.first_question(index)
    assert q["passage_ids"] and not q["no_passage"] and q["why"]["source_note"]
    c = index.by_id[q["passage_ids"][0]]
    assert c["ref"] == "Art. 3(12)" and c["page"] and c["version"] and c["url"]


def test_no_unsupported_source_note(index):
    q = iv.make_question(C.TOPICS()["ctl_owner"], index, {"turns": []}, nxt={"missing": "m", "clarifies": "c", "source_note": "Kaynak şunu der", "passage_ids": ["yok:yok"]}, text="Sorumlu sahip var mı?")
    assert q["no_passage"] and q["why"]["source_note"] == "" or q["generic_why"]


def test_technical_error_keeps_answer_without_extra_question(index):
    s = start(index)
    class Boom(FakeClient):
        def __init__(self):
            super().__init__([])
            class _C:
                def create(self, **kw): raise RuntimeError("ağ hatası")
            class _Chat: completions = _C()
            self.chat = _Chat()
    try:
        iv.process(s, 0, EXAMPLE, index, Boom())
    except RuntimeError:
        pass
    assert s["turns"][0]["answer"] == EXAMPLE and len(s["turns"]) == 1 and C.answered_count(s) == 1
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    assert C.answered_count(s) == 1 and len(s["turns"]) == 2


def test_transparency_and_approval_do_not_downgrade():
    base = {"purpose": "p", "eu_link": "yes", "annex3_area": "employment", "profiling": "no", "art6_3_exception": "none", "influences_decision": "yes",
            "action_mode": "recommend", "max_harm": "minimal", "t_interacts_people": "yes", "human_review": "yes", **GATES_NO}
    s = {"turns": [{"id": "A1", "question": {"topic": "annex3", "kind": "question"}, "answer": "x", "updates": {"slots": dict(base), "controls": {}, "basis": []}}]}
    c = C.classify(s)
    assert c["risk_label"] == "HIGH" and c["legal"]["high"]
    s["turns"][0]["updates"]["slots"]["annex3_area"] = "none"
    c = C.classify(s)
    assert c["risk_label"] == "LOW" and c["transparency"]


def test_low_with_unaddressed_relevant_control_is_not_presented_as_complete(index):
    """Old problem: a 'complete' LOW was given with 17 controls unasked."""
    s = {"settings": {}, "turns": [{"id": "A1", "question": {"topic": "purpose", "kind": "question"}, "answer": "x",
         "updates": {"slots": {"purpose": "p", "action_mode": "recommend", "annex3_area": "none", "max_harm": "limited", **GATES_NO}, "controls": {}, "basis": []}}]}
    c = C.classify(s)
    assert c["risk_label"] == "LOW" and c["assessment_status"] == "partial" and "C09" in c["control_gaps"]
    assert any("C09" in u for u in c["uncertainties"])
    assert P.candidates(s)[0]["topic"]["id"] == "review_basis"  # the missing relevant control is completed first
    s["turns"][0]["updates"]["slots"]["human_review"] = "yes"
    assert C.classify(s)["assessment_status"] == "complete"


def test_scope_settings_default_turkey_and_no_eu_question(index):
    s = iv.new_state()
    assert s["settings"]["scope_country"] == "Türkiye" and s["settings"]["eu_link"] == "unknown"
    iv.add_first(s, index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    c = C.classify(s)
    assert c["applicability"] == "unknown" and c["legal"]["code"] != "out_of_scope_declared"  # the Türkiye assumption does not produce an 'outside the EU' result
    assert "eu_link" not in [x["topic"]["id"] for x in P.candidates(s)]
    s2 = iv.new_state({"eu_link": "no"})
    assert C.classify(s2)["applicability"] == "no" and C.classify(s2)["legal"]["code"] == "out_of_scope_declared"
    s3 = iv.new_state({"eu_link": "yes"})
    assert C.derive(s3)["slots"]["eu_link"] == "yes" and C.derive(s3)["slots_src"]["eu_link"] == "AYAR"


def test_result_summary_is_short_and_sourced(index):
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    run_truth(s, index, TRUTH)
    rep = rp.build_report(s, index)
    assert rep["status_text"] == "Tamamlandı" and rep["risk_label"] == "LOW"
    assert len(rep["key_findings"]) <= 3 and len(rep["improvements"]) <= 3
    assert rep["use_text"].count(".") <= 3 and rep["note"] == "Bu ön değerlendirme kullanıcı beyanlarına dayanır; uygunluk belgesi değildir."
    assert rep["context"]["scope_country"] == "Türkiye" and rep["context"]["applicability_tr"] == "Belirlenmedi"
    assert any("uygulanabilirliği belirlenmedi" in g for g in rep["critical_gaps"])  # the critical limitation was not hidden
    by = {c["no"] for c in rep["cited"]}
    for k in rep["key_findings"]:
        assert set(k["cite_nos"]) <= by and (k["cite_nos"] or k["method"])  # the source is either marked or explicitly 'method'
    assert all(c["ref"] and c["page"] and c["version"] for c in rep["cited"])
    names = " ".join(b["title"] + b["short"] for b in rep["bibliography"]).lower()
    assert "matrix" not in names and "toolkit" not in names and "compliance check" not in names
    groups = {c["group"] for c in rep["controls"]}
    assert groups <= {"Beyanla ele alındı", "Gerekçeli kapsam dışı", "Bilinmiyor", "Henüz değerlendirilmedi"}
    pdf = rp.pdf_bytes(rep)
    import io, pypdf
    pages = [p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf)).pages]
    assert "Sonucu belirleyen bulgular" in pages[0] and "C01–C21 kapsam haritası" not in pages[0] and "Kullanıcı cevapları" not in pages[0]
    assert any("Yöntem ve ayrıntılar" in p for p in pages[1:]) and "Cevap A" not in pages[0] and "policy" not in pages[0].lower()


def groups():
    return C.interview_cfg()["prohibition_groups"]


def test_no_vague_followups_in_catalog():
    for tp in C.interview_cfg()["topics"]:
        assert not iv.is_vague(tp["default_q"]) or tp["id"] == "prohibited_detail", tp["id"]
    for g in groups().values():
        assert not iv.is_vague(g["detail_q"]) and g["detail_options"] and g["requires"]
    assert iv.is_vague("Az önceki cevabınızla ilgili biraz daha ayrıntı verebilir misiniz?") and iv.is_vague("Biraz açar mısınız?")


def test_options_are_never_truncated():
    src = open("app.py").read()
    assert "text-overflow: clip" in src and "white-space: normal" in src and "st.columns(min(len(opts)" not in src
    for tp in C.interview_cfg()["topics"]:
        for o in tp.get("options", []):
            assert "…" not in o and "..." not in o and iv.humanize(o) == o
    for g in groups().values():
        assert all("…" not in o for o in g["detail_options"])


def test_gate_no_skips_detail_branches(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "eu_link": "yes", **GATES_NO}}]))
    d = C.derive(s)
    assert d["slots"]["p_facial_scraping"] == "none" and d["slots_src"]["p_facial_scraping"] == "A1" and d["derived"]["p_rbi_law_enforcement"]
    assert d["slots"]["p_manipulation"] == "none"
    assert not [c for c in P.candidates(s) if c["kind"] == "followup"]
    s["turns"][0]["updates"]["slots"]["g_biometric"] = "yes"  # precondition 'yes': a single neutral detail question is asked
    fu = [c for c in P.candidates(s) if c["kind"] == "followup"]
    assert len(fu) == 1 and fu[0]["fact"] == "detail:biometric" and "amaçla" in fu[0]["text"]


# ---------------------------------------------------------------- source scenario ≠ user fact (hiring + another domain)
HIRING = "İşe alım agentı; adaylara uygunluk puanı veriyor ve işe alım kararını destekliyor."
CREDIT = "Kredi başvurularına risk puanı veren bir agent; kararı analist veriyor."


def first_turn(index, text, slots, **kw):
    s = start(index)
    iv.process(s, 0, text, index, FakeClient([{"slots": slots, **kw}]))
    return s


def test_scoring_does_not_open_social_scoring_or_crime(index):
    """'Scoring' alone is not social scoring/crime prediction; the detail branch does not open until the preconditions come from user declarations."""
    for text, area, ev in ((HIRING, "employment", "adaylara uygunluk puanı veriyor"), (CREDIT, "essential_services", "risk puanı veren")):
        s = first_turn(index, text, {"purpose": text, "g_person_eval": "yes", "annex3_area": area}, evidence={"g_person_eval": ev, "annex3_area": text.split(";")[0]})
        d = C.derive(s)
        assert d["slots"]["g_person_eval"] == "yes" and d["slots"].get("annex3_area") == area  # the declaration was recorded; the area is not asked again
        assert d["slots"].get("p_social_scoring") is None and d["slots"].get("p_crime_profiling") is None  # not assumed
        assert not C.detail_groups(d["slots"]) and not [c for c in P.candidates(s) if c["kind"] == "followup"]
        assert s["turns"][-1]["question"]["topic"] != "annex3"
        cand = {c["topic"]["id"]: c for c in P.candidates(s)}
        assert "eval_criteria" in cand and cand["eval_criteria"]["rank"][0] == 2  # missing precondition: a required (blocking) neutral criteria question
    q = C.TOPICS()["eval_criteria"]["default_q"]
    assert "ölçüt" in q and "sosyal" not in q.lower() and "suç" not in q.lower()


def test_influencing_a_decision_is_not_behaviour_manipulation(index):
    # a decision-support statement is not evidence for 'no' (no negation marker); it is recorded if explicitly denied
    s0 = first_turn(index, HIRING, {"purpose": "p", "g_influence": "no"}, evidence={"g_influence": "işe alım kararını destekliyor"})
    assert C.derive(s0)["slots"].get("g_influence") in (None, "unknown")
    s = first_turn(index, "Kararı destekliyor; kullanıcıyı davranışa yönlendirmiyor, hayır.", {"purpose": "p", "g_influence": "no"}, evidence={"g_influence": "yönlendirmiyor"})
    assert C.derive(s)["slots"]["g_influence"] == "no" and C.derive(s)["slots"]["p_manipulation"] == "none"
    assert "Karar vericiye yalnızca öneri sunması buna girmez" in C.TOPICS()["g_influence"]["default_q"]
    # if the model still writes yes: the prohibition sub-field cannot be filled from this answer; the detail question asked is neutral
    s2 = first_turn(index, HIRING, {"purpose": "p", "g_influence": "yes", "p_manipulation": "possible"}, evidence={"g_influence": "işe alım kararını destekliyor"})
    assert C.derive(s2)["slots"].get("p_manipulation") is None
    q = s2["turns"][-1]["question"]
    assert q["kind"] == "followup" and q["fact"] == "detail:influence" and q["options"][0].startswith("Açık")


def test_evidence_must_come_from_user_text(index):
    s = first_turn(index, HIRING, {"purpose": "p", "g_person_eval": "yes", "eval_uses_social_traits": "yes"},
                   evidence={"g_person_eval": "adaylara uygunluk puanı veriyor", "eval_uses_social_traits": "sosyal davranışa göre puanlıyor"})
    d = C.derive(s)
    assert d["slots"]["g_person_eval"] == "yes"
    assert d["slots"].get("eval_uses_social_traits") in (None, "unknown")  # the quote is not in the user's text → not recorded
    assert s["turns"][0]["updates"]["rejected"] and not C.detail_groups(d["slots"])
    assert iv.evidence_ok("adaylara uygunluk puanı veriyor", HIRING) and not iv.evidence_ok("sosyal medya verisi", HIRING)


def _ask_criteria(index, text="İşe alım agentı; adaylara puan veriyor."):
    s = first_turn(index, text, {"purpose": "p", "g_person_eval": "yes"}, evidence={"g_person_eval": "adaylara puan veriyor"})
    s["turns"][-1]["question"] = iv.make_question(C.TOPICS()["eval_criteria"], index, s)
    return s


def test_criteria_answers_drive_preconditions(index):
    s = _ask_criteria(index)  # 1) only work-related criteria → the sub-prohibition branches are skipped, the reason is recorded
    iv.process(s, len(s["turns"]) - 1, "Yalnızca deneyim ve test sonucu; sosyal veri ve suç riski yok", index,
               FakeClient([{"slots": {"eval_uses_social_traits": "no", "eval_predicts_crime": "no"}}]))
    d = C.derive(s)
    assert d["slots"]["p_social_scoring"] == "none" and d["slots"]["p_crime_profiling"] == "none" and d["derived"]["p_social_scoring"]
    assert not [c for c in P.candidates(s) if c["kind"] == "followup"]
    s2 = _ask_criteria(index)  # 2) social/personality data yes → only that branch, a single neutral detail question, with the Art. 5(1)(c) passage
    iv.process(s2, len(s2["turns"]) - 1, "Sosyal medya davranışı da kullanılıyor, suç tahmini yok", index,
               FakeClient([{"slots": {"eval_uses_social_traits": "yes", "eval_predicts_crime": "no"}}]))
    q = s2["turns"][-1]["question"]
    assert q["kind"] == "followup" and q["fact"] == "detail:social" and "bağlam" in q["text"] and q["passage_ids"]
    assert all(index.by_id[p]["ref"].startswith("Art. 5(1)(c)") for p in q["passage_ids"])
    assert C.derive(s2)["slots"]["p_crime_profiling"] == "none"  # the crime branch was not opened


def test_dont_know_in_criteria_does_not_confirm_or_clear(index):
    s = _ask_criteria(index)
    iv.process(s, len(s["turns"]) - 1, "bilmiyorum", index, FakeClient([{"slots": {}}]))
    q = s["turns"][-1]["question"]
    assert q["kind"] == "followup" and q["fact"] == "eval_criteria"  # one simple follow-up
    iv.process(s, len(s["turns"]) - 1, "hâlâ bilmiyorum", index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert d["slots"]["eval_uses_social_traits"] == "unknown" and d["slots"].get("p_social_scoring") is None  # neither 'yes' nor 'no'
    assert not [c for c in P.candidates(s) if c["fact"] == "eval_criteria"]
    cls = C.classify(s)
    assert cls["risk_label"] is None and any("yasak" in u.lower() for u in cls["uncertainties"])


def _to_followup(index, group, slots):
    s = first_turn(index, "x", {"purpose": "p", **slots})
    assert s["turns"][-1]["question"]["fact"] == "detail:" + group, s["turns"][-1]["question"]
    return s


def test_forbidden_requires_basis_and_preconditions(index):
    b = [{"key": "p_social_scoring", "conditions_confirmed": True, "exception_considered": True, "scope_date_considered": True}]
    pre = {"g_person_eval": "yes", "eval_uses_social_traits": "yes"}
    s = _to_followup(index, "social", pre)
    iv.process(s, len(s["turns"]) - 1, "Evet, farklı bağlamda kullanılıyor", index, FakeClient([{"slots": {"p_social_scoring": "supported"}}]))  # no basis
    assert s["phase"] != "forbidden" and C.derive(s)["slots"]["p_social_scoring"] == "possible"
    s2 = _to_followup(index, "social", pre)
    iv.process(s2, len(s2["turns"]) - 1, "Evet, farklı bağlamda kullanılıyor", index, FakeClient([{"slots": {"p_social_scoring": "supported"}, "basis": b}]))
    assert s2["phase"] == "forbidden" and C.classify(s2)["risk_label"] == "FORBIDDEN"
    rep = rp.build_report(s2, index)
    assert rep["verification"]["all_ok"] and rp.pdf_bytes(rep)[:4] == b"%PDF"
    s3 = {"settings": {}, "turns": [{"id": "A1", "question": {"topic": "purpose", "kind": "question"}, "answer": "x",
          "updates": {"slots": {"purpose": "p", "g_person_eval": "no", "p_social_scoring": "supported"}, "controls": {}, "basis": b}}]}
    assert C.classify(s3)["risk_label"] != "FORBIDDEN"  # 'supported' without a precondition does not yield FORBIDDEN


def test_forbidden_with_undetermined_applicability_but_not_when_declared_out(index):
    b = [{"key": "p_social_scoring", "conditions_confirmed": True, "exception_considered": True, "scope_date_considered": True}]
    pre = {"g_person_eval": "yes", "eval_uses_social_traits": "yes"}
    s = _to_followup(index, "social", pre)
    iv.process(s, len(s["turns"]) - 1, "Evet, farklı bağlamda", index, FakeClient([{"slots": {"p_social_scoring": "supported"}, "basis": b}]))
    assert s["phase"] == "forbidden" and C.classify(s)["applicability"] == "unknown"
    s2 = iv.new_state({"eu_link": "no"}); iv.add_first(s2, index)
    iv.process(s2, 0, "x", index, FakeClient([{"slots": {"purpose": "p", **pre}}]))
    iv.process(s2, len(s2["turns"]) - 1, "evet", index, FakeClient([{"slots": {"p_social_scoring": "supported"}, "basis": b}]))
    assert s2["phase"] != "forbidden"


# ---------------------------------------------------------------- new interview flow
def test_first_question_and_not_vague():
    assert C.TOPICS()["purpose"]["default_q"] == "Değerlendireceğimiz agentın yaptığı işi kısaca anlatır mısınız?"
    assert not iv.is_vague(C.TOPICS()["purpose"]["default_q"]) and not iv.is_vague("Puanın hangi ölçütlere göre hesaplandığını anlatır mısınız?")
    assert iv.is_vague("Anlatır mısınız?") and iv.is_vague("Biraz ayrıntı verir misiniz?")


def test_context_mention_is_not_two_questions():
    top = C.TOPICS()["g_biometric"]
    assert not iv.mixes_topics("Adayların puanlandığını söylediniz. Agent kişileri tanımak için yüz veya ses verisi kullanıyor mu?", top)  # a context sentence is allowed
    assert iv.mixes_topics("Agent biyometrik veri kullanıyor mu? İşe alım gibi hassas bir alanda mı çalışıyor?", top)
    assert iv.mixes_topics("Agent biyometrik veri kullanıyor mu, ve işe alım gibi hassas bir alanda mı çalışıyor?", top)


def test_options_keep_all_and_record_value(index):
    q = iv.make_question(C.TOPICS()["annex3"], index, {"turns": []})
    assert "Bilmiyorum" in q["options"] and "Başka bir alan" in q["options"] and len(q["options"]) == 6  # no truncation
    assert q["option_sets"]["İşe alım / çalışan yönetimi"] == {"annex3_area": "employment"}
    s = start(index)
    s["turns"][0]["question"] = iv.make_question(C.TOPICS()["g_biometric"], index, s)
    iv.process(s, 0, "Hayır", index, FakeClient([{"slots": {}}]))  # the model extracted nothing; the option itself carries the field value
    d = C.derive(s)
    assert d["slots"]["g_biometric"] == "no" and s["turns"][0]["updates"]["choice"] == {"label": "Hayır", "set": {"g_biometric": "no"}, "controls": {}}
    assert d["slots"]["p_rbi_law_enforcement"] == "none"


def test_open_questions_have_no_forced_options(index):
    assert iv.first_question(index)["options"] == []
    q = iv.make_question(C.TOPICS()["eval_criteria"], index, {"turns": []})
    assert q["options"] == [] and "ölçüt" in q["text"]


def test_approval_exists_before_approval_bound():
    bound = C.TOPICS()["approval_bound"]
    assert not C.topic_active(bound, {"action_mode": "unknown"})  # 'is the approval action-specific?' is not asked before it is known whether an approval exists
    assert not C.topic_active(bound, {"action_mode": "unknown", "approval_exists": "no"})
    assert not C.topic_active(bound, {"action_mode": "unknown", "approval_exists": "unknown"})
    assert C.topic_active(bound, {"action_mode": "unknown", "approval_exists": "yes"})
    s = {"settings": {}, "turns": [{"id": "A1", "question": {"topic": "purpose", "kind": "question"}, "answer": "x",
         "updates": {"slots": {"purpose": "p", "action_mode": "approved_action"}, "controls": {}, "basis": []}}]}
    d = C.derive(s)
    assert d["slots"]["approval_exists"] == "yes" and C.topic_active(bound, d["slots"])  # an equivalent fact (acts after approval) closes the approval-exists fact
    assert "approval_exists" not in [c["topic"]["id"] for c in P.candidates(s)]
    assert "approval_bound" in [c["topic"]["id"] for c in P.candidates(s)]  # asked because it is needed for the result (L3/L4)


def test_stop_and_rollback_are_separate_facts():
    assert "human_intervention" not in C.interview_cfg()["slots"]
    a, b = C.TOPICS()["stop_ability"], C.TOPICS()["rollback_ability"]
    assert a["slots"] == ["can_stop"] and b["slots"] == ["can_rollback"] and "geri" not in a["default_q"] and "durdur" not in b["default_q"]


def test_valid_model_choice_is_not_replaced_by_priority(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p"}, "topic": "annex3", "question": "Agent hangi alanda çalışıyor?"}]))
    pre = {**s, "turns": [dict(t, answer=None) for t in s["turns"]]}
    cand_ids = [c["topic"]["id"] for c in P.candidates(pre) if c["kind"] == "question"][:iv.N_CAND]
    assert "annex3" in cand_ids
    q = s["turns"][-1]["question"]
    assert q["topic"] == "annex3" and q["text"] == "Agent hangi alanda çalışıyor?"  # a valid proposal was kept even though the fixed order differs


def test_invalid_model_choice_uses_neutral_catalog_question_of_a_seen_topic(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "annex3_area": "none"}, "topic": "annex3", "question": "Bu alanı soralım mı acaba?"}]))
    q = s["turns"][-1]["question"]
    assert q["topic"] != "annex3"  # a topic resolved by this answer is invalid
    assert not iv.is_vague(q["text"]) and q["text"] == iv.humanize(C.TOPICS()[q["topic"]]["default_q"]) or q["kind"] != "question"


def test_ui_has_no_usage_counters_and_no_old_limits():
    import re
    src = open("app.py").read()
    for bad in ("usage_caption", "usage_summary", "setup_usage", "Netleşenler", "planner.progress", "(25)"):
        assert bad not in src, bad
    assert not re.search(r"\b25\b", open("README.md").read()) and "25 soru" not in src
    assert not re.search(r"h1,h2,h3,p,li,label,span,div", src)  # no global color override
    assert "primaryColor" in open(".streamlit/config.toml").read()


def test_semantic_check_is_off_by_default(index):
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    fc = FakeClient([])
    rep = rp.build_report(s, index, fc)  # no extra call even if a client is passed
    assert fc.calls == 0 and rep["verification"]["semantic_done"] is False


# ---------------------------------------------------------------- data consistency: explicit negative declaration, control declaration, cross-field conflict
def _turn(i, topic, slots, ans="x", kind="question", ctl=None):
    return {"id": i, "question": {"topic": topic, "kind": kind}, "answer": ans, "updates": {"slots": slots, "controls": ctl or {}, "basis": []}}


def test_explicit_rejection_survives_paraphrased_evidence(index):
    s = _ask_criteria(index)
    ans = "Hayır, sosyal medya veya kişilik analizi yok; suç tahmini de yok."
    iv.process(s, len(s["turns"]) - 1, ans, index, FakeClient([{"slots": {"eval_uses_social_traits": "no", "eval_predicts_crime": "no"},
               "evidence": {"eval_uses_social_traits": "kişilik analizi yapılmıyor", "eval_predicts_crime": "suç tahmini yapılmıyor"}}]))
    d = C.derive(s)
    assert d["slots"]["eval_uses_social_traits"] == "no" and d["slots"]["eval_predicts_crime"] == "no"  # did not fall to unknown
    assert d["slots"]["p_social_scoring"] == "none" and d["slots"]["p_crime_profiling"] == "none"
    assert not [c for c in P.candidates(s) if c["fact"] == "eval_criteria"]
    # safety: a positive value stays strict; a negative value requires a negation marker in the answer
    assert not iv.evidence_ok("sosyal medya verisi", "Deneyim ve test sonucu", "yes")
    assert not iv.evidence_ok("kişilik analizi yapılmıyor", "Kişilik analizi yapılıyor", "no") or iv.evidence_ok("kişilik analizi yapılmıyor", "Kişilik analizi yapılıyor", "no") is False
    assert iv.evidence_ok("kişilik analizi yapılmıyor", ans, "no")
    assert C.derive(first_turn(index, "Kişilik analizi yapılıyor", {"purpose": "p", "eval_uses_social_traits": "no"}, evidence={"eval_uses_social_traits": "kişilik analizi yapılmıyor"}))["slots"].get("eval_uses_social_traits") in (None, "unknown")


def test_compound_followup_answer_maps_to_every_slot(index):
    def followup(index):
        s = _ask_criteria(index)
        iv.process(s, len(s["turns"]) - 1, "bilmiyorum", index, FakeClient([{"slots": {}}]))
        assert s["turns"][-1]["question"]["kind"] == "followup"
        return s
    s = followup(index)
    q = s["turns"][-1]["question"]
    assert q["option_sets"]["Hayır, ikisi de yok"] == {"eval_uses_social_traits": "no", "eval_predicts_crime": "no"}
    iv.process(s, len(s["turns"]) - 1, "Hayır, ikisi de yok", index, FakeClient([{"slots": {"eval_predicts_crime": "no"}}]))  # even if the model returns only one of them
    d = C.derive(s)
    assert d["slots"]["eval_uses_social_traits"] == "no" and d["slots"]["eval_predicts_crime"] == "no" and d["slots"]["p_social_scoring"] == "none"
    s2 = followup(index)
    iv.process(s2, len(s2["turns"]) - 1, "Evet, sosyal davranış veya kişilik verisi var", index, FakeClient([{"slots": {}}]))
    d2 = C.derive(s2)
    assert d2["slots"]["eval_uses_social_traits"] == "yes" and d2["slots"]["eval_predicts_crime"] == "unknown"  # the field that was not rejected was not counted as 'no'


def test_declaration_answers_are_not_turned_into_unknown(index):
    s = start(index)
    s["turns"][0]["question"] = iv.make_question(C.TOPICS()["ctl_conformity"], index, s)
    assert s["turns"][0]["question"]["control_sets"]["Hayır"]["C20"]["status"] == "declared_absent"
    iv.process(s, 0, "Uygunluk değerlendirmesi yapılmadı.", index, FakeClient([{"slots": {}}]))  # the model did not return a control status
    assert C.derive(s)["controls"]["C20"]["status"] == "declared_absent"
    s2 = start(index)
    s2["turns"][0]["question"] = iv.make_question(C.TOPICS()["ctl_conformity"], index, s2)
    iv.process(s2, 0, "Hayır", index, FakeClient([{"slots": {}}]))
    assert C.derive(s2)["controls"]["C20"]["status"] == "declared_absent"
    for txt in ("Yapılmadı ama önümüzdeki ay planlıyoruz", "Bilmiyorum"):  # a conflicting/uncertain answer is not counted as 'absent'
        s3 = start(index)
        s3["turns"][0]["question"] = iv.make_question(C.TOPICS()["ctl_conformity"], index, s3)
        iv.process(s3, 0, txt, index, FakeClient([{"slots": {}}]))
        assert C.derive(s3)["controls"]["C20"]["status"] == "unknown"
    st = {"settings": {}, "turns": [_turn("A1", "ctl_conformity", {}, ctl={"C20": {"status": "declared_absent", "note": ""}}),
                                    _turn("A2", "ctl_conformity", {}, ctl={"C20": {"status": "unknown", "note": ""}})]}
    assert C.derive(st)["controls"]["C20"]["status"] == "declared_absent"  # a later 'unknown' does not overwrite it


BASE = {"purpose": "p", "annex3_area": "none", "max_harm": "limited", **GATES_NO}


def _conflict_state():
    return {"settings": {}, "turns": [_turn("A1", "purpose", {**BASE, "action_mode": "envelope_action"}, "Agent adayları onaysız eliyor"),
                                      _turn("A2", "approval_bound", {"approval_bound": "yes"}, "Evet, işleme özel onay")]}


def test_cross_slot_approval_conflict_blocks_certain_autonomy(index):
    s = _conflict_state()
    d = C.derive(s)
    assert "approval_bound" in d["conflicts"] and C.autonomy_conflict(d)
    cls = C.classify(s)
    assert cls["L"] is None and cls["risk_label"] is None and "action_mode" in cls["unresolved"]  # no certain L/label
    assert any("çelişiyor" in u for u in cls["uncertainties"])
    top = P.candidates(s)[0]
    assert top["kind"] == "conflict" and top["fact"] == "approval_bound"
    q = iv.make_question(top["topic"], index, s, kind="conflict", cand=top)
    assert "hangi işlem" in q["text"] and "Tüm işlemler işleme özel onayla yapılıyor" in q["options"] and q["no_passage"]
    # resolution: all operations with action-specific approval → the conflict ends, L3
    s["turns"].append({"id": "A3", "question": q, "answer": "Tüm işlemler işleme özel onayla yapılıyor", "updates": {}})
    iv.process(s, 2, "Tüm işlemler işleme özel onayla yapılıyor", index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert not d["conflicts"] and not d["unresolved_conflicts"] and d["slots"]["action_mode"] == "approved_action" and C.classify(s)["L"] == "L3"


def test_unresolved_approval_conflict_stays_uncertain_and_is_asked_once(index):
    for label in ("Bazı işlemler onaylı, bazıları onaysız", "Bilmiyorum"):
        s = _conflict_state()
        top = P.candidates(s)[0]
        q = iv.make_question(top["topic"], index, s, kind="conflict", cand=top)
        s["turns"].append({"id": "A3", "question": q, "answer": label, "updates": {}})
        iv.process(s, 2, label, index, FakeClient([{"slots": {}}]))
        d = C.derive(s)
        assert "approval_bound" in d["unresolved_conflicts"] and d["slots"]["approval_bound"] == "unknown"
        cls = C.classify(s)
        assert cls["L"] is None and cls["risk_label"] is None  # unresolved: no certain approval/autonomy finding and no label
        assert not [c for c in P.candidates(s) if c["kind"] == "conflict"]  # not asked a second time


def test_approval_exists_no_vs_bound_yes_is_a_conflict_and_thresholds_unchanged():
    s = {"settings": {}, "turns": [_turn("A1", "purpose", {**BASE, "action_mode": "unknown", "approval_exists": "no"}), _turn("A2", "approval_bound", {"approval_bound": "yes"})]}
    assert "approval_bound" in C.derive(s)["conflicts"]
    ok = {"settings": {}, "turns": [_turn("A1", "purpose", {**BASE, "action_mode": "approved_action", "approval_bound": "yes", "limits_enforced": "technical"})]}
    c = C.classify(ok)
    assert c["L"] == "L3" and c["I"] == "I1" and c["risk_label"] == "MEDIUM"  # thresholds did not change
    hi = {"settings": {}, "turns": [_turn("A1", "purpose", {**BASE, "max_harm": "critical", "action_mode": "envelope_action"}), _turn("A2", "approval_bound", {"approval_bound": "yes"})]}
    assert C.classify(hi)["risk_label"] == "HIGH"  # I3 is already HIGH: the conflict neither lowers nor fixes the label, only L is uncertain
