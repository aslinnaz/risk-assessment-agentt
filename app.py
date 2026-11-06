"""TrustAI Risk Assessment — Streamlit interface (preliminary assessment based on human declarations)."""
import html

import streamlit as st

import classify as C
import config
import interview as iv
import llm
import rag
import report as rp

st.set_page_config(page_title="TrustAI Risk Assessment", page_icon="🛡️", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Colors come from the .streamlit/config.toml theme (navy, white, a single teal accent). Only narrowly scoped component styles live here. */
.block-container { max-width: 760px; padding-top: 2.2rem; }
.ta-steps { display:flex; gap:6px; margin:4px 0 18px 0; }
.ta-step { flex:1; text-align:center; padding:6px 4px; border-radius:8px; font-size:.9rem; border:1px solid #CBD5E1; }
.ta-step.on { background:#0F9D94; color:#fff; border-color:#0F9D94; font-weight:600; }
.ta-step.done { border-color:#0F9D94; }
.ta-card { border:1px solid #CBD5E1; border-left:6px solid #0F9D94; border-radius:10px; padding:14px 16px; margin:10px 0 14px 0; background:#fff; }
.ta-card.forbidden { border-left-color:#0B1F3A; border-left-width:10px; }
.ta-muted { opacity:.75; }
/* Option buttons: full text, line wrapping, height fitted to content (Streamlit's default is nowrap + ellipsis) */
.stButton > button, .stButton > button *, .stFormSubmitButton > button, .stFormSubmitButton > button * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; word-break: break-word; }
.stButton > button { height:auto !important; min-height:2.5rem; text-align:left; justify-content:flex-start; padding:.5rem .9rem; line-height:1.35; }
.stButton > button p { text-align:left; margin:0; }
</style>
""", unsafe_allow_html=True)

PHASES = ["Kapsam", "Kullanım", "Yetki ve etki", "Sonuç"]
LABEL_TR = {"LOW": "DÜŞÜK (LOW)", "MEDIUM": "ORTA (MEDIUM)", "HIGH": "YÜKSEK (HIGH)", "FORBIDDEN": "YASAK KULLANIM BULGUSU (FORBIDDEN)"}


@st.cache_resource(show_spinner="Kaynak indeksi yükleniyor…")
def index():
    return rag.get_index()


@st.cache_resource
def _client(key: str):
    return llm.get_client()


def client_or_none():
    try:
        key = config.api_key()
        if not key:
            raise llm.SetupError("OPENAI_API_KEY tanımlı değil. .env dosyasına anahtarı yazın (bkz. .env.example) ve uygulamayı yeniden başlatın.")
        return _client(key), None
    except llm.SetupError as e:
        return None, str(e)


def S():
    if "state" not in st.session_state:
        st.session_state.state = None
    if "pending" not in st.session_state:
        st.session_state.pending = None
    return st.session_state


def queue(kind, idx=None, key=None, text=""):
    if key:
        text = st.session_state.get(key, "")
    if text.strip() or kind in ("confirm", "reset", "retry"):
        st.session_state.pending = {"kind": kind, "idx": idx, "text": text}


def progress(state):
    cur = 0
    if state:
        if state["phase"] in ("summary", "forbidden") or state.get("report"):
            cur = 3
        else:
            last = state["turns"][-1]["question"]
            cur = PHASES.index(last.get("phase")) if last.get("phase") in PHASES else 0
    cells = "".join(f'<div class="ta-step {"on" if i == cur else "done" if i < cur else ""}">{"✓ " if i < cur else ""}{p}</div>' for i, p in enumerate(PHASES))
    st.markdown(f'<div class="ta-steps">{cells}</div>', unsafe_allow_html=True)


def why_box(idx, q, key):
    """Saved short rationale + source; closed by default, opening it makes no API call."""
    with st.expander("Neden bu soru?"):
        w = q.get("why")
        reason = (w.get("clarifies") or w.get("missing")) if isinstance(w, dict) else w
        if reason:
            st.write(reason)
        if q.get("framing") == "general":
            st.write("AB bağlantısı olmadığı beyan edildi; bu soru AI Act yükümlülüğü değil, genel TrustAI değerlendirmesi kapsamındadır.")
        if q.get("kind") == "conflict":
            st.write("Bu soru kaynağa değil, önceki cevaplarınızın tutarlılığına dayanır.")
        elif q.get("no_passage"):
            st.write("Bu soru için ilgili kaynak pasajı bulunamadı; bu bir belirsizlik olarak kayıtlıdır.")
        if q.get("adapted") and q.get("passage_ids"):
            st.write("Bu ilkeden uyarlanmış bir soru; kaynak soruyu birebir içermez ve yasal zorunluluk olarak sunulmaz.")
        for n, pid in enumerate(q.get("passage_ids", [])):
            c = idx.by_id.get(pid)
            if not c:
                continue
            m = rag.source_meta(c["source_id"])
            sec = f" · {c['section']}" if c.get("section") and c["section"] not in c["ref"] else ""
            st.markdown(f"**Kaynak:** {m['short']} · {c['ref']}{sec} · PDF s. {c['page']}" + (f" · [bağlantı]({c['url']})" if c.get("url") else ""))
            if st.toggle("Pasajı göster", key=f"tg_{key}_{n}"):
                st.caption("“" + c["text"] + "”")


def render_question(state, idx):
    """Current question only: text, options (optional), answer field and rationale link."""
    i = len(state["turns"]) - 1
    t = state["turns"][i]
    if state["phase"] != "interview" or t.get("answer") is not None:
        return
    q = t["question"]
    st.markdown(f"#### {q['text']}")
    why_box(idx, q, t["id"])
    for j, o in enumerate(q.get("options") or []):  # stacked one per row, full text; the selected label is stored together with its field/value
        st.button(o, key=f"op_{t['id']}_{j}", on_click=queue, args=("answer", i, None, o), use_container_width=True)
    with st.form(f"f_{t['id']}", clear_on_submit=True):
        st.text_area("Cevabınız", height=100, key=f"in_{t['id']}", placeholder="Kendi sözlerinizle yazabilirsiniz; “bilmiyorum” da geçerli bir cevaptır.")
        st.form_submit_button("Gönder", type="primary", on_click=queue, args=("answer", i, f"in_{t['id']}"))


def render_history(state):
    """Closed 'Interview history': previous questions/answers and answer editing (no nested expanders)."""
    done = [(i, t) for i, t in enumerate(state["turns"]) if t.get("answer") is not None]
    if not done:
        return
    with st.expander("Görüşme geçmişi", expanded=False):
        for i, t in done:
            if t["question"].get("kind") == "correction":
                st.markdown(f"*Düzeltme:* {t['answer']}")
                continue
            st.markdown(f"**{t['question']['text']}**")
            st.write(t["answer"])
            if st.toggle("Bu cevabı düzenle", key=f"edt_{t['id']}"):
                st.write("Düzeltirseniz bu cevaptan sonraki sorular ve sonuçlar geçersiz sayılır; soru hakkınız artmaz.")
                st.text_area("Yeni cevap", value=t["answer"], key=f"ed_{t['id']}")
                st.button("Kaydet", key=f"sv_{t['id']}", on_click=queue, args=("edit", i, f"ed_{t['id']}"))
            st.divider()


def render_summary(state):
    cls = C.classify(state)
    st.subheader("Sizi şöyle anladım")
    for title, text, _ in iv.summary_lines(state):
        st.markdown(f"**{title}:** {text}")
    if cls["unresolved"] and C.answered_count(state) >= config.MAX_QUESTIONS:
        st.warning(f"Soru sınırına ({config.MAX_QUESTIONS}) ulaşıldı. Eksik alanlar: " + ", ".join(C.policy()["critical_labels_tr"].get(k, k) for k in cls["unresolved"]) + ". Yeni soru döngüsü açılmaz; gerekirse aşağıdan düzeltme yazabilirsiniz.")
    st.write("Yanlış anlaşılan bir şey varsa aşağıya yazın; bu düzeltme soru sayınızı artırmaz.")
    with st.form("corr", clear_on_submit=True):
        st.text_area("Düzeltme (isteğe bağlı)", height=80, key="corr_in")
        st.form_submit_button("Düzeltmeyi uygula", on_click=queue, args=("correction", None, "corr_in"))
    st.button("Doğru, sonucu göster", type="primary", on_click=queue, args=("confirm",))


def source_details(rep, nos, key):
    """Short source markers; when expanded: document, article/section, page, version and link (no empty labels)."""
    by = {c["no"]: c for c in rep["cited"]}
    nos = [n for n in nos if n in by]
    if not nos:
        return
    with st.expander("Kaynak: " + " ".join(f"[K{n}]" for n in nos), expanded=False):
        for n in nos:
            c = by[n]
            sec = f" · {c['section']}" if c["section"] and c["section"] not in c["ref"] else ""
            st.markdown(f"**[K{n}] {c['source']}** · {c['ref']}{sec} · PDF s. {c['page']} · {c['version']}" + (f" · [bağlantı]({c['url']})" if c["url"] else ""))
            st.caption("“" + c["snippet"] + "…”")


def render_result(state, idx):
    rep = state["report"]
    lab = rep["risk_label"]
    title = LABEL_TR[lab] if lab else "Değerlendirme tamamlanamadı"
    ctx = rep["context"]
    sub = rep["label_note"] or ("Beşinci bir risk seviyesi değildir; kritik bilgiler eksik olduğu için etiket verilmedi." if not lab else "TrustAI ön risk etiketi; AI Act'in resmî kategorisi değildir.")
    st.markdown(f'<div class="ta-card {"forbidden" if lab == "FORBIDDEN" else ""}"><b>Risk seviyesi: {html.escape(title)}</b> &nbsp;·&nbsp; <b>Durum: {html.escape(rep["status_text"])}</b>'
                f'<br><span class="ta-muted">{html.escape(sub)}</span></div>', unsafe_allow_html=True)
    st.write(f"Agent faaliyet kapsamı: {ctx['scope_country']} · Hukuki uygulanabilirlik: {ctx['applicability_tr']} · Referans çerçeve: {ctx['reference']}")
    if lab == "FORBIDDEN":
        st.info("Değerlendirme görüşmesi durduruldu; bu hedef agentın teknik olarak durdurulduğu anlamına gelmez.")
    st.subheader("Değerlendirilen kullanım")
    st.write(rep["use_text"])
    st.subheader("Sonucu belirleyen bulgular")
    for k in rep["key_findings"]:
        st.markdown(f"**•** {k['text']}")
        if k["decl"]:
            st.write(k["decl"])  # the user's statement
        source_details(rep, k["cite_nos"], k["id"])  # source used in the assessment (shown separately)
        if k["method"] and not k["cite_nos"]:
            st.write("Yöntem: TrustAI proje karar tablosu (dış kaynak değildir).")
    if rep["improvements"]:
        st.subheader("Gerekli iyileştirmeler")
        for im in rep["improvements"]:
            st.markdown(f"**•** {im['text']}")
            source_details(rep, im["cite_nos"], im["id"])
    if rep["critical_gaps"]:
        st.subheader("Sonucu etkileyen kritik eksikler")
        for g in rep["critical_gaps"]:
            st.markdown(f"**•** {g}")
    st.write(rep["note"])
    st.download_button("PDF raporunu indir", rp.pdf_bytes(rep), file_name="trustai_risk_assessment.pdf", mime="application/pdf", type="primary")
    with st.expander("Yöntem ve ayrıntılar"):
        st.markdown("**Kapsam ve çerçeve.** Referans çerçeve ile hukuki uygulanabilirlik ayrı alanlardır. " + rep["disclaimer"])
        if rep["legal_scope_note"]:
            st.write(rep["legal_scope_note"])
        for a in rep["dates"]:
            st.write(f"{a['what_tr']} — {a['date']} ({a['ref']}). {a['status_tr']}")
        st.write(rep["currency"])
        cl = rep["cls"]
        if cl["L"] and cl["I"]:
            st.markdown(f"**Operasyonel etki:** otonomi {cl['L']} · etki {cl['I']} · yardımcı güvence profili {cl['assurance']} — {C.policy()['assurance_postures_tr'][cl['assurance']]}. "
                        f"{C.policy()['assurance_note_tr']} Karar tablosu: {rep['policy_version']}.")
            for n in rep["level_notes"]:
                st.write("• " + n)
        st.markdown("**Bulgular, kullanıcı beyanı ve kaynak ilişkisi**")
        for f in rep["findings"]:
            ans = ", ".join(f"Cevap {a}" for a in f["answers"])
            nos = " ".join(f"[K{n}]" for n in f["cite_nos"])
            st.write(f"{f['id']}. {f['text']}" + (f" — Beyan: {ans}" if ans else "") + (f" — Kaynak: {nos}" if nos else ""))
        v = rep["verification"]
        st.write("Atıf kontrolü programatiktir (pasaj varlığı ve anahtar ifade); pasajın iddiayı anlam olarak desteklediğinin garantisi değildir."
                 + (f" İşaretlenenler: {', '.join(v['flagged'])}." if v["flagged"] else ""))
        st.markdown("**C01–C21 kapsam haritası** (beyan durumu; güvenlik skoru değildir)")
        st.write(", ".join(f"{k}: {n}" for k, n in rep["control_counts"].items()))
        for grp in ("Beyanla ele alındı", "Gerekçeli kapsam dışı", "Bilinmiyor", "Henüz değerlendirilmedi"):
            rows = [c for c in rep["controls"] if c["group"] == grp]
            if rows:
                st.markdown(f"*{grp}* — " + "; ".join(f"{c['id']} {c['title']}" + (f" ({c['note']})" if c["note"] and grp != "Henüz değerlendirilmedi" else "") for c in rows))
        if rep["noncritical_gaps"]:
            st.markdown("**Kritik olmayan eksikler**")
            for g in rep["noncritical_gaps"]:
                st.write("• " + g)
        st.markdown("**Kullanıcı cevapları**")
        for a in rep["answers"]:
            st.write(f"Cevap {a['id']} — {a['q']} → {a['a']}")
        st.markdown("**Kaynak pasajları**")
        for c in rep["cited"]:
            sec = f" · {c['section']}" if c["section"] and c["section"] not in c["ref"] else ""
            st.write(f"[K{c['no']}] {c['source']} ({c['version']}) · {c['ref']}{sec} · PDF s. {c['page']}" + (f" · {c['url']}" if c["url"] else ""))
        st.markdown("**Kaynak listesi ve sürümler**")
        for b in rep["bibliography"]:
            st.write(f"{b['title']} — {b['kind']}; sürüm: {b['version']}; tarih: {b['date']}" + (f"; {b['url']}" if b["url"] else "") + f". {b['note']}")


def run_pending(idx, client):
    p = st.session_state.pending
    st.session_state.pending = None  
    state = st.session_state.state
    try:
        if p["kind"] == "reset":
            st.session_state.state = None
        elif p["kind"] == "retry":
            with st.spinner("Tekrar deneniyor…"):
                iv.process(state, len(state["turns"]) - 1, state["turns"][-1]["answer"], idx, client)
        elif p["kind"] in ("answer", "edit"):
            with st.spinner("Cevabınız değerlendiriliyor…"):
                iv.process(state, p["idx"], p["text"], idx, client)
        elif p["kind"] == "correction":
            with st.spinner("Düzeltme işleniyor…"):
                iv.process(state, len(state["turns"]) - 1, p["text"], idx, client, mode="correction")
        elif p["kind"] == "confirm":
            state["summary_confirmed"] = True
        if state and ((state["phase"] == "forbidden") or (state["summary_confirmed"] and not state.get("report"))):
            with st.spinner("Rapor hazırlanıyor…"):
                state["report"] = rp.build_report(state, idx)  
    except llm.SetupError as e:
        if st.session_state.state:
            st.session_state.state["error"] = str(e)
    except Exception as e:  # noqa: BLE001
        if st.session_state.state:
            st.session_state.state["error"] = "İşlem tamamlanamadı: " + llm._clean(e) + " Cevabınız kayıtlı; tekrar deneyebilirsiniz."


def sidebar(idx, client, setup_msg, started):
    """Settings and setup live in the sidebar, separate from the interview area."""
    with st.sidebar:
        if not started:
            st.markdown("**Değerlendirme ayarları**")
            dflt = C.policy()["context_defaults"]
            st.caption("Değerlendirilen agentın faaliyet kapsamı (uygulamanın konumu değil). Türkiye varsayımı AB kapsamı dışı anlamına gelmez; AI Act değerlendirme referansıdır.")
            st.text_input("Agentın faaliyet kapsamı", value=dflt["scope_country"], key="cfg_country")
            st.radio("AB ile bağlantı", ["Belirlenmedi", "Var — AB'de kullanılıyor veya AB'deki kişileri/kuruluşları etkiliyor", "Yok"], key="cfg_eu")
            st.divider()
        st.markdown("**Kurulum**")
        idx.refresh_embeddings()  # loads a ready index from disk without any API call
        stt = idx.status()
        st.write(f"API anahtarı: {'bulundu' if not setup_msg else 'yok'}")
        st.write(f"Kaynak pasajları: {stt['chunks']}")
        if stt["embeddings"] and not stt["missing"]:
            st.write("Anlamsal arama indeksi: hazır")
        elif stt["embeddings"]:
            st.write(f"Anlamsal arama indeksi: hazır ({stt['missing']} pasaj anahtar kelimeyle aranır)")
            if st.button("Eksik pasajları ekle", disabled=client is None):
                _build_index(idx, client)
        else:
            st.write("Anlamsal arama indeksi: yok (arama anahtar kelimeyle çalışır)")
            if st.button("Embedding indeksini oluştur", disabled=client is None):
                _build_index(idx, client)


def _build_index(idx, client):
    with st.spinner("İndeks hazırlanıyor…"):
        try:
            idx.build_embeddings(client)
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error("İndeks hazırlanamadı: " + llm._clean(e))


def main():
    st.markdown("### TrustAI Risk Assessment")
    st.write("AI agent hakkında bilgisi olan biriyle kısa bir görüşme yapıp, beyanlarınıza dayalı bir risk ön değerlendirmesi hazırlar.")
    client, setup_msg = client_or_none()
    idx = index()
    state = S().state
    sidebar(idx, client, setup_msg, started=state is not None)
    progress(state)
    if state is None:
        if setup_msg:
            st.warning("Kurulum gerekli: " + setup_msg)
        if idx.status()["chunks"] == 0:
            st.warning("Kaynak PDF'leri bulunamadı. README'deki kaynak hazırlama adımını uygulayın.")
        if st.button("Değerlendirmeyi başlat", type="primary", disabled=bool(setup_msg) or idx.status()["chunks"] == 0):
            dflt = C.policy()["context_defaults"]
            eu_map = {"Belirlenmedi": "unknown", "Yok": "no"}
            s_ = iv.new_state({"scope_country": (st.session_state.get("cfg_country") or dflt["scope_country"]).strip() or dflt["scope_country"],
                               "eu_link": eu_map.get(st.session_state.get("cfg_eu", "Belirlenmedi"), "yes")})
            iv.add_first(s_, idx)
            st.session_state.state = s_
            st.rerun()
        return
    if st.session_state.pending:
        if client is None:
            state["error"] = setup_msg
            st.session_state.pending = None
        else:
            run_pending(idx, client)
            st.rerun()
    state = st.session_state.state
    if state is None:
        st.rerun()
    if state.get("error"):
        st.error(state["error"])
        if state["phase"] == "interview" and state["turns"][-1].get("answer") is not None:
            st.button("Tekrar dene", on_click=queue, args=("retry",))
    n = C.answered_count(state)
    if state.get("report"):
        render_result(state, idx)
    elif state["phase"] == "summary":
        render_summary(state)
    elif state["phase"] == "interview" and state["turns"][-1].get("answer") is None:
        st.caption(f"{n + 1}. soru · en fazla {config.MAX_QUESTIONS}")
        render_question(state, idx)
    render_history(state)
    st.divider()
    st.button("Yeni değerlendirme", on_click=queue, args=("reset",))


main()
