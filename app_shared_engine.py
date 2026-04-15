
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from trend_engine import (
    DEFAULT_AUDIT_PATH,
    DEFAULT_FEEDBACK_PATH,
    DEFAULT_MEMORY_PATH,
    apply_feedback_statuses,
    generate_trends,
    get_trend_audit_entries,
    load_audit_log,
    load_feedback_log,
    load_json,
    review_trend,
)

st.set_page_config(
    page_title="Deviations Trending MVP",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📊 Deviations Trending MVP")
st.caption("Live-Demo mit Review, Audit Trail und Trend Memory.")

st.markdown(
    """
<style>
section[data-testid="stSidebar"] { width: 360px !important; }
div[data-testid="stMarkdownContainer"] { overflow-wrap: anywhere; }
div[data-testid="stMarkdownContainer"] p { white-space: normal !important; }
code { white-space: pre-wrap !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 1.1rem; }
</style>
    """,
    unsafe_allow_html=True,
)

mode = st.radio("Modus", ["Live-Analyse (Excel Upload)", "Repository-Modus (trends.json)"], horizontal=True)

data = None
if mode == "Live-Analyse (Excel Upload)":
    up = st.file_uploader("Excel (.xlsx) hochladen", type=["xlsx"])
    if up is None:
        st.info("Bitte eine Excel-Datei hochladen, um die Analyse live zu starten.")
        st.stop()
    df_in = pd.read_excel(up, sheet_name=0)
else:
    p = Path("trends.json")
    upj = st.file_uploader("Optional: trends.json hochladen", type=["json"])
    if upj is not None:
        data = json.load(upj)
    elif p.exists():
        data = load_json(p, default={})
    else:
        st.warning("Kein trends.json gefunden. Nutze Live-Upload oder lege trends.json im Repo-Root ab.")
        st.stop()

st.sidebar.header("Analyse & Filter")

sim_edge = st.sidebar.slider(
    "Similarity-Kante (Graph) — Edge Threshold",
    0.40,
    0.85,
    0.62,
    0.01,
    help="Ab wann zwei Events als ähnlich verbunden werden.",
)
cohesion_min = st.sidebar.slider(
    "Trend Similarity — Mindestkohäsion",
    0.40,
    0.90,
    0.58,
    0.01,
    help="Wie homogen ein Trend-Cluster im Mittel sein muss.",
)
memory_threshold = st.sidebar.slider(
    "Trend Memory Match",
    0.40,
    0.95,
    0.60,
    0.01,
    help="Ab wann ein neuer Cluster einem bestätigten Trend aus dem Trend Memory zugeordnet wird.",
)

if mode == "Live-Analyse (Excel Upload)":
    data = generate_trends(
        df_in,
        sim_threshold=sim_edge,
        cohesion_min=cohesion_min,
        memory_match_threshold=memory_threshold,
    )
else:
    trends_raw = data.get("trends", [])
    data["trends"] = apply_feedback_statuses(trends_raw, feedback_path=DEFAULT_FEEDBACK_PATH)

st.download_button(
    "⬇️ trends.json herunterladen",
    data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="trends.json",
    mime="application/json",
)

trends = pd.DataFrame(data.get("trends", []))
if trends.empty:
    st.warning("Keine Trends gefunden. Tipp: Similarity-Kante senken oder Kohäsion-Minimum senken.")
    st.stop()

subcats = ["(alle)"] + sorted(trends["subcategory"].fillna("UNSPECIFIED").unique().tolist())
defects = ["(alle)"] + sorted(trends["defect_code"].fillna("UNSPECIFIED").unique().tolist())
sel_sub = st.sidebar.selectbox("Event Subcategory", subcats)
sel_def = st.sidebar.selectbox("Event Defect Code", defects)
review_filter = st.sidebar.selectbox("Review-Status", ["(alle)", "proposed", "confirmed", "rejected", "renamed"])
memory_filter = st.sidebar.selectbox("Trend Memory", ["(alle)", "nur Vorschläge aus Memory", "nur neue Trends"])
min_events = st.sidebar.slider("Min. Events pro Trend", 3, int(max(3, trends["n_events"].max())), 3)
min_sim = st.sidebar.slider("Min. Similarity (Trend)", 0.40, 0.95, 0.58, 0.01)
search = st.sidebar.text_input("Suche (Titel/Summary/Muster)")

f = trends.copy()
if sel_sub != "(alle)":
    f = f[f["subcategory"] == sel_sub]
if sel_def != "(alle)":
    f = f[f["defect_code"] == sel_def]
if review_filter != "(alle)":
    f = f[f["review_status"] == review_filter]
if memory_filter == "nur Vorschläge aus Memory":
    f = f[f["suggested_from_memory"] == True]
elif memory_filter == "nur neue Trends":
    f = f[f["suggested_from_memory"] != True]

f = f[(f["n_events"] >= int(min_events)) & (f["similarity"] >= float(min_sim))]

if search.strip():
    s = search.strip().lower()

    def match(row):
        blob = " ".join([
            str(row.get("trend_title", "") or ""),
            str(row.get("trend_summary", "") or ""),
            " ".join(row.get("patterns") or []),
            str(row.get("suggested_title", "") or ""),
            str(row.get("suggested_summary", "") or ""),
        ]).lower()
        return s in blob

    f = f[f.apply(match, axis=1)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trends (gefiltert)", int(len(f)))
c2.metric("Events in Trends", int(f["n_events"].sum()) if not f.empty else 0)
c3.metric("Bestätigte Trends im Memory", len(load_json(DEFAULT_MEMORY_PATH, default=[])))
c4.metric("Audit-Einträge", len(load_audit_log(DEFAULT_AUDIT_PATH)))

st.subheader("📋 Trend-Übersicht (priorisiert)")

tbl = f.copy()
tbl = tbl.sort_values(["n_events", "similarity", "subcategory", "defect_code"], ascending=[False, False, True, True])
tbl.insert(0, "rank", range(1, len(tbl) + 1))

show_cols = [
    "rank", "subcategory", "defect_code", "n_events", "similarity",
    "review_status", "suggested_from_memory", "memory_match_score", "trend_title", "trend_summary"
]
st.dataframe(
    tbl[show_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "rank": st.column_config.NumberColumn("#", format="%d"),
        "subcategory": st.column_config.TextColumn("Subcategory"),
        "defect_code": st.column_config.TextColumn("Defect Code"),
        "n_events": st.column_config.NumberColumn("Events", format="%d"),
        "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
        "review_status": st.column_config.TextColumn("Review"),
        "suggested_from_memory": st.column_config.CheckboxColumn("Aus Memory"),
        "memory_match_score": st.column_config.NumberColumn("Memory Score", format="%.3f"),
        "trend_title": st.column_config.TextColumn("Trend"),
        "trend_summary": st.column_config.TextColumn("Beschreibung"),
    },
    height=min(720, 44 + 28 * min(len(tbl), 18)),
)

st.subheader("🔎 Trend-Details")

options = ["(wähle Trend)"] + [
    f"[{int(r.rank)}] {r.trend_title}  —  {r.subcategory} → {r.defect_code}  (n={int(r.n_events)}, sim={float(r.similarity):.3f})"
    for r in tbl.itertuples(index=False)
]
choice = st.selectbox("Trend auswählen", options, index=1 if len(options) > 1 else 0)

if choice != "(wähle Trend)":
    rank = int(choice.split("]")[0].strip("[ "))
    row = tbl[tbl["rank"] == rank].iloc[0].to_dict()

    st.markdown(f"### {row['trend_title']}")
    st.write(row["trend_summary"])

    colx, coly = st.columns([1, 2])
    with colx:
        st.metric("Events", int(row["n_events"]))
        st.metric("Similarity", float(row["similarity"]))
        st.write(f"**Gruppe:** `{row['subcategory']} → {row['defect_code']}`")
        st.write(f"**Review-Status:** `{row.get('review_status', 'proposed')}`")
        if row.get("canonical_trend_id"):
            st.write(f"**Canonical Trend ID:** `{row['canonical_trend_id']}`")
    with coly:
        st.write("**Häufige Muster (Phrasen):**")
        pats = row.get("patterns") or []
        st.write(", ".join(pats[:10]) if pats else "—")
        if row.get("suggested_from_memory"):
            st.info(
                f"Dieser Trend wurde aus dem Trend Memory vorgeschlagen "
                f"(Score: {float(row.get('memory_match_score', 0.0)):.3f})."
            )
            if row.get("suggested_title"):
                st.write(f"**Vorgeschlagener bestätigter Titel:** {row['suggested_title']}")
            if row.get("suggested_summary"):
                st.write(f"**Vorgeschlagene bestätigte Beschreibung:** {row['suggested_summary']}")

    st.write("**QE Numbers:**")
    st.code("\n".join(row.get("qe_numbers") or []))

    st.write("**Beispiel-Titel:**")
    samples = row.get("sample_titles") or []
    if samples:
        for s in samples:
            st.write(f"- {s}")
    else:
        st.write("—")

    st.subheader("✅ Review")
    reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))
    st.session_state["reviewer_name"] = reviewer

    new_title = st.text_input("Titel anpassen", value=row.get("trend_title", ""), key=f"title_{row['trend_instance_id']}")
    new_summary = st.text_area("Beschreibung anpassen", value=row.get("trend_summary", ""), key=f"summary_{row['trend_instance_id']}")
    review_comment = st.text_area("Kommentar / Begründung", key=f"comment_{row['trend_instance_id']}")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Bestätigen", key=f"confirm_{row['trend_instance_id']}"):
            review_trend(
                trend=row,
                action="confirmed",
                reviewed_by=reviewer,
                comment=review_comment,
                new_title=new_title,
                new_summary=new_summary,
            )
            st.success("Trend wurde bestätigt und ins Trend Memory übernommen.")
            st.rerun()
    with c2:
        if st.button("Ablehnen", key=f"reject_{row['trend_instance_id']}"):
            review_trend(
                trend=row,
                action="rejected",
                reviewed_by=reviewer,
                comment=review_comment,
            )
            st.warning("Trend wurde abgelehnt.")
            st.rerun()
    with c3:
        if st.button("Umbenennen", key=f"rename_{row['trend_instance_id']}"):
            review_trend(
                trend=row,
                action="renamed",
                reviewed_by=reviewer,
                comment=review_comment,
                new_title=new_title,
                new_summary=new_summary,
            )
            st.success("Trendtitel und Beschreibung wurden aktualisiert.")
            st.rerun()

    with st.expander("Audit Trail für diesen Trend", expanded=False):
        audit_entries = get_trend_audit_entries(row["trend_instance_id"], path=DEFAULT_AUDIT_PATH)
        if audit_entries:
            st.dataframe(pd.DataFrame(audit_entries), use_container_width=True, hide_index=True)
        else:
            st.info("Für diesen Trend gibt es noch keine Audit-Einträge.")

st.subheader("🧾 Globaler Audit Trail")
audit_all = load_audit_log(DEFAULT_AUDIT_PATH)
if audit_all:
    st.dataframe(pd.DataFrame(audit_all), use_container_width=True, hide_index=True, height=240)
else:
    st.info("Noch keine Audit-Einträge vorhanden.")

with st.expander("Trend Memory", expanded=False):
    memory_df = pd.DataFrame(load_json(DEFAULT_MEMORY_PATH, default=[]))
    if not memory_df.empty:
        st.dataframe(memory_df, use_container_width=True, hide_index=True)
    else:
        st.info("Das Trend Memory ist aktuell leer.")

with st.expander("Feedback Log", expanded=False):
    feedback_df = pd.DataFrame(load_feedback_log(DEFAULT_FEEDBACK_PATH))
    if not feedback_df.empty:
        st.dataframe(feedback_df, use_container_width=True, hide_index=True)
    else:
        st.info("Noch keine Feedback-Einträge vorhanden.")

with st.expander("Raw JSON Preview", expanded=False):
    st.json(data)
