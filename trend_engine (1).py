
import json
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

REQUIRED_FIELDS = [
    "Name (QE)",
    "Title (QE)",
    "Event Subcategory (EV)",
    "Event Defect Code (EV)",
    "Direct cause details (QE)",
    "Day of Created Date (QE)",
]

DEFAULT_MEMORY_PATH = Path("trend_memory.json")
DEFAULT_FEEDBACK_PATH = Path("trend_feedback.json")
DEFAULT_AUDIT_PATH = Path("trend_audit.json")
DEFAULT_TRENDS_PATH = Path("trends.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    x = str(x)
    x = x.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    x = " ".join(x.split())
    return x.strip()


def _norm_colname(c: str) -> str:
    return " ".join(str(c).strip().lower().replace("\n", " ").split())


def _safe_json_load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _safe_json_save(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path | str, default=None):
    return _safe_json_load(Path(path), [] if default is None else default)


def save_json(path: Path | str, data) -> None:
    _safe_json_save(Path(path), data)


def _map_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    norm = {_norm_colname(c): c for c in cols}

    def pick(candidates):
        for cand in candidates:
            for n, orig in norm.items():
                if cand in n:
                    return orig
        return None

    col_name = pick(["name (qe)", "qe number", "qe#", "qe id", "qe-id", "event id", "qe"])
    col_title = pick(["title (qe)", "title", "short description", "beschreibung", "titel"])
    col_subcat = pick(["event subcategory", "subcategory", "sub category", "unterkategorie"])
    col_defect = pick(["event defect code", "defect code", "defect", "fehlercode", "code"])
    col_cause = pick(["direct cause details", "direct cause", "cause details", "cause", "ursache"])
    col_date = pick(["day of created date", "created date", "creation date", "date created", "created", "datum"])

    rename = {}
    if col_name:
        rename[col_name] = "Name (QE)"
    if col_title:
        rename[col_title] = "Title (QE)"
    if col_subcat:
        rename[col_subcat] = "Event Subcategory (EV)"
    if col_defect:
        rename[col_defect] = "Event Defect Code (EV)"
    if col_cause:
        rename[col_cause] = "Direct cause details (QE)"
    if col_date:
        rename[col_date] = "Day of Created Date (QE)"

    df = df.rename(columns=rename)

    for f in REQUIRED_FIELDS:
        if f not in df.columns:
            df[f] = ""

    for f in REQUIRED_FIELDS:
        df[f] = df[f].apply(_clean_text)

    df = df[~((df["Name (QE)"] == "") & (df["Title (QE)"] == ""))].copy()
    return df


def _tokenize(t: str):
    t = _clean_text(t).lower().replace("/", " ")
    out = []
    for w in t.split():
        w = "".join(ch for ch in w if ch.isalnum())
        if len(w) >= 4:
            out.append(w)
    return out


def _top_phrases(texts, top_k=8):
    stop = {
        "und", "oder", "der", "die", "das", "mit", "auf", "von", "für", "ist", "eine", "ein",
        "bei", "wurde", "werden", "nicht", "als", "aufgrund", "nach", "vor", "während",
        "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were",
        "issue", "problem", "found", "noted", "event", "events", "details", "direct", "cause"
    }
    from collections import Counter
    cnt = Counter()
    for t in texts:
        toks = [w for w in _tokenize(t) if w not in stop]
        for i in range(len(toks) - 1):
            cnt[f"{toks[i]} {toks[i+1]}"] += 1
        for i in range(len(toks) - 2):
            cnt[f"{toks[i]} {toks[i+1]} {toks[i+2]}"] += 1

    phrases = [p for p, _ in cnt.most_common(top_k)]
    if not phrases:
        c2 = Counter()
        for t in texts:
            for w in [w for w in _tokenize(t) if w not in stop]:
                c2[w] += 1
        phrases = [w for w, _ in c2.most_common(top_k)]
    return phrases


def _normalize_phrase(phrase: str) -> str:
    phrase = _clean_text(phrase).strip(" -_,.;:")
    if not phrase:
        return ""
    return phrase[0].lower() + phrase[1:] if len(phrase) > 1 else phrase.lower()


def _extract_cause_signal(causes, titles, phrases):
    text_blob = " ".join([_clean_text(t) for t in titles] + [_clean_text(c) for c in causes]).lower()

    cause_rules = [
        (r"\b(fehlend|fehlende|missing|nicht vorhanden|unvollständig|unklar)\b.*\b(dokumentation|nachweis|protokoll|unterlage|bericht|doku)\b",
         "Hinweise auf gemeinsame Ursachen liegen in fehlender oder unvollständiger Dokumentation."),
        (r"\b(dokumentation|nachweis|protokoll|unterlage|bericht|doku)\b.*\b(fehlend|fehlende|missing|nicht vorhanden|unvollständig|unklar)\b",
         "Hinweise auf gemeinsame Ursachen liegen in fehlender oder unvollständiger Dokumentation."),
        (r"\b(schulung|training|einweisung|qualifizierung)\b.*\b(fehlend|unzureichend|nicht erfolgt)\b",
         "Hinweise auf gemeinsame Ursachen deuten auf unzureichende Schulung oder Einweisung hin."),
        (r"\b(fehleinstellung|einstellung|parameter|kalibrier|justier)\b",
         "Hinweise auf gemeinsame Ursachen deuten auf fehlerhafte Einstellungen oder Justagen hin."),
        (r"\b(verschleiß|abnutzung|beschädig|defekt|bruch|leck)\b",
         "Hinweise auf gemeinsame Ursachen deuten auf technische Defekte oder Verschleiß hin."),
        (r"\b(reinigung|verschmutz|kontamination|rückstand)\b",
         "Hinweise auf gemeinsame Ursachen deuten auf Reinigungs- oder Kontaminationsprobleme hin."),
        (r"\b(prozess|ablauf|workflow|schnittstelle|übergabe)\b",
         "Hinweise auf gemeinsame Ursachen deuten auf Schwächen im Prozessablauf oder in der Übergabe hin."),
    ]

    for pattern, sentence in cause_rules:
        if re.search(pattern, text_blob):
            return sentence

    if phrases:
        phrase = _normalize_phrase(phrases[0])
        return f"Es gibt Hinweise auf eine gemeinsame Ursache im Zusammenhang mit {phrase}."

    return "Es gibt Hinweise auf eine gemeinsame Ursache, die in den Beschreibungen wiederholt ähnlich erscheint."


def _domain_trend_title(titles, causes, phrases):
    text_blob = " ".join([_clean_text(t) for t in titles] + [_clean_text(c) for c in causes]).lower()

    specific_rules = [
        (
            r"\b(personenaufzug|aufzug|elevator|lift)\b.*\b(stecken|steck|blockier|stillstand|stuck|stopp)\b|\b(stecken|steck|blockier|stillstand|stuck|stopp)\b.*\b(personenaufzug|aufzug|elevator|lift)\b",
            "Wiederholte Vorfälle mit blockierten oder steckengebliebenen Personenaufzügen",
        ),
        (
            r"\b(dokumentation|nachweis|protokoll|unterlage|bericht|doku)\b.*\b(fehlend|fehlende|missing|nicht vorhanden|unklar|unvollständig)\b|\b(fehlend|fehlende|missing|nicht vorhanden|unklar|unvollständig)\b.*\b(dokumentation|nachweis|protokoll|unterlage|bericht|doku)\b",
            "Wiederholte Abweichungen durch fehlende oder unvollständige Dokumentation",
        ),
        (
            r"\b(etikett|label|kennzeichnung|beschriftung)\b",
            "Wiederholte Abweichungen bei Kennzeichnung oder Etikettierung",
        ),
        (
            r"\b(reinigung|verschmutz|kontamination|rückstand)\b",
            "Wiederholte Abweichungen im Zusammenhang mit Reinigung oder Kontamination",
        ),
        (
            r"\b(sensor|messung|signal|alarm|anzeige)\b",
            "Wiederholte Abweichungen bei Sensorik, Messwerten oder Signalen",
        ),
    ]

    for pattern, label in specific_rules:
        if re.search(pattern, text_blob):
            return label

    core = _normalize_phrase(phrases[0]) if phrases else "ähnlichen Abweichungen"
    return f"Wiederholte Vorkommnisse im Zusammenhang mit {core}"


def _domain_trend_summary(subcat, defect, n, phrases, examples, causes, titles):
    topic = _normalize_phrase(phrases[0]) if phrases else "ähnlichen Beschreibungen"
    cause_sentence = _extract_cause_signal(causes, titles, phrases)
    example_sentence = ""
    if examples:
        example_sentence = f" Typische Beispiele sind: {'; '.join([e[:110] + ('…' if len(e) > 110 else '') for e in examples])}."
    return (
        f"In der Gruppe {subcat} → {defect} treten {n} ähnliche Events wiederholt auf, "
        f"vor allem im Zusammenhang mit {topic}. "
        f"{cause_sentence}{example_sentence}"
    )


def _build_similarity(texts):
    v_word = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Xw = v_word.fit_transform(texts)
    Xc = v_char.fit_transform(texts)
    X = hstack([Xw.multiply(0.70), Xc.multiply(0.30)])
    return cosine_similarity(X)


def _connected_components(sim, threshold):
    n = sim.shape[0]
    visited = [False] * n
    comps = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            neigh = [v for v in range(n) if v != u and sim[u, v] >= threshold]
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(sorted(comp))

    comps.sort(key=lambda c: (-len(c), c))
    return comps


def _cohesion(sim, idxs):
    if len(idxs) < 2:
        return 0.0
    sub = sim[np.ix_(idxs, idxs)]
    tri = sub[np.triu_indices(len(idxs), k=1)]
    return float(tri.mean()) if tri.size else 0.0


def _representatives(sim, idxs, k=3):
    sub = sim[np.ix_(idxs, idxs)]
    scores = sub.mean(axis=1)
    order = np.argsort(-scores)
    return [idxs[int(i)] for i in order[:min(k, len(idxs))]]


def _text_signature(parts: List[str]) -> str:
    joined = " | ".join(sorted(_clean_text(p) for p in parts if _clean_text(p)))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]


def make_trend_instance_id(subcat: str, defect: str, qe_numbers: List[str]) -> str:
    raw = f"{_clean_text(subcat)}|{_clean_text(defect)}|{'|'.join(sorted(_clean_text(x) for x in qe_numbers))}"
    return "TR-" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _build_cluster_memory_text(trend: Dict[str, Any]) -> str:
    pieces = [
        trend.get("trend_title", ""),
        trend.get("trend_summary", ""),
        " ".join(trend.get("patterns", []) or []),
        " ".join(trend.get("sample_titles", []) or []),
    ]
    return " | ".join(_clean_text(p) for p in pieces if _clean_text(p))


def _build_memory_item_text(memory_item: Dict[str, Any]) -> str:
    pieces = [
        memory_item.get("canonical_title", ""),
        memory_item.get("canonical_summary", ""),
        " ".join(memory_item.get("patterns", []) or []),
        " ".join(memory_item.get("sample_titles", []) or []),
    ]
    return " | ".join(_clean_text(p) for p in pieces if _clean_text(p))


def load_trend_memory(path: Path | str = DEFAULT_MEMORY_PATH) -> List[Dict[str, Any]]:
    return load_json(path, default=[])


def load_feedback_log(path: Path | str = DEFAULT_FEEDBACK_PATH) -> List[Dict[str, Any]]:
    return load_json(path, default=[])


def load_audit_log(path: Path | str = DEFAULT_AUDIT_PATH) -> List[Dict[str, Any]]:
    return load_json(path, default=[])


def append_feedback(entry: Dict[str, Any], path: Path | str = DEFAULT_FEEDBACK_PATH) -> None:
    p = Path(path)
    data = load_feedback_log(p)
    data.append(entry)
    save_json(p, data)


def append_audit(entry: Dict[str, Any], path: Path | str = DEFAULT_AUDIT_PATH) -> None:
    p = Path(path)
    data = load_audit_log(p)
    data.append(entry)
    save_json(p, data)


def get_trend_audit_entries(trend_instance_id: str, path: Path | str = DEFAULT_AUDIT_PATH) -> List[Dict[str, Any]]:
    return [x for x in load_audit_log(path) if x.get("trend_instance_id") == trend_instance_id]


def match_to_trend_memory(cluster_trend: Dict[str, Any], memory_items: List[Dict[str, Any]], threshold: float = 0.60):
    candidates = [
        item for item in memory_items
        if _clean_text(item.get("subcategory")) == _clean_text(cluster_trend.get("subcategory"))
        and _clean_text(item.get("defect_code")) == _clean_text(cluster_trend.get("defect_code"))
    ]
    if not candidates:
        return None, 0.0

    cluster_text = _build_cluster_memory_text(cluster_trend)
    texts = [cluster_text] + [_build_memory_item_text(x) for x in candidates]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score >= threshold:
        return candidates[best_idx], best_score
    return None, best_score


def _next_canonical_trend_id(memory: List[Dict[str, Any]]) -> str:
    nums = []
    for item in memory:
        cid = str(item.get("canonical_trend_id", ""))
        if cid.startswith("CT-"):
            try:
                nums.append(int(cid.split("-")[1]))
            except Exception:
                pass
    return f"CT-{(max(nums) + 1) if nums else 1:04d}"


def review_trend(
    trend: Dict[str, Any],
    action: str,
    reviewed_by: str,
    comment: str = "",
    new_title: Optional[str] = None,
    new_summary: Optional[str] = None,
    memory_path: Path | str = DEFAULT_MEMORY_PATH,
    feedback_path: Path | str = DEFAULT_FEEDBACK_PATH,
    audit_path: Path | str = DEFAULT_AUDIT_PATH,
) -> Dict[str, Any]:
    reviewed_by = _clean_text(reviewed_by) or "Unbekannt"
    action = _clean_text(action).lower()
    now = _utc_now_iso()
    allowed = {"confirmed", "rejected", "renamed"}
    if action not in allowed:
        raise ValueError(f"Unsupported action: {action}")

    memory = load_trend_memory(memory_path)
    trend_copy = dict(trend)

    old_title = trend_copy.get("trend_title")
    old_summary = trend_copy.get("trend_summary")
    canonical_trend_id = trend_copy.get("canonical_trend_id")

    if action == "renamed":
        if _clean_text(new_title):
            trend_copy["trend_title"] = _clean_text(new_title)
        if _clean_text(new_summary):
            trend_copy["trend_summary"] = _clean_text(new_summary)
        trend_copy["review_status"] = "renamed"

    elif action == "rejected":
        trend_copy["review_status"] = "rejected"

    elif action == "confirmed":
        trend_copy["review_status"] = "confirmed"
        if not canonical_trend_id:
            canonical_trend_id = _next_canonical_trend_id(memory)
            trend_copy["canonical_trend_id"] = canonical_trend_id

        existing = None
        for item in memory:
            if item.get("canonical_trend_id") == canonical_trend_id:
                existing = item
                break

        if existing is None:
            memory.append({
                "canonical_trend_id": canonical_trend_id,
                "subcategory": trend_copy.get("subcategory"),
                "defect_code": trend_copy.get("defect_code"),
                "canonical_title": trend_copy.get("trend_title"),
                "canonical_summary": trend_copy.get("trend_summary"),
                "patterns": trend_copy.get("patterns", [])[:10],
                "sample_titles": trend_copy.get("sample_titles", [])[:5],
                "example_qe_numbers": trend_copy.get("qe_numbers", [])[:20],
                "times_confirmed": 1,
                "last_confirmed_at": now,
                "created_at": now,
                "updated_at": now,
            })
        else:
            existing["canonical_title"] = trend_copy.get("trend_title")
            existing["canonical_summary"] = trend_copy.get("trend_summary")
            existing["patterns"] = list(dict.fromkeys((existing.get("patterns", []) or []) + (trend_copy.get("patterns", []) or [])))[:10]
            existing["sample_titles"] = list(dict.fromkeys((existing.get("sample_titles", []) or []) + (trend_copy.get("sample_titles", []) or [])))[:5]
            existing["example_qe_numbers"] = list(dict.fromkeys((existing.get("example_qe_numbers", []) or []) + (trend_copy.get("qe_numbers", []) or [])))[:20]
            existing["times_confirmed"] = int(existing.get("times_confirmed", 0)) + 1
            existing["last_confirmed_at"] = now
            existing["updated_at"] = now

        save_json(memory_path, memory)

    feedback_entry = {
        "review_id": f"REV-{_text_signature([trend_copy.get('trend_instance_id', ''), now, action, reviewed_by])}",
        "trend_instance_id": trend_copy.get("trend_instance_id"),
        "canonical_trend_id": trend_copy.get("canonical_trend_id"),
        "action": action,
        "reviewed_by": reviewed_by,
        "reviewed_at": now,
        "comment": _clean_text(comment),
        "old_title": old_title,
        "new_title": trend_copy.get("trend_title"),
        "old_summary": old_summary,
        "new_summary": trend_copy.get("trend_summary"),
    }
    append_feedback(feedback_entry, feedback_path)

    audit_entry = {
        "audit_id": f"AUD-{_text_signature([trend_copy.get('trend_instance_id', ''), now, action, reviewed_by, feedback_entry['review_id']])}",
        "trend_instance_id": trend_copy.get("trend_instance_id"),
        "canonical_trend_id": trend_copy.get("canonical_trend_id"),
        "event_type": action,
        "event_at": now,
        "event_by": reviewed_by,
        "comment": _clean_text(comment),
        "old_title": old_title,
        "new_title": trend_copy.get("trend_title"),
        "old_summary": old_summary,
        "new_summary": trend_copy.get("trend_summary"),
    }
    append_audit(audit_entry, audit_path)
    return trend_copy


def apply_feedback_statuses(trends: List[Dict[str, Any]], feedback_path: Path | str = DEFAULT_FEEDBACK_PATH) -> List[Dict[str, Any]]:
    feedback = load_feedback_log(feedback_path)
    latest = {}
    for entry in feedback:
        tid = entry.get("trend_instance_id")
        ts = entry.get("reviewed_at", "")
        if tid and (tid not in latest or ts >= latest[tid].get("reviewed_at", "")):
            latest[tid] = entry

    out = []
    for trend in trends:
        t = dict(trend)
        fb = latest.get(t.get("trend_instance_id"))
        if fb:
            t["review_status"] = fb.get("action", t.get("review_status", "proposed"))
            if fb.get("new_title"):
                t["trend_title"] = fb["new_title"]
            if fb.get("new_summary"):
                t["trend_summary"] = fb["new_summary"]
            if fb.get("canonical_trend_id"):
                t["canonical_trend_id"] = fb["canonical_trend_id"]
        out.append(t)
    return out


def generate_trends(
    df: pd.DataFrame,
    sim_threshold: float = 0.62,
    cohesion_min: float = 0.58,
    memory_path: Path | str = DEFAULT_MEMORY_PATH,
    feedback_path: Path | str = DEFAULT_FEEDBACK_PATH,
    memory_match_threshold: float = 0.60,
) -> Dict[str, Any]:
    df = _map_headers(df)
    trends = []
    group_stats = []
    memory = load_trend_memory(memory_path)

    grouped = df.groupby(["Event Subcategory (EV)", "Event Defect Code (EV)"], dropna=False, sort=True)

    for (subcat, defect), g in grouped:
        subcat = subcat if subcat else "UNSPECIFIED"
        defect = defect if defect else "UNSPECIFIED"

        titles = g["Title (QE)"].tolist()
        causes = g["Direct cause details (QE)"].tolist()
        ids = g["Name (QE)"].tolist()

        group_stats.append({
            "subcategory": subcat,
            "defect_code": defect,
            "n_events_group": int(len(g)),
        })

        if len(g) < 3:
            continue

        sem = [f"{_clean_text(t)} | {_clean_text(c)}" for t, c in zip(titles, causes)]
        sim = _build_similarity(sem)
        comps = _connected_components(sim, threshold=float(sim_threshold))

        for comp in comps:
            if len(comp) < 3:
                continue

            coh = _cohesion(sim, comp)
            if coh < float(cohesion_min):
                continue

            comp_titles = [titles[i] for i in comp]
            comp_causes = [causes[i] for i in comp]
            comp_ids = [ids[i] for i in comp]

            phrases = _top_phrases(comp_titles + comp_causes, top_k=8)
            trend_title = _domain_trend_title(comp_titles, comp_causes, phrases)

            reps = _representatives(sim, comp, k=3)
            examples = [_clean_text(titles[i]) for i in reps if _clean_text(titles[i])]

            summary = _domain_trend_summary(
                subcat=subcat,
                defect=defect,
                n=len(comp),
                phrases=phrases,
                examples=examples,
                causes=comp_causes,
                titles=comp_titles,
            )

            trend = {
                "trend_instance_id": make_trend_instance_id(subcat, defect, comp_ids),
                "canonical_trend_id": None,
                "subcategory": subcat,
                "defect_code": defect,
                "trend_title": trend_title,
                "trend_summary": summary,
                "suggested_title": None,
                "suggested_summary": None,
                "n_events": int(len(comp)),
                "similarity": round(coh, 3),
                "qe_numbers": comp_ids,
                "sample_titles": examples,
                "patterns": phrases[:10],
                "review_status": "proposed",
                "suggested_from_memory": False,
                "memory_match_score": 0.0,
            }

            match, score = match_to_trend_memory(trend, memory, threshold=memory_match_threshold)
            if match:
                trend["canonical_trend_id"] = match.get("canonical_trend_id")
                trend["suggested_from_memory"] = True
                trend["memory_match_score"] = round(score, 3)
                trend["suggested_title"] = match.get("canonical_title")
                trend["suggested_summary"] = match.get("canonical_summary")
                trend["trend_title"] = match.get("canonical_title") or trend["trend_title"]
                trend["trend_summary"] = match.get("canonical_summary") or trend["trend_summary"]
            else:
                trend["memory_match_score"] = round(score, 3) if score else 0.0

            trends.append(trend)

    trends = apply_feedback_statuses(trends, feedback_path=feedback_path)
    trends.sort(
        key=lambda t: (
            -t["n_events"],
            -t["similarity"],
            t["subcategory"],
            t["defect_code"],
            t["trend_title"],
        )
    )

    return {
        "meta": {
            "version": "deviations-trending-mvp-v5-shared-engine",
            "trend_definition": "Connected components on TFIDF similarity graph within Subcategory+Defect (Title+DirectCause).",
            "created_date_note": "Day of Created Date is never used for clustering.",
            "parameters": {
                "sim_threshold_edge": sim_threshold,
                "cohesion_min": cohesion_min,
                "memory_match_threshold": memory_match_threshold,
            },
        },
        "group_rollup": group_stats,
        "trends": trends,
    }
