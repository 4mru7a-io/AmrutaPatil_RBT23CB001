"""Document similarity checker.

Embeds both documents sentence-by-sentence with sentence-transformers, scores
them, shows the matching sentence pairs, and (optionally) asks a local LLM to
explain the result.

LLM calls go through Open WebUI's OpenAI-compatible API when an API key is
available (``OPENWEBUI_API_KEY`` or ``apikey.txt``); otherwise the app falls
back to Ollama's native API.

Run:  streamlit run app.py
"""

import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://localhost:8080").rstrip("/")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:latest")
API_KEY_FILE = Path(__file__).with_name("apikey.txt")

EMBED_MODELS = {
    "all-MiniLM-L6-v2 (fast, English)": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2 (better, slower)": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2 (multilingual)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
MAX_SENTENCES = 3000  # per document; keeps the similarity matrix small


# --------------------------------------------------------------------------
# Text extraction + sentence splitting
# --------------------------------------------------------------------------
def extract_text(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        from docx import Document

        document = Document(io.BytesIO(data))
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n\n".join(p for p in parts if p.strip())
    return data.decode("utf-8", errors="ignore")


_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}")


def split_sentences(text: str, min_chars: int = 25) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n(?=\s*(?:[-*•]|\d+[.)])\s+)", "\n\n", text)  # split bullets
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # undo hard-wrapped lines (PDFs)
    parts = (s.strip() for s in _BOUNDARY.split(text))
    return [s for s in parts if len(s) >= min_chars][:MAX_SENTENCES]


# --------------------------------------------------------------------------
# Embeddings + scoring
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(model, sentences: list[str]) -> np.ndarray:
    return np.asarray(
        model.encode(
            sentences,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def compare(emb_a: np.ndarray, emb_b: np.ndarray, threshold: float) -> dict:
    if emb_a.size == 0 or emb_b.size == 0:
        raise ValueError("Both documents need at least one usable sentence.")

    sim = emb_a @ emb_b.T  # unit vectors -> cosine
    a_best = sim.max(axis=1)
    b_best = sim.max(axis=0)

    mean_a, mean_b = emb_a.mean(axis=0), emb_b.mean(axis=0)
    centroid_cos = float(
        mean_a @ mean_b / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-9)
    )
    return {
        "sim": sim,
        "a_best": a_best,
        "b_best": b_best,
        "a_idx": sim.argmax(axis=1),
        "b_idx": sim.argmax(axis=0),
        # headline: how well each doc's sentences are covered by the other, averaged
        "match_score": float((a_best.mean() + b_best.mean()) / 2),
        "centroid_cos": centroid_cos,
        "a_cov": float((a_best >= threshold).mean()),
        "b_cov": float((b_best >= threshold).mean()),
    }


def verdict(score: float) -> str:
    # Rough heuristics for MiniLM/MPNet-style models, not calibrated thresholds.
    if score >= 0.85:
        return "Near-duplicate / close paraphrase"
    if score >= 0.65:
        return "Strongly related content"
    if score >= 0.45:
        return "Same general topic, different content"
    return "Mostly different"


# --------------------------------------------------------------------------
# LLM backend (Open WebUI first, then plain Ollama)
# --------------------------------------------------------------------------
def get_api_key() -> str:
    """Read the Open WebUI key from the environment or the local key file."""
    key = os.getenv("OPENWEBUI_API_KEY", "").strip()
    if key:
        return key
    try:
        return API_KEY_FILE.read_text(encoding="utf-8").strip().strip("\"'")
    except OSError:
        return ""


@st.cache_data(ttl=30, show_spinner=False)
def list_llm_models() -> tuple[list[str], str | None]:
    """Return available model ids and which backend answered."""
    key = get_api_key()
    if key:
        try:
            r = requests.get(
                f"{OPENWEBUI_URL}/api/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=3,
            )
            r.raise_for_status()
            models = sorted(
                {
                    m["id"]
                    for m in r.json().get("data", [])
                    if m.get("id") and m.get("id") != "arena-model"
                }
            )
            if models:
                return models, "Open WebUI"
        except (requests.RequestException, ValueError, KeyError):
            pass

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        models = sorted(
            {m["name"] for m in r.json().get("models", []) if m.get("name")}
        )
        return models, "Ollama" if models else None
    except (requests.RequestException, ValueError):
        return [], None


def _stream_openwebui(model: str, prompt: str, key: str):
    with requests.post(
        f"{OPENWEBUI_URL}/api/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "think": False,  # keep Gemma-style reasoning out of the streamed answer
        },
        stream=True,
        timeout=(10, 600),
    ) as r:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail") or r.text
            except ValueError:
                detail = r.text
            raise requests.HTTPError(
                f"Open WebUI returned {r.status_code}: {detail}", response=r
            )

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or choices[0].get("message") or {}
                text = delta.get("content")
                if text:
                    yield text


def _stream_ollama(model: str, prompt: str):
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True, "think": False},
        stream=True,
        timeout=(10, 600),
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            if chunk.get("response"):
                yield chunk["response"]
            if chunk.get("done"):
                break


def stream_explanation(model: str, prompt: str):
    """Stream the explanation, falling back to Ollama if Open WebUI is down."""
    key = get_api_key()
    if key:
        started = False
        try:
            for token in _stream_openwebui(model, prompt, key):
                started = True
                yield token
            if started:
                return
        except requests.RequestException:
            if started:
                raise
            # Open WebUI was unreachable before producing output: try Ollama.
    yield from _stream_ollama(model, prompt)


def build_prompt(name_a, name_b, res, sents_a, sents_b, k=8) -> str:
    top = np.argsort(-res["a_best"])[:k]
    matched = "\n".join(
        f'- ({res["a_best"][i]:.2f}) A: "{sents_a[i][:200]}"  <->  B: "{sents_b[res["a_idx"][i]][:200]}"'
        for i in top
    )
    low_a = "\n".join(
        f'- "{sents_a[i][:200]}"' for i in np.argsort(res["a_best"])[:5]
    )
    low_b = "\n".join(
        f'- "{sents_b[i][:200]}"' for i in np.argsort(res["b_best"])[:5]
    )
    return f"""You are comparing two documents. An embedding model produced these results.

Document A: {name_a}
Document B: {name_b}
Overall match score: {res["match_score"]:.2f} (0 = unrelated, 1 = identical)

Most similar sentence pairs (similarity, A vs B):
{matched}

Sentences in A with the weakest match in B:
{low_a}

Sentences in B with the weakest match in A:
{low_b}

In under 200 words, explain: (1) what the two documents share, (2) how they differ, \
(3) whether the overlap looks like copying/paraphrase or just a shared topic. \
Use only the evidence above. Do not invent content."""


def render_results(state, threshold, top_n, use_llm, llm_model):
    """Render a stored comparison; cheap to re-run when widgets change."""
    sents_a, sents_b = state["sents_a"], state["sents_b"]
    a_best, b_best = state["a_best"], state["b_best"]
    a_cov = float((a_best >= threshold).mean())
    b_cov = float((b_best >= threshold).mean())

    # ---- scores ----
    st.subheader("Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Match score", f"{max(state['match_score'], 0) * 100:.1f}%")
    m2.metric("Whole-doc cosine", f"{max(state['centroid_cos'], 0) * 100:.1f}%")
    m3.metric(f"A covered by B (≥{threshold:.2f})", f"{a_cov * 100:.0f}%")
    m4.metric(f"B covered by A (≥{threshold:.2f})", f"{b_cov * 100:.0f}%")
    st.progress(float(np.clip(state["match_score"], 0, 1)))
    st.caption(
        f"**{verdict(state['match_score'])}** · {len(sents_a)} sentences in A, "
        f"{len(sents_b)} in B. Match score = average best-sentence match in each "
        "direction. Whole-doc cosine compares the averaged embeddings and runs "
        "higher for any two on-topic documents."
    )

    # ---- matching pairs ----
    st.subheader("Closest sentence pairs")
    order = np.argsort(-a_best)[:top_n]
    st.dataframe(
        pd.DataFrame(
            {
                "similarity": [round(float(a_best[i]), 3) for i in order],
                "Document A": [sents_a[i] for i in order],
                "Document B": [sents_b[state["a_idx"][i]] for i in order],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    # ---- unmatched ----
    u1, u2 = st.columns(2)
    for col, label, sents, best in (
        (u1, "Only in A (weakest matches)", sents_a, a_best),
        (u2, "Only in B (weakest matches)", sents_b, b_best),
    ):
        with col, st.expander(label):
            for i in np.argsort(best)[:10]:
                st.markdown(f"`{best[i]:.2f}`  {sents[i]}")

    # ---- LLM explanation ----
    if use_llm:
        head, action = st.columns([5, 1])
        head.subheader(f"Explanation ({llm_model})")
        if action.button("Regenerate", key="regenerate_explanation"):
            st.session_state.pop("explanation", None)
            st.rerun()

        explanation_key = (state["signature"], llm_model)
        cached = st.session_state.get("explanation")
        if cached and cached["key"] == explanation_key:
            st.markdown(cached["text"])
        else:
            prompt_res = {
                "a_best": a_best,
                "b_best": b_best,
                "a_idx": state["a_idx"],
                "match_score": state["match_score"],
            }
            try:
                text = st.write_stream(
                    stream_explanation(
                        llm_model,
                        build_prompt(
                            state["names"][0],
                            state["names"][1],
                            prompt_res,
                            sents_a,
                            sents_b,
                        ),
                    )
                )
                if text:
                    st.session_state["explanation"] = {
                        "key": explanation_key,
                        "text": text,
                    }
                else:
                    st.warning("The model returned an empty response.")
            except Exception as e:
                st.warning(f"LLM request failed: {e}")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Document Similarity", layout="wide")
st.title("Document similarity checker")

with st.sidebar:
    st.header("Settings")
    embed_label = st.selectbox("Embedding model", list(EMBED_MODELS))
    threshold = st.slider(
        "Match threshold",
        0.3,
        0.95,
        0.7,
        0.01,
        help="A sentence counts as 'covered' if its best match in the other doc scores at least this.",
    )
    top_n = st.slider("Pairs to show", 5, 50, 15)
    st.divider()
    models, backend = list_llm_models()
    use_llm = st.checkbox("Explain with local LLM", value=bool(models))
    if models:
        idx = models.index(DEFAULT_LLM_MODEL) if DEFAULT_LLM_MODEL in models else 0
        llm_model = st.selectbox(
            "LLM model", models, index=idx, disabled=not use_llm
        )
        st.caption(
            f"Connected to **{backend}**"
            + (" using the API key." if backend == "Open WebUI" and get_api_key() else ".")
        )
    else:
        llm_model = DEFAULT_LLM_MODEL
        st.caption(
            f"No local LLM reachable (tried Open WebUI at {OPENWEBUI_URL} "
            f"and Ollama at {OLLAMA_URL})."
        )

col1, col2 = st.columns(2)
types = ["txt", "md", "pdf", "docx"]
file_a = col1.file_uploader("Document A", type=types, key="a")
file_b = col2.file_uploader("Document B", type=types, key="b")

signature = None
if file_a and file_b:
    signature = (
        file_a.name,
        getattr(file_a, "size", None),
        getattr(file_a, "file_id", None),
        file_b.name,
        getattr(file_b, "size", None),
        getattr(file_b, "file_id", None),
        embed_label,
    )

if st.button("Compare", type="primary", disabled=not (file_a and file_b)):
    st.session_state.pop("comparison", None)
    st.session_state.pop("explanation", None)

    try:
        text_a, text_b = extract_text(file_a), extract_text(file_b)
    except Exception as e:
        st.error(f"Couldn't read one of the files: {e}")
        st.stop()

    sents_a, sents_b = split_sentences(text_a), split_sentences(text_b)
    if not sents_a or not sents_b:
        st.error(
            "Couldn't extract usable sentences from one of the files "
            "(scanned PDF with no text layer?)."
        )
        st.stop()

    try:
        model = load_embedder(EMBED_MODELS[embed_label])
        with st.spinner("Embedding..."):
            emb_a, emb_b = embed(model, sents_a), embed(model, sents_b)
        res = compare(emb_a, emb_b, threshold)
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        st.stop()

    st.session_state["comparison"] = {
        "signature": signature,
        "names": (file_a.name, file_b.name),
        "sents_a": sents_a,
        "sents_b": sents_b,
        "a_best": res["a_best"],
        "b_best": res["b_best"],
        "a_idx": res["a_idx"],
        "match_score": res["match_score"],
        "centroid_cos": res["centroid_cos"],
    }

state = st.session_state.get("comparison")
if state:
    if state["signature"] != signature:
        st.info("Files or embedding model changed. Click **Compare** to update the result.")
    else:
        render_results(state, threshold, top_n, use_llm, llm_model)

