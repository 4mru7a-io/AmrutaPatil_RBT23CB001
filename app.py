"""
BPE Tokenizer & Text Analyzer
-----------------------------
A Streamlit dashboard that shows how the pretrained GPT-2 Byte Pair Encoding
(BPE) tokenizer splits text into tokens and token IDs. Everything runs locally:
no LLM API and no API key. The tokenizer files are downloaded once from the
Hugging Face Hub and then cached on disk.
"""

import html
import re

import pandas as pd
import streamlit as st
from transformers import AutoTokenizer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME = "gpt2"

SAMPLE_TEXT = (
    "Tokenization is the foundation of modern natural language processing. "
    "Unbelievably long words like internationalization are split into subword "
    "units. Numbers such as 2024 and 3.14159, punctuation!? and emojis "
    "\U0001F600\U0001F680 are handled too."
)

# Pastel colours with dark text -> readable in both light and dark themes.
PALETTE = ["#FDE68A", "#BFDBFE", "#BBF7D0", "#FBCFE8", "#DDD6FE", "#FED7AA"]

GENERAL_EXAMPLE_WORDS = [
    "cat", "unbelievably", "tokenization", "internationalization",
    "Transformers", "electroencephalogram",
]

# Word-count method: a "word" is a run of letters/digits (Unicode aware),
# optionally joined by apostrophes or hyphens (e.g. don't, state-of-the-art).
WORD_PATTERN = re.compile(r"\w+(?:['\u2019-]\w+)*")

PAGE_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.app-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0;}
.app-subtitle {font-size: 1.05rem; opacity: .75; margin-bottom: 1.2rem;}
.panel {border: 1px solid rgba(128,128,128,.35); border-radius: 12px;
        padding: 14px 16px; line-height: 1.7; white-space: pre-wrap;
        word-break: break-word; max-height: 420px; overflow-y: auto;}
.tok {display: inline-flex; flex-direction: column; align-items: center;
      margin: 3px 2px; padding: 4px 8px; border-radius: 8px; color: #111;
      font-family: ui-monospace, Consolas, monospace; font-size: .92rem;
      line-height: 1.2; white-space: pre;}
.tok .i {font-size: .68rem; opacity: .65; margin-top: 2px;}
</style>
"""


# --------------------------------------------------------------------------- #
# Tokenizer + analysis helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading pretrained GPT-2 tokenizer...")
def load_tokenizer():
    """Load the real pretrained GPT-2 BPE tokenizer once and cache it."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def visible(text: str) -> str:
    """Make whitespace visible so it can be seen in tables and blocks."""
    return text.replace("\n", "\u21b5").replace("\t", "\u21e5").replace(" ", "\u00b7")


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def tokenize(tokenizer, text: str) -> dict:
    """Run real BPE tokenization and return aligned ids / tokens / decoded."""
    # add_special_tokens=False -> nothing is silently added. (GPT-2 adds none
    # by default anyway; being explicit keeps behaviour consistent.)
    ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    decoded = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in ids]
    return {"ids": ids, "tokens": tokens, "decoded": decoded}


def build_dataframe(result: dict) -> pd.DataFrame:
    rows = []
    for n, (tok, tid, dec) in enumerate(
        zip(result["tokens"], result["ids"], result["decoded"]), start=1
    ):
        # A byte-level token can be only part of a multi-byte character
        # (e.g. an emoji); decoding it alone gives the replacement character.
        shown = "<partial UTF-8 bytes>" if "\ufffd" in dec else visible(dec)
        rows.append({"Token Number": n, "Token": tok, "Token ID": tid,
                     "Decoded Token": shown})
    return pd.DataFrame(rows)


def tokens_to_html(result: dict) -> str:
    parts = []
    for i, (dec, tid) in enumerate(zip(result["decoded"], result["ids"])):
        label = "\u2bd1" if "\ufffd" in dec else visible(dec)
        color = PALETTE[i % len(PALETTE)]
        parts.append(
            f'<span class="tok" style="background:{color}" '
            f'title="Token #{i + 1}">'
            f'<span class="t">{html.escape(label)}</span>'
            f'<span class="i">{tid}</span></span>'
        )
    return f'<div class="panel">{"".join(parts)}</div>'


def split_examples(tokenizer, words) -> pd.DataFrame:
    """Show how individual words (with a leading space, as in running text) split."""
    rows = []
    for w in words:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        toks = tokenizer.convert_ids_to_tokens(ids)
        rows.append({"Word": w, "Subword tokens": " + ".join(toks),
                     "Token IDs": ", ".join(map(str, ids)), "# Tokens": len(ids)})
    return pd.DataFrame(rows)


def words_that_split(tokenizer, text: str, limit: int = 8):
    """Pick words from the user's text that BPE breaks into several tokens."""
    unique = {w for w in WORD_PATTERN.findall(text) if w.isalpha() and len(w) >= 5}
    scored = []
    for w in unique:
        n = len(tokenizer.encode(" " + w, add_special_tokens=False))
        if n > 1:
            scored.append((n, w))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [w for _, w in scored[:limit]]


# --------------------------------------------------------------------------- #
# Session-state callbacks
# --------------------------------------------------------------------------- #
def _load_sample():
    st.session_state.input_text = SAMPLE_TEXT
    st.session_state.analyzed_text = None


def _clear():
    st.session_state.input_text = ""
    st.session_state.analyzed_text = None


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def render_sidebar(tokenizer):
    with st.sidebar:
        st.header("Tokenizer Info")
        st.markdown(
            "- **Tokenizer:** GPT-2\n"
            "- **Algorithm:** Byte Pair Encoding\n"
            "- **Model:** pretrained GPT-2"
        )
        st.caption(f"Vocabulary size: {tokenizer.vocab_size:,}")
        st.divider()
        st.subheader("Special tokens")
        st.markdown(
            "GPT-2 has one special token, `<|endoftext|>` (ID 50256). "
            "This app encodes with `add_special_tokens=False`, so **no special "
            "tokens are added automatically** and none are counted unless you "
            "literally type `<|endoftext|>` in your text."
        )
        st.divider()
        st.subheader("Notes")
        st.markdown(
            "- `\u0120` in a token means *a space came before this piece*.\n"
            "- `\u00b7` = space, `\u21b5` = newline in the visual blocks.\n"
            "- Tokenization runs 100% locally."
        )


def render_metrics(text: str, result: dict):
    n_tokens = len(result["ids"])
    n_chars = len(text)
    n_words = count_words(text)
    avg = n_chars / n_tokens if n_tokens else 0.0

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Total Tokens", f"{n_tokens:,}"),
        (c2, "Total Words", f"{n_words:,}"),
        (c3, "Total Characters", f"{n_chars:,}"),
        (c4, "Avg Characters / Token", f"{avg:.2f}"),
    ]:
        with col.container(border=True):
            st.metric(label, value)
    st.caption(
        "**Word count method:** a word is a run of letters/digits, optionally "
        "joined by an apostrophe or hyphen (so *don't* and *state-of-the-art* "
        "count as one word each). Emojis and bare punctuation are not words. "
        "**Characters** include spaces and newlines."
    )


def render_side_by_side(text: str, result: dict):
    st.subheader("Original vs. Tokenized")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original paragraph**")
        st.markdown(f'<div class="panel">{html.escape(text)}</div>',
                    unsafe_allow_html=True)
    with right:
        st.markdown("**Tokens (colored blocks, token ID underneath)**")
        st.markdown(tokens_to_html(result), unsafe_allow_html=True)


def render_outputs(result: dict):
    st.subheader("Tokenized Output")
    st.caption("Tokens in their original order, separated by ` | `.")
    st.code(" | ".join(result["tokens"]), language=None)

    st.subheader("Token IDs")
    st.caption("The integer the model actually receives for each token above.")
    st.code(str(result["ids"]), language=None)


def render_table(result: dict):
    st.subheader("Token-by-Token Analysis")
    df = build_dataframe(result)
    st.dataframe(df, hide_index=True)
    st.download_button("Download table as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="token_analysis.csv", mime="text/csv")


def render_validation(tokenizer, text: str, result: dict):
    ids, tokens = result["ids"], result["tokens"]
    checks = {
        "Token count equals number of IDs": len(tokens) == len(ids),
        "Every token maps back to its ID": tokenizer.convert_tokens_to_ids(tokens) == ids,
        "Decoding all IDs reproduces the original text exactly":
            tokenizer.decode(ids, clean_up_tokenization_spaces=False) == text,
    }
    with st.expander("Validation checks", expanded=False):
        for name, ok in checks.items():
            st.write(("\u2705 " if ok else "\u274c ") + name)
        if len(ids) > 1024:
            st.warning("This text has more than 1,024 tokens, which exceeds "
                       "GPT-2's context window. Analysis still works.")


def render_education(tokenizer, text: str):
    st.subheader("How BPE Splits Words")
    st.markdown(
        "**Byte Pair Encoding** starts from single bytes/characters and "
        "repeatedly merges the most frequent adjacent pair found in training "
        "data into a new symbol. Common words become one token (`\u0120the`), "
        "while rare or long words are assembled from several smaller subword "
        "pieces. Nothing is ever 'unknown', because any text can fall back to "
        "bytes."
    )
    own = words_that_split(tokenizer, text)
    if own:
        st.markdown("**Words from your text that were split into several tokens:**")
        st.dataframe(split_examples(tokenizer, own), hide_index=True)
    st.markdown("**General examples:**")
    st.dataframe(split_examples(tokenizer, GENERAL_EXAMPLE_WORDS), hide_index=True)

    st.subheader("Why Token IDs Are Numbers")
    st.markdown(
        "Neural networks do math on numbers, not strings. The tokenizer holds "
        f"a fixed vocabulary of {tokenizer.vocab_size:,} pieces; each piece's "
        "position in that table is its **token ID**. The model turns each ID "
        "into an embedding vector (a row lookup) and processes those vectors. "
        "IDs carry no meaning by themselves; the same text always gives the "
        "same IDs for the same tokenizer, but IDs from different tokenizers "
        "are not interchangeable."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="BPE Tokenizer & Text Analyzer",
                       page_icon="\U0001F9E9", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown('<div class="app-title">BPE Tokenizer &amp; Text Analyzer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Explore how pretrained language '
                'models understand text.</div>', unsafe_allow_html=True)

    try:
        tokenizer = load_tokenizer()
    except Exception as exc:  # network down, bad install, etc.
        st.error(
            "Could not load the GPT-2 tokenizer. The first run needs internet "
            "access to download the tokenizer files from Hugging Face "
            "(afterwards they are cached locally).\n\n"
            f"Details: {type(exc).__name__}: {exc}"
        )
        st.stop()

    render_sidebar(tokenizer)

    st.session_state.setdefault("input_text", "")
    st.session_state.setdefault("analyzed_text", None)

    st.text_area("Enter or paste a paragraph", key="input_text", height=200,
                 placeholder="Type or paste any text here...")

    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    analyze = b1.button("Analyze Text", type="primary")
    b2.button("Sample Paragraph", on_click=_load_sample)
    b3.button("Clear", on_click=_clear)

    if analyze:
        text = st.session_state.input_text
        if not text.strip():
            st.session_state.analyzed_text = None
            st.warning("Please enter some text (or click 'Sample Paragraph') first.")
        else:
            st.session_state.analyzed_text = text

    text = st.session_state.analyzed_text
    if not text:
        st.info("Enter a paragraph and click **Analyze Text** to see the tokens.")
        return

    try:
        result = tokenize(tokenizer, text)
    except Exception as exc:
        st.error(f"Tokenization failed: {type(exc).__name__}: {exc}")
        return

    st.divider()
    render_metrics(text, result)
    st.divider()
    render_side_by_side(text, result)
    st.divider()
    render_outputs(result)
    st.divider()
    render_table(result)
    render_validation(tokenizer, text, result)
    st.divider()
    render_education(tokenizer, text)


if __name__ == "__main__":
    main()
