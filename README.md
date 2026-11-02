# BPE Tokenizer & Text Analyzer

A Streamlit dashboard that shows how the **pretrained GPT-2 Byte Pair Encoding (BPE) tokenizer**
splits a paragraph into tokens and token IDs. Runs locally: no LLM API, no API key.

## Setup (Windows PowerShell)

```powershell
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it (if blocked, first run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The first launch downloads the small GPT-2 tokenizer files from Hugging Face (internet needed once);
they are cached afterwards.

## Features
- Text area, **Sample Paragraph**, **Clear**, and **Analyze Text** buttons
- Real GPT-2 BPE via `AutoTokenizer.from_pretrained("gpt2")`, `encode()` and `convert_ids_to_tokens()`
- Metric cards: tokens, words, characters, average characters per token
- Colored token blocks with IDs, original-vs-tokenized view, token table (CSV download)
- Validation checks (count match, token-to-ID match, lossless round trip)
- Educational section: how BPE splits words and why IDs are numbers

## Special tokens
Encoding uses `add_special_tokens=False`. GPT-2 adds none by default, so the results contain only
tokens that come from your text. Typing the literal string `<|endoftext|>` yields the special ID 50256.

## Word count method
A word is a run of letters/digits, optionally joined by an apostrophe or hyphen. Emojis and bare
punctuation are not words.

## Structure
```
bpe_text_analyzer/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
