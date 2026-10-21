# AI News Detector

An application for automatically recognizing AI-generated news and human-written news.

## Table of Contents

- [Methodology](#methodology)
- [Tech Stack](#tech-stack)
- [Installation and Launch](#installation-and-launch)

---

## Methodology

The chosen methodological framework is a **classical machine learning approach based on explicit feature extraction**, rather than an end-to-end neural network classifier.

The input text is not fed into a neural encoder. Instead, it is transformed into a fixed-size **32-dimensional numerical vector** consisting of:

- statistical metrics
- lexical metrics
- morphological metrics
- stylistic metrics

This vector is then used as input for a **LightGBM** gradient-boosted decision tree model.

```
Input text  →  Feature extraction (32-dim vector)  →  LightGBM  →  AI-generated / Human-written
```

---

## Tech Stack

All software dependencies are described in [`requirements.txt`](requirements.txt). The project is divided into six functional groups:

| Group | Libraries |
|-------|-----------|
| **Machine learning and data analysis** | LightGBM, scikit-learn, NumPy, pandas |
| **Text processing and parsing** | `re`, `collections`, BeautifulSoup, feedparser |
| **Web development** | Flask, requests, `concurrent.futures` |
| **Frontend** | Bootstrap 5, Bootstrap Icons (via CDN) |
| **LLM integration** | requests (for the HTTP APIs of Ollama, GigaChat and DeepSeek), tqdm |
| **Python standard library** | `pickle`, `pathlib`, `argparse`, `csv`, `sys`, `os`, `time`, `random`, `uuid` |

---

## Installation and Launch

**1. Clone the repository**

```bash
git clone https://github.com/sWAyaK/An-app-for-automatically-recognizing-AI-generated-news-and-human-written-news/
cd An-app-for-automatically-recognizing-AI-generated-news-and-human-written-news
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
python web_app.py
```

Then open the address shown in the terminal (usually `http://127.0.0.1:5000`) in your browser.
