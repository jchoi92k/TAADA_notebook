# TAADA - Tool for the Automatic analysis of Decoding Ability

TAADA is an NLP tool specifically designed to annotate and count lexical and sub-lexical features related to decoding in English. The features include metrics for grapheme, phoneme, and syllable counts, word frequency, contextual diversity, neighborhood effects, rhymes, and conditional probability.

## Download the App

**For most users:** Download the standalone executable at [linguisticanalysistools.org/taada.html](https://www.linguisticanalysistools.org/taada.html)

This repository is for demonstration purposes.

## Quick Start (GUI Application)

### Requirements
- Python 3.8+
- Dependencies: `spacy`, `PyQt5`, `pandas`, `numpy`
- Required data files:
  - `decoding_1_dataframe.csv` (download required)
  - `pronounce_dic_tran.pkl`

### Installation

```bash
# Create virtual environment
uv venv taada-env
source taada-env/bin/activate  # On Windows: taada-env\Scripts\activate.bat

# Install dependencies
uv pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Run

```bash
python taada_app_main.py
```

## Jupyter Notebook

The `calculate_cond_prob_final_for_github_clean.ipynb` notebook demonstrates the analysis workflow and calculations used in the GUI application.

## Features

Analyzes text at multiple levels:
- All words (lemmatized/unlemmatized)
- Content words only (lemmatized/unlemmatized)

Computes metrics including:
- Basic counts (syllables, letters, phonemes)
- Conditional probability
- Neighborhood effects (orthographic, phonological)
- Word frequency (SUBTLEX-US, COCA)
- Rhyme patterns