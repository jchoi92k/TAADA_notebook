# TAADA - Tool for the Automatic analysis of Decoding Ability

TAADA is an NLP tool specifically designed to annotate and count lexical and sub-lexical features related to decoding in English. The features include metrics for grapheme, phoneme, and syllable counts, word frequency, contextual diversity, neighborhood effects, rhymes, and conditional probability.

## Download the App

**For most users:** Download the standalone executable at [linguisticanalysistools.org/taada.html](https://www.linguisticanalysistools.org/taada.html)

This repository is for demonstration purposes.

## Quick Start (GUI Application)

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

### Using the App
Steps for running TAADA are as follows

1. Open TAADA.
2. Choose files in the TAADA interface.
3. Choose level of analysis where you can select all words or content words and lemmatized words or original words.
4. Choose tests in which you are interested. These include basic decoding counts, conditional probability metrics, neighborhood effects (from the English Lexicon Project), word frequency (from COCA), and rhyme counts from PerDICT.
5. Enter the name of the .csv file you want to write.
6. Run the tests.

Refer to the [index description sheet](https://docs.google.com/spreadsheets/d/1YIecVwflmF0ik-gxyZbPap7QiK1qVWfq/edit?gid=1631969960#gid=1631969960) for more information on the features.

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