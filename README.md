# Autocorrect Keyboard System

This project detects misspelled words in a sentence and provides autocorrection suggestions using Python.

## Objective

Develop an autocorrect keyboard system that identifies misspelled words in text input and provides correction suggestions.

The project includes:

- Text input from the user
- Word tokenization
- Misspelled word detection
- Best correction generation
- Alternative spelling suggestions
- Corrected sentence output

## Project Structure

```text
autocorrect-keyboard-system/
  README.md
  SUBMISSION.md
  requirements.txt
  src/
    autocorrect.py
  tests/
    test_autocorrect.py
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Project

Interactive mode:

```powershell
python src/autocorrect.py --interactive
```

Command-line mode:

```powershell
python src/autocorrect.py --text "I havv a beutiful hous"
```

Example output:

```text
Corrected text:
I have a beautiful house
```

## Features

- Preserves punctuation and spaces
- Gives best correction
- Shows alternative suggestions
- Handles uppercase words at the beginning of sentences


## Input Modes

```powershell
python src/autocorrect.py --text "I havv a beutiful hous"
python src/autocorrect.py --interactive
echo "I havv a beutiful hous" | python src/autocorrect.py
```

## Limitations

Suggestions are ranked by word frequency without looking at the neighbouring words, so some corrections can be wrong (for example `quik` may become `quit`). The alternative suggestions are shown so the user can pick the right word.

## Tests

```powershell
python -m pytest -q
```
