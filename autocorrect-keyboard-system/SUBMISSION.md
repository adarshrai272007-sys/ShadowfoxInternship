# Autocorrect Keyboard System

## Objective

The objective of this project is to develop an autocorrect keyboard system that detects misspelled words in text input and provides correction suggestions.

## Implementation Approach

The project uses the `pyspellchecker` library in Python. The system accepts a text sentence, checks each word, finds misspelled words, and replaces them with the most likely correct word.

## Key Components

### Spell Checker

The spell checker identifies words that are not present in the dictionary and generates possible corrections.

### Tokenization

The input sentence is split into word and non-word tokens. This helps preserve punctuation, spaces, and sentence structure.

### Correction Suggestions

For each misspelled word, the system provides:

- Best correction
- Alternative suggestions

## How It Works

1. The user enters a sentence.
2. The sentence is tokenized.
3. Each word is checked using the spell checker.
4. Misspelled words are corrected.
5. The corrected sentence and suggestions are displayed.

## Example

Input:

```text
I havv a beutiful hous
```

Output:

```text
I have a beautiful house
```

## Benefits

- Improves text input accuracy
- Reduces manual correction effort
- Can be extended into a keyboard or chat application

## Conclusion

This project demonstrates a beginner-friendly natural language processing application. It identifies spelling mistakes and automatically suggests corrections, improving user typing accuracy.

