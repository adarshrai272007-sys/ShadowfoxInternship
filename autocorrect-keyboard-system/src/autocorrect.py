"""
Autocorrect Keyboard System

This script identifies misspelled words and provides correction suggestions.

Interactive mode:
    python src/autocorrect.py

Command-line mode:
    python src/autocorrect.py --text "I havv a beutiful hous"
"""

import argparse
import re

from spellchecker import SpellChecker


def parse_args():
    parser = argparse.ArgumentParser(description="Autocorrect misspelled words in text.")
    parser.add_argument("--text", default=None, help="Text to autocorrect.")
    return parser.parse_args()


def tokenize(text):
    return re.findall(r"[A-Za-z]+|[^A-Za-z]", text)


def preserve_case(original_word, corrected_word):
    if original_word.isupper():
        return corrected_word.upper()
    if original_word[0].isupper():
        return corrected_word.capitalize()
    return corrected_word


def autocorrect_text(text):
    spell_checker = SpellChecker()
    tokens = tokenize(text)
    corrected_tokens = []
    suggestions = {}

    for token in tokens:
        if not token.isalpha():
            corrected_tokens.append(token)
            continue

        lower_token = token.lower()
        if lower_token in spell_checker:
            corrected_tokens.append(token)
            continue

        best_correction = spell_checker.correction(lower_token) or lower_token
        alternatives = sorted(spell_checker.candidates(lower_token) or [])

        suggestions[token] = {
            "best_correction": best_correction,
            "alternatives": alternatives[:5],
        }

        corrected_tokens.append(preserve_case(token, best_correction))

    return "".join(corrected_tokens), suggestions


def print_results(original_text, corrected_text, suggestions):
    print("\nOriginal text:")
    print(original_text)

    print("\nCorrected text:")
    print(corrected_text)

    print("\nSuggestions:")
    if not suggestions:
        print("No spelling mistakes found.")
        return

    for word, details in suggestions.items():
        print(
            f"- {word}: {details['best_correction']} "
            f"| alternatives: {details['alternatives']}"
        )


def main():
    args = parse_args()

    if args.text:
        text = args.text
    else:
        print("Autocorrect Keyboard System")
        text = input("Enter text: ")

    corrected_text, suggestions = autocorrect_text(text)
    print_results(text, corrected_text, suggestions)


if __name__ == "__main__":
    main()

