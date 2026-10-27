"""Suggest and apply spelling corrections to text."""

import argparse
import re
import sys

from spellchecker import SpellChecker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autocorrect misspelled words in text."
    )
    parser.add_argument("--text", help="Text to autocorrect.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for text when --text is not supplied.",
    )
    return parser.parse_args()


def tokenize(text):
    """Split text into words and separators without changing punctuation."""
    return re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)


def preserve_case(original_word, corrected_word):
    if original_word.isupper():
        return corrected_word.upper()
    if original_word.istitle():
        return corrected_word.capitalize()
    return corrected_word


def autocorrect_text(text, spell_checker=None):
    """Return corrected text and suggestions keyed by the misspelled word."""
    if spell_checker is None:
        spell_checker = SpellChecker()

    corrected_tokens = []
    suggestions = {}

    for token in tokenize(text):
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
    print("Original text:")
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

    if args.text is not None:
        text = args.text
    elif args.interactive:
        print("Autocorrect Keyboard System")
        try:
            text = input("Enter text: ")
        except EOFError:
            print("No input received.", file=sys.stderr)
            return 1
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Provide --text, pipe text to this command, or use --interactive.")
        return 2

    corrected_text, suggestions = autocorrect_text(text)
    print_results(text, corrected_text, suggestions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
