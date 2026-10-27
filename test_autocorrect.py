from autocorrect import autocorrect_text, preserve_case, tokenize


def test_corrects_sentence_from_readme():
    corrected, suggestions = autocorrect_text("I havv a beutiful hous")
    assert corrected == "I have a beautiful house"
    assert set(suggestions) == {"havv", "beutiful", "hous"}


def test_correct_text_is_unchanged_and_has_no_suggestions():
    corrected, suggestions = autocorrect_text("This sentence is correct.")
    assert corrected == "This sentence is correct."
    assert suggestions == {}


def test_punctuation_and_spacing_preserved():
    corrected, _ = autocorrect_text("Beutiful,  day!!")
    assert corrected == "Beautiful,  day!!"


def test_case_is_preserved():
    assert preserve_case("HAVV", "have") == "HAVE"
    assert preserve_case("Havv", "have") == "Have"
    assert preserve_case("havv", "have") == "have"


def test_tokenize_round_trips():
    text = "Hi, there -- 42 times!"
    assert "".join(tokenize(text)) == text


def test_suggestions_contain_best_correction_and_alternatives():
    _, suggestions = autocorrect_text("beutiful")
    assert suggestions["beutiful"]["best_correction"] == "beautiful"
    assert isinstance(suggestions["beutiful"]["alternatives"], list)


def test_empty_text():
    assert autocorrect_text("") == ("", {})
