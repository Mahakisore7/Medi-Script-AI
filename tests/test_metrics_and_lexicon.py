import pytest

pytest.importorskip("rapidfuzz")

from mediscript.evaluation.metrics import (  # noqa: E402
    character_error_rate,
    evaluate_batch,
    exact_match,
    word_error_rate,
)
from mediscript.recognition.lexicon_trie import LexiconTrie  # noqa: E402


def test_cer_identical_is_zero():
    assert character_error_rate("amoxicillin", "amoxicillin") == 0.0


def test_cer_single_substitution():
    assert character_error_rate("amoxicilin", "amoxicillin") == pytest.approx(1 / 11)


def test_wer_and_exact_match():
    assert word_error_rate("tab ozad", "tab ozad 10") == pytest.approx(1 / 3)
    assert exact_match(" Ozad ", "ozad")


def test_evaluate_batch_empty():
    assert evaluate_batch([], [])["n_samples"] == 0


def test_lexicon_trie_lookup():
    trie = LexiconTrie(["Paracetamol", "Pantoprazole"])
    assert trie.is_valid_word("paracetamol")
    assert trie.is_valid_prefix("pan")
    assert not trie.is_valid_word("para")
    assert "r" in trie.get_allowed_next_characters("pa")
