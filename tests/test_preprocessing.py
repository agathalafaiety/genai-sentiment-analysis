from sentiment_analysis.preprocessing import combine_review_text, normalize_text


def test_normalize_preserves_accents_and_negation() -> None:
    result = normalize_text("  NÃO gostei! Veja https://example.com  ")
    assert result == "não gostei! veja url"


def test_combine_ignores_missing_values() -> None:
    assert combine_review_text("Ótimo", None) == "ótimo"
    assert combine_review_text(float("nan"), "Chegou bem") == "chegou bem"
