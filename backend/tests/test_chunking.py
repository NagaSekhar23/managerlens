from app.services.chunking import chunk_text


def test_short_text_produces_a_single_chunk():
    text = "Paragraph one.\n\nParagraph two."
    chunks = chunk_text(text, max_chars=800)
    assert chunks == ["Paragraph one.\n\nParagraph two."]


def test_splits_when_max_chars_exceeded():
    para_a = "A" * 500
    para_b = "B" * 500
    chunks = chunk_text(f"{para_a}\n\n{para_b}", max_chars=800)

    assert len(chunks) == 2
    assert chunks[0] == para_a
    assert chunks[1] == para_b


def test_keeps_paragraphs_intact_never_splits_mid_paragraph():
    paragraphs = [f"Paragraph {i} " + " ".join(["word"] * 20) for i in range(10)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, max_chars=300)

    # Every original paragraph appears whole in exactly one chunk.
    for para in paragraphs:
        assert sum(para in chunk for chunk in chunks) == 1


def test_ignores_blank_paragraphs():
    text = "Paragraph one.\n\n\n\nParagraph two."
    chunks = chunk_text(text, max_chars=800)
    assert chunks == ["Paragraph one.\n\nParagraph two."]


def test_empty_text_produces_no_chunks():
    assert chunk_text("", max_chars=800) == []


def test_single_paragraph_larger_than_max_chars_is_kept_whole():
    huge_paragraph = " ".join(["word"] * 1000)
    chunks = chunk_text(huge_paragraph, max_chars=800)
    assert chunks == [huge_paragraph]
