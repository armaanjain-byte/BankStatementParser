from archive.ocr import extract_tokens

print("=== WITHOUT PREPROCESSING (NoPre) ===")
tok_no_pre = extract_tokens("test_set/24.jpg", gpu=False, preprocess=False, min_confidence=0.0)
for t in tok_no_pre[:15]:
    print(f"  Conf: {t.confidence:.2f} | Text: {repr(t.text)}")

print("\n=== WITH PREPROCESSING (Pre) ===")
tok_pre = extract_tokens("test_set/24.jpg", gpu=False, preprocess=True, min_confidence=0.0)
for t in tok_pre[:15]:
    print(f"  Conf: {t.confidence:.2f} | Text: {repr(t.text)}")
