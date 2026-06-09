"""Check for duplicate token texts at different y positions."""
import json
data = json.load(open("outputs/28_ocr.json"))
tokens = data["tokens"]

dupes = {}
for t in tokens:
    y = (t["bbox"][1] + t["bbox"][3]) // 2
    dupes.setdefault(t["text"], []).append(y)

print("Tokens with identical text appearing at multiple y-positions:")
print("(These cause false positives in substring-based y-spread analysis)")
print()
for text, positions in dupes.items():
    if len(positions) > 1:
        print(f'  "{text}" -> y-positions: {positions}')
