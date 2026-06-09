from pathlib import Path
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)

test_set_dir = Path("test_set")
output_dir = Path("outputs")

output_dir.mkdir(exist_ok=True)

for image_path in sorted(test_set_dir.glob("*.jpg")):
    print(f"Processing {image_path.name}...")

    result = ocr.predict(str(image_path))

    output_file = output_dir / f"{image_path.stem}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for page in result:
            for line in page["rec_texts"]:
                f.write(line + "\n")

    print(f"Saved -> {output_file}")

print("Done.")