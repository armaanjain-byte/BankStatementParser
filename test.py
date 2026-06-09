from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=False, lang="en")

result = ocr.ocr("test_set/23.jpg", cls=False)

print(result[:5])