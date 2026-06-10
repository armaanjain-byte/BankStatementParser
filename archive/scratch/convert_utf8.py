with open("scratch/rows_output.txt", "r", encoding="utf-16-le") as f:
    content = f.read()
with open("scratch/rows_output_utf8.txt", "w", encoding="utf-8") as f:
    f.write(content)
