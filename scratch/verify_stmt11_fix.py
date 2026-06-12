"""Verify the Pattern 9 fix end-to-end for statement 11 row patterns."""
import sys
sys.path.insert(0, ".")

from src.constants import DATE_RE, MONEY_RE
from src.date_validator import validate_date_candidate
from src.normalizers.date_normalizer import normalize_date_strings
from src.transaction_reconstructor import Row

# Simulate the 8 genuine transaction rows from statement 11
sample_rows_text = [
    "01/12 054368486BKLGJ1J3 ACCELERATED SERVICES LL 126.13 63,913.66",
    "02/12 051404884LYR0Y8N1 RAMEY'S #8 BRANDON MS 73.83 63,987.49",
    "05/12 5543286885SRSSXJ4 PUBLIC STORAGE 08052 40.00 63,913.66",
    "06/12 55417348A4E5RBXMW C SPIRE MADISON MADISON MS 88.25 63,825.41",
    "15/12 05436848D8PKSBRHK ACCELERATED SERVICES 126.13 63,875.41",
    "17/12 55263528DRBGEHD1E POS RECURRING DEBIT ATM 19.80 63,855.61",
    "08/12 85586688H36HHRVS5 CITI ONLINE PAYMENT WEST 65.85 63,809.56",
    "20/12 85486148LWGPJDJX0 POS RECURRING DEBIT AT&T 69.55 63,740.01",
]

rows = [Row(y_center=float(i*30), tokens=[], text=t, original_text=t) for i, t in enumerate(sample_rows_text)]

print("BEFORE normalization:")
for r in rows:
    m = DATE_RE.search(r.text)
    print(f"  DATE_RE: {repr(m.group(0)) if m else 'NO MATCH':15}  row: {r.text[:70]}")

rows, count = normalize_date_strings(rows)

print()
print(f"AFTER normalization (normalized {count} dates):")
for r in rows:
    m = DATE_RE.search(r.text)
    amounts = MONEY_RE.findall(r.text)
    print(f"  DATE_RE: {repr(m.group(0)) if m else 'NO MATCH':15}  amounts={amounts[:2]}  row: {r.text[:70]}")

print()
# Check that validate_date_candidate accepts the canonicalized dates
print("VALIDATION CHECK:")
canonicalized_dates = []
rows2 = [Row(y_center=float(i*30), tokens=[], text=t, original_text=t) for i, t in enumerate(sample_rows_text)]
rows2, _ = normalize_date_strings(rows2)
for r in rows2:
    m = DATE_RE.search(r.text)
    if m:
        v, reason = validate_date_candidate(m.group(0))
        print(f"  date='{m.group(0)}'  valid={v}  reason={reason}")
    else:
        print(f"  NO DATE in: {r.text[:60]}")
