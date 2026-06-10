import json
from collections import defaultdict

with open("failure_reports.json", "r", encoding="utf-8") as f:
    failures = json.load(f)

def classify_failure(failure: dict) -> tuple[str, bool]:
    # Returns (category, is_recoverable_transaction)
    ftype = failure["failure_type"]
    candidate = failure["candidate"]
    sid = failure["statement_id"]
    
    if ftype == "insufficient_numeric_values":
        # Check if it's metadata in Statement 25
        if sid == "25" and candidate in ("18-Dec-2006", "27-Jul-2017"):
            return "row fragmentation", False
        return "insufficient numeric values", True
        
    elif ftype == "date_parse_failed":
        # Statement 26 candidates are actual transaction rows where dates were mangled
        if sid == "26" and candidate in ("20 CASH", "20 BYCASH", "20 NEFL", "2 NEHL"):
            return "OCR corruption", True
        return "row fragmentation", False
        
    elif ftype == "no_date_detected":
        # Identify non-transaction lines
        exact_non_txns = {
            "opening balance", "opening balance 9y.0u", "tkansachontoal 462wu jzw",
            "closlngelance jw", "financial esl literacy toolkit", "liveworksheets",
            "estpac", "westpac choice", "statement of recent transactions",
            "jessica redpath account card number", "gosnells; wa_ 6110", "au",
            "transactions", "current balance: s850.38", "dale descrpllon dablt",
            "start balanee 600.201.ooo", "end balance 302!41.139"
        }
        
        prefix_non_txns = [
            "unless the constituent", "excludes the amount", "clanfications pleaee",
            "we would like", "trom you sharing", "cooyright", "rcf: sade", "kct:",
            "timed ut", "arrtnt ju2", "csni sort", "account 200", "rct: 305", "kf:", "kej:",
            "correrpndenee hshce", "ol424", "mseii ecom", "upvp_w/"
        ]
        
        cand_lower = candidate.lower().strip()
        is_non_txn = cand_lower in exact_non_txns
        if not is_non_txn:
            for pref in prefix_non_txns:
                if cand_lower.startswith(pref):
                    is_non_txn = True
                    break
        
        if is_non_txn:
            return "row fragmentation", False
        else:
            return "OCR corruption", True
            
    return "other", False

classified_data = []
for f in failures:
    cat, rec = classify_failure(f)
    classified_data.append({
        "statement_id": f["statement_id"],
        "failure_type": f["failure_type"],
        "candidate": f["candidate"],
        "category": cat,
        "recoverable": rec
    })

# Aggregate by statement and category
statement_summary = defaultdict(lambda: defaultdict(lambda: {"total": 0, "recoverable": 0}))
global_summary = defaultdict(lambda: {"total": 0, "recoverable": 0})

categories = ["OCR corruption", "row fragmentation", "semantic mismatch", "insufficient numeric values", "unsupported format", "other"]

for item in classified_data:
    sid = item["statement_id"]
    cat = item["category"]
    rec = item["recoverable"]
    
    statement_summary[sid][cat]["total"] += 1
    if rec:
        statement_summary[sid][cat]["recoverable"] += 1
        
    global_summary[cat]["total"] += 1
    if rec:
        global_summary[cat]["recoverable"] += 1

print("=== STATEMENTS SUMMARY ===")
for sid in sorted(statement_summary.keys()):
    print(f"\nStatement {sid}:")
    for cat in categories:
        stats = statement_summary[sid][cat]
        if stats["total"] > 0:
            print(f"  {cat:<30} | Total: {stats['total']:2d} | Recoverable: {stats['recoverable']:2d}")

print("\n=== STATEMENT 27 DETAILED CLASSIFICATION ===")
for item in classified_data:
    if item["statement_id"] == "27":
        print(f"  [{item['category']:<28}] Rec: {item['recoverable']} | Cand: {item['candidate']!r}")

print("\n=== GLOBAL SUMMARY ===")
for cat in categories:
    stats = global_summary[cat]
    print(f"  {cat:<30} | Total: {stats['total']:2d} | Recoverable: {stats['recoverable']:2d}")
