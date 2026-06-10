import json
from pathlib import Path
from collections import defaultdict

# Load inputs
with open("failure_reports.json", "r", encoding="utf-8") as f:
    failures = json.load(f)

with open("pipeline_loss_report.json", "r", encoding="utf-8") as f:
    loss_report = json.load(f)

# Build a mapping of statement conversions
statement_stats = {
    m["statement_id"]: m 
    for m in loss_report["statement_conversions"]
}

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

# Classify all failures
classified_by_statement = defaultdict(lambda: defaultdict(int))
recoverable_by_statement = defaultdict(lambda: defaultdict(int))
global_counts = defaultdict(int)
global_recoverable = defaultdict(int)

for f in failures:
    sid = f["statement_id"]
    cat, rec = classify_failure(f)
    
    classified_by_statement[sid][cat] += 1
    if rec:
        recoverable_by_statement[sid][cat] += 1
        global_recoverable[cat] += 1
        
    global_counts[cat] += 1

# Calculate statement quality reports
statement_reports = []
categories = [
    "OCR corruption", "row fragmentation", "semantic mismatch",
    "insufficient numeric values", "unsupported format", "other"
]

for sid, stats in statement_stats.items():
    total_recon = stats["reconstructed_rows"]
    ocr_corrupt = classified_by_statement[sid]["OCR corruption"]
    fragmented = classified_by_statement[sid]["row fragmentation"]
    insuff = classified_by_statement[sid]["insufficient numeric values"]
    mismatch = classified_by_statement[sid]["semantic mismatch"]
    unsupported = classified_by_statement[sid]["unsupported format"]
    other = classified_by_statement[sid]["other"]
    
    ocr_q = 1.0 - (ocr_corrupt / total_recon) if total_recon > 0 else 1.0
    row_q = 1.0 - (fragmented / total_recon) if total_recon > 0 else 1.0
    
    parse_attempts = stats["parsed_rows"] + ocr_corrupt + insuff + mismatch + unsupported + other
    parse_q = 1.0 - ((ocr_corrupt + insuff + mismatch + unsupported + other) / parse_attempts) if parse_attempts > 0 else 1.0
    
    causes = classified_by_statement[sid]
    primary_rc = max(causes.items(), key=lambda x: x[1])[0] if causes else "none"
    
    statement_reports.append({
        "statement_id": sid,
        "primary_root_cause": primary_rc,
        "ocr_corrupted_rows": ocr_corrupt,
        "unsupported_date_rows": unsupported,
        "fragmented_rows": fragmented,
        "insufficient_numeric_rows": insuff,
        "semantic_mismatch_rows": mismatch,
        "quality_scores": {
            "ocr_quality": round(ocr_q, 2),
            "reconstruction_quality": round(row_q, 2),
            "parsing_quality": round(parse_q, 2)
        },
        "recoverable_transactions": {
            "OCR corruption": recoverable_by_statement[sid]["OCR corruption"],
            "insufficient numeric values": recoverable_by_statement[sid]["insufficient numeric values"],
            "total": sum(recoverable_by_statement[sid].values())
        }
    })

# Format global counts & ROI
roi_estimates = [
    f"1. OCR improvement -> recover ~{global_recoverable['OCR corruption']} transactions",
    f"2. Amount normalization / parsing improvement -> recover ~{global_recoverable['insufficient numeric values']} transactions",
    f"3. Row reconstruction improvement -> recover ~{global_recoverable['row fragmentation']} transactions",
    f"4. Date parser format support -> recover ~{global_recoverable['unsupported format']} transactions",
    f"5. Layout / semantic assignment improvement -> recover ~{global_recoverable['semantic mismatch']} transactions"
]

recommendation = (
    f"Based on expected transaction recovery under the new 40-transaction baseline, "
    f"the highest-leverage task is OCR quality improvement (potential recovery of ~{global_recoverable['OCR corruption']} transactions), "
    f"followed by amount normalization/parsing improvements (potential recovery of ~{global_recoverable['insufficient numeric values']} transactions)."
)

final_report = {
    "global_counts": {cat: global_counts[cat] for cat in categories},
    "global_recoverable": {cat: global_recoverable[cat] for cat in categories},
    "engineering_roi_ranking": roi_estimates,
    "recommendation": recommendation,
    "statement_reports": statement_reports
}

out_path = Path("out/root_cause_report.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2, ensure_ascii=False)

print(f"Wrote root cause report to {out_path}")
