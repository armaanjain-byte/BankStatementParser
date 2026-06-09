import json
import re
from pathlib import Path
from collections import defaultdict

def classify_failure(failure: dict) -> str:
    ftype = failure.get("failure_type", "")
    candidate = failure.get("candidate", failure.get("raw_text", ""))
    
    if ftype == "insufficient_numeric_values":
        return "INSUFFICIENT_NUMERIC_VALUES"
    elif ftype == "semantic_mismatch":
        return "SEMANTIC_MISMATCH"
    elif ftype in ("no_date_detected", "date_parse_failed"):
        # Check unsupported dates
        if re.search(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b', candidate):
            return "DATE_FORMAT_UNSUPPORTED"
            
        # Check specific OCR corruptions
        if (re.search(r'[&!@\[\]_]', candidate) or 
            re.search(r'\d{1,2}[-\.]\w{1,3}[-\.]20[a-zA-Z0-9]{2}', candidate) or
            re.search(r'2W2|202L|0AL204|4k24|0x4L2Q2L', candidate)):
            return "OCR_CORRUPTION"
            
        if re.search(r'(?:gh|Ist|11").*?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', candidate, re.I):
            return "OCR_CORRUPTION"
            
        # Fragmentation indicators
        if not re.search(r'\d', candidate):
            return "ROW_FRAGMENTATION"
            
        if "UPVP" in candidate or "UPVF" in candidate or "NEFT" in candidate:
            # If it has a mangled date in it
            if re.search(r'\d{1,2}[-\.]\w{1,3}[-\.]20', candidate) or re.search(r'\d{2}-\d{3}', candidate):
                return "OCR_CORRUPTION"
            # Otherwise it's just a dangling description line without a date
            return "ROW_FRAGMENTATION"
            
        if len(candidate) < 15:
            return "ROW_FRAGMENTATION"
            
        # Fallback for date failures is OCR corruption based on user hint
        return "OCR_CORRUPTION"
    else:
        return "OTHER"

def main():
    failures_file = Path("out/failure_reports.json")
    loss_file = Path("out/pipeline_loss_report.json")
    
    if not failures_file.exists() or not loss_file.exists():
        print("Required input files not found.")
        return
        
    with open(failures_file, "r", encoding="utf-8") as f:
        failures = json.load(f)
        
    with open(loss_file, "r", encoding="utf-8") as f:
        loss_report = json.load(f)
        
    statement_stats = {
        m["statement_id"]: m 
        for m in loss_report["statement_conversions"]
    }
        
    # 1. Reclassify
    reclassified = defaultdict(lambda: defaultdict(int))
    global_counts = defaultdict(int)
    
    # We may have duplicate reports in failure_reports if the script was run multiple times.
    # To avoid double counting, we'll only count the latest ones or just clear out duplicates.
    # Actually, failure_reports might have old ones because we didn't clear it.
    # Let's deduplicate by candidate and statement_id, or just assume the last `total_failures` match.
    # Wait, evaluate_dataset.py appends to it? Yes.
    # Let's only take the last N failures, where N is sum of failures in loss_report.
    total_failures_in_loss = sum(loss_report["ranked_failure_categories"].values())
    recent_failures = failures[-total_failures_in_loss:] if total_failures_in_loss > 0 else []
    
    if not recent_failures and failures:
        # fallback
        recent_failures = failures

    for fail in recent_failures:
        sid = fail["statement_id"]
        rc = classify_failure(fail)
        reclassified[sid][rc] += 1
        global_counts[rc] += 1
        
    # 2. Quality Scores
    # OCR quality: 1.0 - (ocr_corrupted_rows / reconstructed_rows)
    # Row reconstruction quality: 1.0 - (fragmented_rows / reconstructed_rows)
    # Parsing quality: 1.0 - (unsupported + insufficient + mismatch) / parsed_rows ... wait, parsed_rows is AFTER they pass.
    # So denominator for parsing should be (parsed_rows + those failures).
    
    report_per_statement = []
    
    for sid, stats in statement_stats.items():
        total_recon = stats["reconstructed_rows"]
        ocr_corrupt = reclassified[sid]["OCR_CORRUPTION"]
        fragmented = reclassified[sid]["ROW_FRAGMENTATION"]
        
        unsupported = reclassified[sid]["DATE_FORMAT_UNSUPPORTED"]
        insuff = reclassified[sid]["INSUFFICIENT_NUMERIC_VALUES"]
        mismatch = reclassified[sid]["SEMANTIC_MISMATCH"]
        
        ocr_q = 1.0 - (ocr_corrupt / total_recon) if total_recon > 0 else 0.0
        row_q = 1.0 - (fragmented / total_recon) if total_recon > 0 else 0.0
        
        parse_attempts = stats["parsed_rows"] + unsupported + insuff + mismatch
        parse_q = 1.0 - ((unsupported + insuff + mismatch) / parse_attempts) if parse_attempts > 0 else 1.0
        
        # Primary root cause for this statement
        causes = reclassified[sid]
        primary_rc = max(causes.items(), key=lambda x: x[1])[0] if causes else "NONE"
        
        report_per_statement.append({
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
            }
        })
        
    # 3. Ranked ROI
    # We estimate recoverability: 
    # If we fix OCR, we get OCR_CORRUPTION rows back.
    # If we fix row reconstruction, we get ROW_FRAGMENTATION back.
    # If we fix date parser, we get DATE_FORMAT_UNSUPPORTED back.
    # If we fix layout/semantic, we get SEMANTIC_MISMATCH back.
    
    roi_estimates = {
        "OCR improvement": global_counts["OCR_CORRUPTION"],
        "Row reconstruction improvement": global_counts["ROW_FRAGMENTATION"],
        "Date parser improvement": global_counts["DATE_FORMAT_UNSUPPORTED"],
        "Amount extraction / Semantic improvement": global_counts["INSUFFICIENT_NUMERIC_VALUES"] + global_counts["SEMANTIC_MISMATCH"]
    }
    
    ranked_roi = sorted(roi_estimates.items(), key=lambda x: x[1], reverse=True)
    roi_output = [f"{idx+1}. {k} -> recover ~{v} rows" for idx, (k, v) in enumerate(ranked_roi)]
    
    recommendation = f"Based on expected transaction recovery, the highest-leverage next engineering task is: {ranked_roi[0][0]}."
    
    final_report = {
        "global_failure_counts": dict(global_counts),
        "engineering_roi_ranking": roi_output,
        "recommendation": recommendation,
        "statement_reports": report_per_statement
    }
    
    out_path = Path("out/root_cause_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
        
    print(f"Wrote root cause analysis to {out_path}")

if __name__ == "__main__":
    main()
