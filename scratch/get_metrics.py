import json
data = json.load(open('out/evaluation_report.json'))
targets = ['11','13','16','66','90','92','97']
for s in data['statement_metrics']:
    if s['statement_id'] in targets:
        print(
            f"stmt={s['statement_id']:3s} "
            f"parsed={s['parsed_row_count']:3d} "
            f"assigned={s['assigned_transaction_count']} "
            f"layout_conf={s['layout_detection_confidence']:.2f} "
            f"semantic_mismatch={s['semantic_mismatch_count']} "
            f"unknown={s['unknown_failure_count']} "
            f"amount_fail={s['amount_extraction_failed_count']}"
        )
