import json

def load_txns(state):
    txns = []
    for stmt_id, data in state.items():
        if stmt_id == "_summary": continue
        for t in data["transactions"]:
            txns.append({
                "statement_id": stmt_id,
                "raw_text": t["raw_text"],
                "date": t["date"],
                "amount": t["amount"]
            })
    return txns

with open("scratch/before_state.json") as f:
    before = json.load(f)
with open("scratch/after_state.json") as f:
    after = json.load(f)
    
before_txns = load_txns(before)
after_txns = load_txns(after)

# Find newly recovered transactions
new_txns = []
for a in after_txns:
    # See if this exact transaction exists in before
    found = False
    for b in before_txns:
        if a["statement_id"] == b["statement_id"] and a["date"] == b["date"] and a["amount"] == b["amount"]:
            found = True
            break
    if not found:
        new_txns.append(a)

print(f"Total newly recovered transactions: {len(new_txns)}")
print("---")
for t in new_txns:
    print(json.dumps(t, indent=2))
