# Output JSON Schema

The ultimate goal of the `BankStatementParser` is to translate unstructured PDF and image bank statements into a standardized, machine-readable JSON format.

Every file generated in the `out/statement_outputs/` directory adheres strictly to the following schema.

## Root Object (`BankStatement`)

The root JSON object represents a single parsed bank statement.

```json
{
  "metadata": {
    "bank_name": "String | null",
    "account_holder": "String | null",
    "account_number": "String | null",
    "statement_period_from": "String | null",
    "statement_period_to": "String | null",
    "currency": "String | null"
  },
  "transactions": [
    // Array of Transaction objects
  ],
  "validation": {
    "passed": "Boolean",
    "confidence": "Float (0.0 to 1.0)",
    "errors": [
      "String (List of validation failure reasons)"
    ]
  }
}
```

## `Transaction` Object

Represents a single horizontal transaction row successfully extracted from the statement table. 

```json
{
  "date": "String | null",
  "description": "String | null",
  "amount": "Float | null",
  "debit": "Float | null",
  "credit": "Float | null",
  "balance": "Float | null"
}
```
*Note: Due to the varying designs of bank statements (e.g. single-column amounts vs explicit debit/credit columns), fields like `amount`, `debit`, and `credit` are mutually exclusive depending on the template. Unparsed fields remain `null`.*

---

## Sample Output

Below is an abbreviated example of a successfully parsed statement output (`70_statement.json`):

```json
{
  "metadata": {
    "bank_name": "HDFC Bank",
    "account_holder": "JOHN DOE",
    "account_number": "123456789",
    "statement_period_from": "01/01/2021",
    "statement_period_to": "31/01/2021",
    "currency": null
  },
  "transactions": [
    {
      "date": "01/01/2021",
      "description": "ATM WITHDRAWAL - MAIN ST",
      "amount": null,
      "debit": 5000.0,
      "credit": null,
      "balance": 45000.0
    },
    {
      "date": "05/01/2021",
      "description": "SALARY CREDIT - Q1",
      "amount": null,
      "debit": null,
      "credit": 50000.0,
      "balance": 95000.0
    }
  ],
  "validation": {
    "passed": true,
    "confidence": 0.95,
    "errors": []
  }
}
```
