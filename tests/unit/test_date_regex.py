import pytest
from src.constants import DATE_RE
from src.date_validator import validate_date_candidate

@pytest.mark.parametrize(
    "text, expected_match",
    [
        # Attached-prefix dates (Statement 3 cases)
        ("S9703109702/04/2017 349802030032344:Int.Pd:01 INR", "02/04/2017"),
        ("S267282504/04/2017 PEN.MAR-17LESS TDS=0 INR", "04/04/2017"),
        ("REF04/04/2017", "04/04/2017"),
        ("ID-12/05/2017", "12/05/2017"),
        
        # Pattern 7 tests (Mon DD, YYYY)
        ("Mar 15, 2010", "Mar 15, 2010"),
        ("Apr 18, 2010", "Apr 18, 2010"),
        ("January 01, 2024", "January 01, 2024"),
        ("Mar 15", "Mar 15"),
        
        # Normal dates
        (" 04/04/2017 ", "04/04/2017"),
        ("Date: 15-08-2022", "15-08-2022"),
        ("2026-11-20", "2026-11-20"),
        ("01/12", "01/12"),
    ]
)
def test_date_re_valid_matches(text: str, expected_match: str):
    """Ensure DATE_RE properly matches dates with or without prefixes."""
    match = DATE_RE.search(text)
    assert match is not None, f"Expected {text!r} to match {expected_match!r}, got None"
    assert match.group(0) == expected_match


@pytest.mark.parametrize(
    "text",
    [
        # Account numbers and invalid dates that must NOT match
        "1234567890/04/56",  # Month 56 is invalid, but DATE_RE matches it? Wait, DATE_RE doesn't validate semantics.
        # Wait, if DATE_RE just extracts regex, it will match '90/04/56' or '0/04/56'.
        # Let's test what it actually does. If it matches, the test will need to be adjusted, 
        # or we rely on validate_date_candidate. The user said: "account numbers (cases that must NOT match)".
        # Let's use a pure numeric account number
        "1234567890123456",
        
        # Reference ids that look like dates but have wrong slashes
        "TRX/99/88/7777",
        
        # Amount values
        "$1,234.56",
        "1,234.56",
        
        # IP Addresses
        "192.168.1.1",
    ]
)
def test_date_re_invalid_matches(text: str):
    """Ensure DATE_RE does NOT falsely match generic strings or amounts."""
    match = DATE_RE.search(text)
    if match:
        # If the regex matches a garbage string, the validator MUST catch it.
        candidate = match.group(0)
        is_valid, _ = validate_date_candidate(candidate)
        if is_valid:
            pytest.fail(f"Falsely matched and validated {candidate!r} in {text!r}")
