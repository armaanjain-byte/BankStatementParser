import re

# The issue: '01/12/21' -> matches '12/21' (two-digit year trailing suffix)
# We need to also guard against the match occurring AFTER a digit+separator
# negative lookbehind: (?<!\d[/\-])
# 
# Also '12/21' is valid for our purposes (month=12, day=21)? But here it's from
# '01/12/21' which is a full date. The pattern fires on the SECOND slash group.
# 
# Solution: add a negative lookbehind to prevent matching if immediately preceded
# by a digit and separator:  (?<!\d/)  and  (?<!\d-)

PATTERN9 = re.compile(r'(?<!\d[/\-])\b(\d{1,2})[/\-](\d{1,2})(?![/\-]\d{2,4})\b')

tests_good = [
    '01/12',
    '02/12',
    '05/12',
    '15/12',
    '17/12',
    '20/12',
    '01/12 054368486BKLGJ1J3 ACCELERATED SERVICES',
    '17/12 55263528DRBGEHD1E POS RECURRING DEBIT',
]
tests_bad = [
    '7.87 64',
    '73.83 63',
    '63,987.49',
    '01/12/2021',
    '01/12/21',
    '01-12-2021',
    '01-12-21',
]

print('FINAL PATTERN 9:')
print('regex:', PATTERN9.pattern)
print()
print('Should MATCH:')
for t in tests_good:
    m = PATTERN9.search(t)
    print(f'  {repr(t[:50]):52} -> {repr(m.group(0)) if m else "NO MATCH"}')

print()
print('Should NOT MATCH:')
for t in tests_bad:
    m = PATTERN9.search(t)
    ok = "OK" if not m else "<-- BAD!"
    print(f'  {repr(t):25} -> {repr(m.group(0)) if m else "no match"} {ok}')
