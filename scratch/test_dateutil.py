from dateutil import parser
import datetime

for t in ['01/12', '02/12', '15/12', '17/12', '20/12']:
    try:
        d = parser.parse(t, fuzzy=True)
        print(repr(t), '->', d.strftime('%Y-%m-%d'))
    except Exception as e:
        print(repr(t), '-> ERROR:', e)
