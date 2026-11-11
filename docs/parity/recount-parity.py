"""Update summary counts from the capability tables."""
import collections
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
source = path.read_text()
areas = {}
statuses = {}
area = None
tier = None
tier_order = ('core', 'addons', 'third-party')
status_order = ('✅', '🟡', '❌')
platform_rows = 0

for line in source.splitlines():
    if line == '## Platform differences':
        break
    section = re.fullmatch(r'## .+ \(\x60([a-z-]+)\x60\)', line)
    if section:
        area = section[1]
        areas[area] = collections.Counter()
        tier = None
    for prefix, value in (
        ('### Core', 'core'),
        ('### Official addons', 'addons'),
        ('### Third-party', 'third-party'),
    ):
        if line.startswith(prefix):
            tier = value
    if not area or not tier:
        continue
    row = re.match(r'^\|\s*([^|]+?)\s*\|.*\|\s*(✅|🟡|❌|🌐)\s*\|', line)
    if not row:
        continue
    label, status = row.groups()
    key = (area, tier, label)
    if key in statuses:
        raise ValueError(f'Duplicate capability row {key}')
    statuses[key] = status
    if status == '🌐':
        platform_rows += 1
    else:
        areas[area][tier, status] += 1

assert len(areas) == 16
assert platform_rows == 10
totals = collections.Counter()
for area, counts in areas.items():
    totals.update(counts)
    summary = ' | '.join(str(counts[tier, status]) for tier in tier_order for status in status_order)
    source, replacements = re.subn(
        rf'^\| \[`{area}`\]\(#{area}\) \|.*$',
        f'| [`{area}`](#{area}) | {summary} |', source, flags=re.M,
    )
    assert replacements == 1, area

summary = ' | '.join(f'**{totals[tier, status]}**' for tier in tier_order for status in status_order)
source, replacements = re.subn(r'^\| \*\*Total\*\* \|.*$', f'| **Total** | {summary} |', source, flags=re.M)
assert replacements == 1
for (area, tier, label), status in statuses.items():
    if status == '🌐':
        continue
    source = re.sub(
        rf'^(  - )(?:✅|🟡|❌)( {re.escape(label)} \(`{area}`, {tier}\))$',
        rf'\g<1>{status}\g<2>', source, flags=re.M,
    )
path.write_text(source)
print(f'{len(statuses)} capability rows. {summary}')
