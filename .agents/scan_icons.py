import re, glob

bad = []
files = ['dashboard/app.py', 'dashboard/copilot_tab.py', 'dashboard/style.py']
files += glob.glob('modules/*.py')

for path in files:
    try:
        lines = open(path, encoding='utf-8').readlines()
        for i, line in enumerate(lines, 1):
            m = re.search(r'(?:page_icon|icon)\s*=\s*["\']([^"\']*)["\']', line)
            if m:
                val = m.group(1)
                if not val:
                    bad.append((path, i, 'EMPTY', line.strip()))
                    continue
                try:
                    val.encode('ascii')
                except UnicodeEncodeError:
                    codepoints = [ord(c) for c in val]
                    if any(0x80 <= cp < 0x1F000 for cp in codepoints):
                        bad.append((path, i, repr(val), line.strip()))
    except Exception as e:
        print(f'Error reading {path}: {e}')

if bad:
    print('POSSIBLE BAD ICONS:')
    for path, ln, val, line in bad:
        print(f'  {path}:{ln}  val={val}')
        print(f'    {line}')
else:
    print('All icon= values look clean')
