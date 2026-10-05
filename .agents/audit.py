import re, glob

# 1. List all tabs
with open('dashboard/app.py', encoding='utf-8') as f:
    src = f.read()

m = re.search(r'st\.tabs\(\[(.*?)\]\)', src, re.DOTALL)
if m:
    entries = [x.strip().strip('"\'') for x in m.group(1).split(',')]
    print(f"TABS ({len(entries)} total):")
    for i, e in enumerate(entries):
        print(f"  [{i:2d}] {e}")
else:
    print("No st.tabs found")

print()

# 2. Scan for fig.show() remnants
print("fig.show() scan:")
found_any = False
for fpath in sorted(glob.glob('modules/*.py') + ['dashboard/app.py', 'dashboard/copilot_tab.py']):
    try:
        content = open(fpath, encoding='utf-8').read()
        count = content.count('fig.show()')
        if count:
            print(f"  FOUND {count}x in {fpath}")
            found_any = True
    except Exception as e:
        print(f"  ERROR reading {fpath}: {e}")
if not found_any:
    print("  Clean — no fig.show() calls outside __main__ blocks")

print()

# 3. Check fig.show() only appears inside __main__ blocks
print("fig.show() in __main__ blocks only check:")
for fpath in ['modules/eda.py', 'modules/time_series.py', 'modules/pareto.py', 'modules/models.py', 'modules/explainability.py']:
    try:
        lines = open(fpath, encoding='utf-8').readlines()
        in_main = False
        bad_lines = []
        for i, line in enumerate(lines, 1):
            if '__main__' in line:
                in_main = True
            if 'fig.show()' in line and not in_main:
                bad_lines.append(i)
        if bad_lines:
            print(f"  BAD: {fpath} has fig.show() outside __main__ at lines {bad_lines}")
        else:
            print(f"  OK: {fpath}")
    except Exception as e:
        print(f"  ERROR: {fpath}: {e}")

print()

# 4. Check requirements.txt key packages
print("requirements.txt check:")
with open('requirements.txt', encoding='utf-8') as f:
    reqs = f.read()
for pkg in ['openai', 'lifetimes', 'scikit-learn', 'plotly', 'streamlit']:
    status = 'PRESENT' if pkg in reqs else 'MISSING'
    print(f"  {pkg}: {status}")
