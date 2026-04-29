import re
import sys
import os

def improve_lint(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Standardize blank lines inside classes
    # Look for indented matches
    def fix_internal_blanks(match):
        indent = match.group(1)
        # Replace multiple blanks with just one blank line
        return f"\n{indent}"

    # Replace 2+ blank lines inside indented blocks with 1 blank line
    # Match: newline, then 1+ spaces, then 1+ lines of just whitespace, then another indented line
    content = re.sub(r'\n(\s+)\n(\s*\n)+(\s+)', r'\n\1\n\3', content)

    # 2. Fix blank lines between decorators and functions (already done but let's double check)
    content = re.sub(r'(@[a-zA-Z0-9_\.]+)\n\s*\n\s*(def )', r'\1\n    \2', content)

    # 3. Trim trailing whitespace
    lines = content.splitlines()
    lines = [line.rstrip() for line in lines]
    content = '\n'.join(lines) + '\n'

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            print(f"Improving lint for {arg}")
            improve_lint(arg)
