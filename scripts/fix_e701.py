import re
import sys
import os

def fix_e701(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Match "if cond: stmt" where stmt starts with a letter or quote
        # Exclude docstrings or comments
        match = re.match(r'^(\s+if\s+.+:)\s+([a-zA-Z\d\'"\{].+)$', line)
        if match:
            indent = match.group(1).split('if')[0]
            if_part = match.group(1)
            stmt_part = match.group(2)
            new_lines.append(f"{if_part}\n{indent}    {stmt_part}\n")
        else:
            new_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            print(f"Fixing E701 for {arg}")
            fix_e701(arg)
