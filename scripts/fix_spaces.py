import re
import sys
import os

def fix_spaces(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Normalize all 3+ newlines to 2 newlines (top level)
    # content = re.sub(r'\n{4,}', '\n\n\n', content) # 2 blank lines max
    
    # Surgical fix for E303: 3+ newlines within an indented block or between docstrings and defs
    # Case 1: Multiple blank lines before an indented def
    content = re.sub(r'\n\s*\n\s*\n(\s+def )', r'\n\n\1', content)
    
    # Case 2: Multiple blank lines after a docstring before a def
    content = re.sub(r'("""\s*)\n\s*\n\s*\n(\s+def )', r'\1\n\n\2', content)

    # Case 3: 3+ blank lines at top level -> 2 blank lines (PEP8)
    content = re.sub(r'\n\n\n\n+', r'\n\n\n', content)

    # Fix continuation lines (E128) - this is harder with regex but let's try some common patterns
    # Like in update_auto_switch_criteria log
    # self.logger.info(f"..."
    #                 f"...")
    # Should be indented +4 or aligned.
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            print(f"Fixing spaces for {arg}")
            fix_spaces(arg)
