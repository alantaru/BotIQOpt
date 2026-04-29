import re
import sys
import os

def cleanup_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove trailing whitespace from each line
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

    # 2. Fix multiple blank lines (max 2)
    content = re.sub(r'\n\n\n+', '\n\n', content)

    # 3. Ensure 2 blank lines before classes and functions (E302)
    # This is a bit complex with regex, but let's try some common cases
    content = re.sub(r'([^\n])\n(class|def) ', r'\1\n\n\2 ', content)
    content = re.sub(r'(\n\nclass|def) ', r'\n\n\1 ', content) # Ensure at least two

    # 4. Remove blank lines with whitespace (W293) - already handled by #1

    # 5. Fix at least two spaces before inline comment (E261)
    content = re.sub(r'([^ ]) #', r'\1  #', content)
    content = re.sub(r'  #', r'  #', content) # Ensure it's exactly 2 or more (already handled)

    # 6. Fix multiple statements on one line (E701) - partial fix
    # e.g. if x: return y -> if x:\n    return y
    # This is risky, let's skip for now unless it's very common.

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            print(f"Cleaning up {arg}")
            cleanup_file(arg)
