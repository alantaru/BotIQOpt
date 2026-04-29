import re
import sys
import os

def fix_indent(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    in_class = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Detect start of a class (always top level in these files)
        if line.startswith('class '):
            in_class = True
            new_lines.append(line)
            i += 1
            continue
            
        # Stop "in_class" if we see something that is definitely not a method and at top level
        # Actually, in these files, almost everything is in a class after the imports.
        
        if in_class:
            # Fix decorators unindented
            if line.startswith('@'):
                decorator = line.strip()
                j = i + 1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                if j < len(lines) and (lines[j].startswith('def ') or lines[j].startswith('    def ')):
                    new_lines.append('    ' + decorator + '\n')
                    # Now we MUST handle the def line next
                    def_line = lines[j]
                    if def_line.startswith('def '):
                        new_lines.append('    ' + def_line)
                    else:
                        new_lines.append(def_line)
                    i = j + 1
                    continue
            
            if line.startswith('def '):
                new_lines.append('    ' + line)
                i += 1
                continue

        new_lines.append(line)
        i += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            print(f"Fixing indentation in {arg}")
            fix_indent(arg)
