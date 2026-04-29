import re
import sys
import os

def final_cleanup(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove blank lines between decorators and def
    # Example: @property\n\n    def ... -> @property\n    def ...
    # Support both 0 and 4 spaces indentation for the decorator
    content = re.sub(r'(@[a-zA-Z0-9_\.]+)\n\s*\n\s*(def )', r'\1\n    \2', content)
    
    # 2. Fix indented @ decorators that should be indented but have extra blank lines
    # This specifically addresses what I saw in Ferramental.py
    # 162:     @property
    # 163:     
    # 164: 
    # 165:     def simulation_mode(self) -> bool:
    content = re.sub(r'(\s+)(@[a-zA-Z0-9_\.]+)\n\s*\n\s*\n\s+(def )', r'\1\2\n\1\3', content)

    # 3. Remove cases where there are multiple blank lines between methods
    content = re.sub(r'\n\s*\n\s*\n\s*\n+', '\n\n\n', content)

    # 4. Remove blank lines at the start of functions (after def line)
    content = re.sub(r'(def .+:)\n\s*\n', r'\1\n', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            print(f"Final cleanup for {arg}")
            final_cleanup(arg)
