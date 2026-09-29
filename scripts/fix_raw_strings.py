# fix_raw_strings.py
import re
import os

for fname in ["generate_progressive_notes.py", "build_all_progressive_guides.py"]:
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Replace PROGRESSIVE_GUIDES[X] = """ with PROGRESSIVE_GUIDES[X] = r"""
        text = re.sub(r'PROGRESSIVE_GUIDES\[(\d+)\]\s*=\s*"""', r'PROGRESSIVE_GUIDES[\1] = r"""', text)
        
        with open(fname, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Fixed raw strings in {fname}")
