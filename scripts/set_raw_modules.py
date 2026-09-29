# set_raw_modules.py
import re

for p in range(1, 7):
    pnames = ["fundamentals", "processing", "features_video", "3d_geometry", "dnn_systems", "advanced_robotics"]
    fname = f"guide_modules/part{p}_{pnames[p-1]}.py"
    with open(fname, "r", encoding="utf-8") as f:
        text = f.read()

    # Replace PARTX_CONTENT = """ with PARTX_CONTENT = r"""
    text = re.sub(r'PART\d+_CONTENT\s*=\s*"""', f'PART{p}_CONTENT = r"""', text)
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Updated {fname} with r\"\"\" raw string")
