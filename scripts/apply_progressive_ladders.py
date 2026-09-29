# apply_progressive_ladders.py
import re
import os
from build_all_progressive_guides import PROGRESSIVE_GUIDES

part_files = [
    ("guide_modules/part1_fundamentals.py", range(1, 7)),
    ("guide_modules/part2_processing.py", range(7, 14)),
    ("guide_modules/part3_features_video.py", range(14, 20)),
    ("guide_modules/part4_3d_geometry.py", range(20, 26)),
    ("guide_modules/part5_dnn_systems.py", range(26, 33)),
    ("guide_modules/part6_advanced_robotics.py", range(33, 40))
]

for file_path, ch_range in part_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Step 0: Ensure newline after opening triple quotes
    content = re.sub(r'(PART\d+_CONTENT\s*=\s*""")##', r'\1\n##', content)

    # Step 1: Strip out any existing Big Picture or Learning Ladder blocks
    content = re.sub(
        r"### 💡 The Big Picture in Plain English.*?(?=\n### |\n## |\Z)",
        "",
        content,
        flags=re.DOTALL
    )
    content = re.sub(
        r"### 🔰 The 3-Level Learning Ladder.*?(?=\n### |\n## |\Z)",
        "",
        content,
        flags=re.DOTALL
    )

    # Step 2: Insert the rich 3-Level Learning Ladder in reverse chapter order
    for ch_num in reversed(list(ch_range)):
        pattern = rf"^## {ch_num}\.\s+.*?\n"
        m = re.search(pattern, content, re.MULTILINE)

        if m:
            ch_start = m.start()
            ch_end = m.end()

            next_h2 = re.search(r"^## \d+\.", content[ch_end:], re.MULTILINE)
            ch_limit = ch_end + next_h2.start() if next_h2 else len(content)

            ladder = "\n" + PROGRESSIVE_GUIDES[ch_num].strip() + "\n\n"

            def_pos = content.find("### Definition", ch_start)
            if def_pos != -1 and def_pos < ch_limit:
                next_h3 = content.find("### ", def_pos + 15)
                if next_h3 != -1 and next_h3 < ch_limit:
                    content = content[:next_h3] + ladder + content[next_h3:]
                else:
                    content = content[:ch_limit] + ladder + content[ch_limit:]
            else:
                content = content[:ch_end] + ladder + content[ch_end:]
        else:
            print(f"Warning: Chapter {ch_num} not found in {file_path}!")

    content = re.sub(r"\n{4,}", "\n\n\n", content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully injected 3-Level Learning Ladders into {file_path}")

print("All modules updated cleanly!")
