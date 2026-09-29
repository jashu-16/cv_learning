# setup_docs_site.py
import os
import shutil
import re

from guide_modules.part1_fundamentals import PART1_CONTENT
from guide_modules.part2_processing import PART2_CONTENT
from guide_modules.part3_features_video import PART3_CONTENT
from guide_modules.part4_3d_geometry import PART4_CONTENT
from guide_modules.part5_dnn_systems import PART5_CONTENT
from guide_modules.part6_advanced_robotics import PART6_CONTENT

# 1. Create docs directory structure
os.makedirs("docs/opencv", exist_ok=True)
os.makedirs("docs/pytorch", exist_ok=True)
os.makedirs("docs/assets", exist_ok=True)
os.makedirs("docs/javascripts", exist_ok=True)
os.makedirs("docs/stylesheets", exist_ok=True)
os.makedirs(".github/workflows", exist_ok=True)

# 2. Copy assets
if os.path.exists("assets"):
    for item in os.listdir("assets"):
        src = os.path.join("assets", item)
        dst = os.path.join("docs/assets", item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
print("Copied assets to docs/assets/")

# 3. Create Home Landing Page docs/index.md
index_content = """# Computer Vision & Robotics Perception Engineering Handbook

> **A Complete Practical Guide: From Absolute Beginner to Advanced Production Engineer**

Welcome to the **Computer Vision & Robotics Perception Engineering Master Handbook**. This curriculum is designed with a **3-Level Progressive Learning System** so you can master complex computer vision concepts step-by-step:

- 🟢 **Level 1 (Beginner)**: Simple plain-English intuition, everyday real-world analogies, and zero math jargon.
- 🟡 **Level 2 (Intermediate)**: Step-by-step arithmetic breakdowns with real numbers, demystified formulas, and copy-paste Python code.
- 🔴 **Level 3 (Advanced)**: Production architecture, hardware SIMD (AVX2/NEON) vectorization, memory stride optimization, and real-time robotics deployment.

---

## 🗺️ Visual Learning Pathway

```mermaid
graph TD
    A[📷 Part 1: Fundamentals & Pixels] --> B[🔍 Part 2: Image Processing & Filters]
    B --> C[🎯 Part 3: Features & Video Tracking]
    C --> D[🧊 Part 4: Camera 3D & Stereo Depth]
    D --> E[🧠 Part 5: Deep Learning & Production Systems]
    E --> F[🤖 Part 6: SLAM, State Estimation & Robotics]
```

---

## 📚 Curriculum Modules

<div class="grid cards" markdown>

-   ### 📷 Part 1: OpenCV & Image Fundamentals
    ---
    **Chapters 1–6** • Light physics, photodiode sensors, ADC quantization, NumPy memory strides, color spaces (BGR, HSV, LAB), stencils & affine/perspective transforms.
    
    [👉 Explore Part 1 (Fundamentals)](opencv/part1_fundamentals.md){ .md-button .md-button--primary }

-   ### 🔍 Part 2: Core Image Processing & Geometry
    ---
    **Chapters 7–13** • 2D spatial convolution, Gaussian/Bilateral filters, CLAHE, Otsu & adaptive thresholding, Canny edge detection, morphology & Hough transforms.
    
    [👉 Explore Part 2 (Processing)](opencv/part2_processing.md){ .md-button .md-button--primary }

-   ### 🎯 Part 3: Features, Matching & Video Tracking
    ---
    **Chapters 14–19** • FAST & ORB keypoints, FLANN & Lowe's ratio test, Homography RANSAC, VideoCapture zero-latency threading, CSRT/KCF tracking & Optical Flow.
    
    [👉 Explore Part 3 (Features & Video)](opencv/part3_features_video.md){ .md-button .md-button--primary }

-   ### 🧊 Part 4: Camera Calibration, 3D Geometry & Stereo
    ---
    **Chapters 20–25** • Pinhole camera matrix, radial/tangential distortion, solvePnP 6DoF pose, Epipolar stereo depth, ArUco fiducials & Watershed marker segmentation.
    
    [👉 Explore Part 4 (3D Vision)](opencv/part4_3d_geometry.md){ .md-button .md-button--primary }

-   ### 🧠 Part 5: Deep Learning, Production & Systems
    ---
    **Chapters 26–32** • OCR pre-processing pipelines, YOLO & NMS vectorization, `cv2.dnn` ONNX runtime inference, multi-threaded pipelines, SIMD vectorization & Docker deployment.
    
    [👉 Explore Part 5 (DNN & Systems)](opencv/part5_dnn_systems.md){ .md-button .md-button--primary }

-   ### 🤖 Part 6: Robotics Perception & Interview Prep
    ---
    **Chapters 33–39** • Visual SLAM triangulation, Kalman filter sensor fusion, Bird's Eye View (BEV/IPM), Exposure fusion & Top 15 FAANG computer vision interview breakdowns.
    
    [👉 Explore Part 6 (Robotics)](opencv/part6_robotics.md){ .md-button .md-button--primary }

</div>

---

## ⚡ Additional Comprehensive Resources

- 📖 **[Complete Monolithic OpenCV Guide (All 39 Chapters)](opencv/full_guide.md)** — The full single-page scrollable handbook with all chapters, diagrams, and code.
- 🧪 **[PyTorch Deep Learning for Vision Notes](pytorch/deep_learning_guide.md)** — PyTorch tensors, autograd, CNN backbones, ViTs, U-Net segmentation, and foundation models.
- 💻 **[GitHub Repository](https://github.com/jashu-16/cv_learning)** — Clean source code, standalone runnable scripts, and diagram generators.
"""

with open("docs/index.md", "w", encoding="utf-8") as f:
    f.write(index_content)
print("Created docs/index.md")

# 4. Helper to adapt image paths for subdirectories
def fix_image_paths(content, is_subfolder=True):
    prefix = "../assets/" if is_subfolder else "assets/"
    # Replace assets/ with ../assets/ if needed
    return re.sub(r'\(assets/([^)]+)\)', rf'({prefix}\1)', content)

# 5. Write individual OpenCV Parts
parts = [
    ("docs/opencv/part1_fundamentals.md", "# Part 1: OpenCV & Image Fundamentals\n\n" + PART1_CONTENT),
    ("docs/opencv/part2_processing.md", "# Part 2: Core Image Processing & Geometry\n\n" + PART2_CONTENT),
    ("docs/opencv/part3_features_video.md", "# Part 3: Features, Matching & Video Tracking\n\n" + PART3_CONTENT),
    ("docs/opencv/part4_3d_geometry.md", "# Part 4: Camera Calibration, 3D Geometry & Stereo\n\n" + PART4_CONTENT),
    ("docs/opencv/part5_dnn_systems.md", "# Part 5: Deep Learning, Production & Systems\n\n" + PART5_CONTENT),
    ("docs/opencv/part6_robotics.md", "# Part 6: Advanced Robotics Perception & Interview Prep\n\n" + PART6_CONTENT),
]

for filepath, raw in parts:
    adapted = fix_image_paths(raw, is_subfolder=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(adapted)
    print(f"Created {filepath}")

# 6. Copy Full Guides
with open("OPENCV_STUDY_GUIDE.md", "r", encoding="utf-8") as f:
    full_opencv = f.read()
with open("docs/opencv/full_guide.md", "w", encoding="utf-8") as f:
    f.write(fix_image_paths(full_opencv, is_subfolder=True))
print("Created docs/opencv/full_guide.md")

if os.path.exists("PYTORCH_DEEP_LEARNING_CV_GUIDE.md"):
    with open("PYTORCH_DEEP_LEARNING_CV_GUIDE.md", "r", encoding="utf-8") as f:
        full_pytorch = f.read()
    with open("docs/pytorch/deep_learning_guide.md", "w", encoding="utf-8") as f:
        f.write(fix_image_paths(full_pytorch, is_subfolder=True))
    print("Created docs/pytorch/deep_learning_guide.md")

# 7. Create MathJax & Mermaid Support Files
mathjax_js = """window.MathJax = {
  tex: {
    inlineMath: [["\\\\(", "\\\\)"], ["$", "$"]],
    displayMath: [["\\\\[", "\\\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  }
});
"""
with open("docs/javascripts/mathjax.js", "w", encoding="utf-8") as f:
    f.write(mathjax_js)
print("Created docs/javascripts/mathjax.js")

extra_css = """/* Custom enhancements for OpenCV study guide */
:root {
  --md-primary-fg-color: #2563eb;
  --md-primary-fg-color--light: #3b82f6;
  --md-primary-fg-color--dark: #1d4ed8;
  --md-accent-fg-color: #7c3aed;
}

/* Ensure code blocks and tables have nice scrolling */
.md-typeset table:not([class]) {
  display: inline-block;
  overflow-x: auto;
  max-width: 100%;
}

/* Ensure diagrams don't overflow */
.mermaid {
  text-align: center;
  margin: 1.5rem 0;
}
"""
with open("docs/stylesheets/extra.css", "w", encoding="utf-8") as f:
    f.write(extra_css)
print("Created docs/stylesheets/extra.css")

# 8. Create mkdocs.yml
mkdocs_yml = """site_name: Computer Vision & Robotics Master Handbook
site_description: Production-grade study guide covering OpenCV, 3D Geometry, SLAM, PyTorch, and Robotics Perception.
site_author: Jashwanth
site_url: https://jashu-16.github.io/cv_learning/
repo_url: https://github.com/jashu-16/cv_learning
repo_name: jashu-16/cv_learning

theme:
  name: material
  language: en
  palette:
    # Light mode
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: deep purple
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    # Dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: deep purple
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - navigation.expand
    - search.suggest
    - search.highlight
    - content.code.copy
    - content.tabs.link
    - content.action.view

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.details
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - tables
  - attr_list
  - def_list
  - md_in_html
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg

extra_javascript:
  - javascripts/mathjax.js
  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js

extra_css:
  - stylesheets/extra.css

nav:
  - Home: index.md
  - OpenCV Engineering:
      - "Part 1: Fundamentals (Ch 1-6)": opencv/part1_fundamentals.md
      - "Part 2: Image Processing (Ch 7-13)": opencv/part2_processing.md
      - "Part 3: Features & Video (Ch 14-19)": opencv/part3_features_video.md
      - "Part 4: 3D Geometry & Stereo (Ch 20-25)": opencv/part4_3d_geometry.md
      - "Part 5: DNN & Systems (Ch 26-32)": opencv/part5_dnn_systems.md
      - "Part 6: Robotics & SLAM (Ch 33-39)": opencv/part6_robotics.md
      - "Full Monolithic Guide (All 39 Ch)": opencv/full_guide.md
  - PyTorch Deep Learning:
      - "Deep Learning for Computer Vision": pytorch/deep_learning_guide.md
"""

with open("mkdocs.yml", "w", encoding="utf-8") as f:
    f.write(mkdocs_yml)
print("Created mkdocs.yml")

# 9. Create GitHub Actions deployment workflow
gh_workflow = """name: Deploy Documentation to GitHub Pages

on:
  push:
    branches:
      - main
      - master
  workflow_dispatch:

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure Git Credentials
        run: |
          git config user.name github-actions[bot]
          git config user.email 41898282+github-actions[bot]@users.noreply.github.com

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt

      - name: Deploy to GitHub Pages
        run: mkdocs gh-deploy --force
"""

with open(".github/workflows/deploy.yml", "w", encoding="utf-8") as f:
    f.write(gh_workflow)
print("Created .github/workflows/deploy.yml")

print("Setup completed successfully!")
