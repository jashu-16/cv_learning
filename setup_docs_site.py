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

Welcome to the **Computer Vision & Robotics Perception Engineering Master Handbook**.

This comprehensive resource covers physical light principles, mathematical representations, core image processing algorithms, 3D geometry & SLAM, deep learning inference, and real-time robotics deployment.

---

## 📚 Curriculum Structure

<div class="grid cards" markdown>

-   :material-camera:{ .lg .middle } **Part 1: OpenCV & Image Fundamentals**

    ---

    Chapters 1–6: Photodiodes, ADC quantization, NumPy memory strides, color spaces (BGR, HSV, LAB), stencils & affine/perspective transforms.

    [:octicons-arrow-right-24: Explore Part 1](opencv/part1_fundamentals.md)

-   :material-filter:{ .lg .middle } **Part 2: Core Image Processing**

    ---

    Chapters 7–13: Spatial convolution, Gaussian/Bilateral filters, CLAHE, Otsu/Adaptive thresholding, Canny edge detection, Morphology & Hough transforms.

    [:octicons-arrow-right-24: Explore Part 2](opencv/part2_processing.md)

-   :material-target:{ .lg .middle } **Part 3: Features, Matching & Video**

    ---

    Chapters 14–19: FAST/ORB keypoints, FLANN & Lowe's ratio test, Homography RANSAC, VideoCapture buffer management, CSRT/KCF tracking & Optical Flow.

    [:octicons-arrow-right-24: Explore Part 3](opencv/part3_features_video.md)

-   :material-cube-outline:{ .lg .middle } **Part 4: Camera Calibration & 3D Vision**

    ---

    Chapters 20–25: Pinhole camera model, lens distortion, solvePnP pose estimation, Epipolar stereo depth, ArUco fiducials & Watershed segmentation.

    [:octicons-arrow-right-24: Explore Part 4](opencv/part4_3d_geometry.md)

-   :material-brain:{ .lg .middle } **Part 5: Deep Learning & Systems**

    ---

    Chapters 26–32: OCR pipelines, YOLO/NMS bounding boxes, `cv2.dnn` ONNX runtime, zero-copy producer-consumer streaming, SIMD acceleration & Docker.

    [:octicons-arrow-right-24: Explore Part 5](opencv/part5_dnn_systems.md)

-   :material-robot:{ .lg .middle } **Part 6: Advanced Robotics & Interview Prep**

    ---

    Chapters 33–39: Visual SLAM triangulation, Kalman filter state tracking, Bird's Eye View (BEV/IPM), Exposure fusion & Top 15 FAANG interview answers.

    [:octicons-arrow-right-24: Explore Part 6](opencv/part6_robotics.md)

</div>

---

## 🚀 Quick Navigation

- 📖 **[Complete Monolithic OpenCV Guide](opencv/full_guide.md)** — All 39 chapters in a single scrollable document for quick search.
- ⚡ **[PyTorch Deep Learning for Vision](pytorch/deep_learning_guide.md)** — Neural networks, CNN backbones, ViTs, and segmentation.
- 💻 **[GitHub Repository](https://github.com/jashu-16/cv_learning)** — Source code, scripts, and visual generators.
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
