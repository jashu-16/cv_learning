# build_full_guide.py
import os
from guide_modules.part1_fundamentals import PART1_CONTENT
from guide_modules.part2_processing import PART2_CONTENT
from guide_modules.part3_features_video import PART3_CONTENT
from guide_modules.part4_3d_geometry import PART4_CONTENT
from guide_modules.part5_dnn_systems import PART5_CONTENT
from guide_modules.part6_advanced_robotics import PART6_CONTENT

HEADER = """# Comprehensive OpenCV & Computer Vision Engineering Master Handbook

> **Target Audience:** Computer Vision Engineers, Robotics Perception Engineers, and Autonomous Systems Developers  
> **Stack:** Python 3.10+, OpenCV (`cv2` 4.x+), NumPy (`>=1.23`), Matplotlib  
> **Philosophy:** Understand physical light & mathematical representations first, algorithmic internals & hardware memory models second, and OpenCV implementation patterns third.

---

## Table of Contents

1. [OpenCV Fundamentals](#1-opencv-fundamentals)
2. [Images & NumPy Fundamentals](#2-images--numpy-fundamentals)
3. [Image Input & Output](#3-image-input--output)
4. [Color Spaces](#4-color-spaces)
5. [Image Manipulation](#5-image-manipulation)
6. [Geometric Transformations](#6-geometric-transformations)
7. [Image Filtering & Smoothing](#7-image-filtering--smoothing)
8. [Image Enhancement](#8-image-enhancement)
9. [Image Thresholding](#9-image-thresholding)
10. [Image Gradients & Edge Detection](#10-image-gradients--edge-detection)
11. [Morphological Image Processing](#11-morphological-image-processing)
12. [Contours & Shape Analysis](#12-contours--shape-analysis)
13. [Hough Transform & Geometric Detection](#13-hough-transform--geometric-detection)
14. [Feature Detection & Description](#14-feature-detection--description)
15. [Feature Matching](#15-feature-matching)
16. [Homography & Image Registration](#16-homography--image-registration)
17. [Video Processing](#17-video-processing)
18. [Object Tracking](#18-object-tracking)
19. [Optical Flow](#19-optical-flow)
20. [Camera Calibration](#20-camera-calibration)
21. [Camera Pose & 3D Geometry](#21-camera-pose--3d-geometry)
22. [Stereo Vision & Depth](#22-stereo-vision--depth)
23. [ArUco & Fiducial Markers](#23-aruco--fiducial-markers)
24. [Image Segmentation](#24-image-segmentation)
25. [Connected Components & Blob Analysis](#25-connected-components--blob-analysis)
26. [OCR & Text Processing](#26-ocr--text-processing)
27. [Object Detection Integration](#27-object-detection-integration)
28. [Deep Learning + OpenCV](#28-deep-learning--opencv-cv2dnn)
29. [OpenCV for Computer Vision Systems](#29-opencv-for-computer-vision-systems)
30. [OpenCV for Robotics Perception](#30-opencv-for-robotics-perception)
31. [Performance Optimization](#31-performance-optimization)
32. [Production & Deployment](#32-production--deployment)
33. [Visual SLAM & 3D Triangulation](#33-visual-slam--3d-triangulation)
34. [Kalman Filter & Motion Tracking](#34-kalman-filter--motion-tracking)
35. [Inverse Perspective Mapping & BEV](#35-inverse-perspective-mapping--bev)
36. [Exposure Fusion & HDR Imaging](#36-exposure-fusion--hdr-imaging)
37. [Barcode & QR Code Pose Localization](#37-barcode--qr-code-pose-localization)
38. [Practical Robotics & Perception Projects](#38-practical-robotics--perception-projects)
39. [OpenCV Interview Preparation & Formulas](#39-opencv-interview-preparation--formulas)

---

"""

def build_guide():
    full_text = (
        HEADER +
        PART1_CONTENT.strip() + "\n\n---\n\n" +
        PART2_CONTENT.strip() + "\n\n---\n\n" +
        PART3_CONTENT.strip() + "\n\n---\n\n" +
        PART4_CONTENT.strip() + "\n\n---\n\n" +
        PART5_CONTENT.strip() + "\n\n---\n\n" +
        PART6_CONTENT.strip() + "\n"
    )
    
    target_path = "/Users/jashwanth/cv_learning/OPENCV_STUDY_GUIDE.md"
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print(f"Successfully generated OPENCV_STUDY_GUIDE.md ({len(full_text):,} characters, {len(full_text.splitlines()):,} lines).")

if __name__ == "__main__":
    build_guide()
