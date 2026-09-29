# inject_all_mermaids.py
import re
import os

def update_part1():
    with open("guide_modules/part1_fundamentals.py", "r") as f:
        content = f.read()

    # Add Mermaid to Ch 1
    mermaid_ch1 = """```mermaid
flowchart LR
    A["Physical Scene Light\nPhi(x, y, lambda, t)"] --> B["Microlens Array\nSpatial Sampling"]
    B --> C["Photodiode Array\nCharge Integration"]
    C --> D["ADC Converter\nVoltage to Integer"]
    D --> E["C++ cv::Mat\nContiguous Buffer"]
    E --> F["NumPy Array\nZero-Copy Python View"]
```"""
    
    # Add Mermaid to Ch 2
    mermaid_ch2 = """```mermaid
flowchart TD
    subgraph MemoryBuffer ["3D Memory Layout (HWC)"]
        P0["Row 0: [B0, G0, R0], [B1, G1, R1]..."]
        P1["Row 1: [B0, G0, R0], [B1, G1, R1]..."]
        P2["Row H-1: [B0, G0, R0], [B1, G1, R1]..."]
    end
    subgraph Strides ["Byte Strides"]
        S0["S0 = Width * Channels * 1 byte"]
        S1["S1 = Channels * 1 byte"]
        S2["S2 = 1 byte"]
    end
    subgraph Slicing ["ROI View (Zero Copy)"]
        ROI["roi = img[y1:y2, x1:x2]\nShares same memory pointer!"]
    end
    MemoryBuffer --> Strides --> Slicing
```"""

    # Add Mermaid to Ch 3
    mermaid_ch3 = """```mermaid
flowchart LR
    A["Encoded File on Disk\n(JPEG / PNG / TIFF)"] -->|cv2.imread| B["Uncompressed RAM Matrix\n(H x W x C uint8)"]
    B -->|cv2.imwrite| C["Compressed File on Disk\n(Lossy / Lossless)"]
    B -->|cv2.imencode| D["In-Memory RAM Buffer\n(Zero Disk I/O)"]
    D -->|cv2.imdecode| B
```"""

    # Add Mermaid to Ch 4
    mermaid_ch4 = """```mermaid
flowchart TD
    BGR["Input BGR Image\n(Coupled Color & Brightness)"] -->|cv2.COLOR_BGR2GRAY| GRAY["Grayscale (Luminance Y)\n0.299R + 0.587G + 0.114B"]
    BGR -->|cv2.COLOR_BGR2HSV| HSV["HSV Color Space\nDecoupled Hue [0,179] vs Value [0,255]"]
    BGR -->|cv2.COLOR_BGR2Lab| LAB["CIE L*a*b*\nPerceptually Uniform Distance Delta E"]
    BGR -->|cv2.COLOR_BGR2YCrCb| YCRCB["YCrCb\nLuma + Chrominance (Video & Skin)"]
    HSV -->|cv2.inRange| MASK["Shadow-Invariant Binary Mask"]
```"""

    # Add Mermaid to Ch 5
    mermaid_ch5 = """```mermaid
flowchart LR
    FG["Foreground Object"] --> M1["Threshold -> Binary Mask"]
    M1 --> M2["cv2.bitwise_not -> Inverted Mask"]
    BG["Background Scene"] --> P1["cv2.bitwise_and(BG, Inverted Mask)\nPunches Black Hole"]
    FG --> P2["cv2.bitwise_and(FG, Mask)\nExtracts Clean Object"]
    P1 --> ADD["cv2.add(Masked BG, Clean FG)\nSeamless Composite"]
    P2 --> ADD
```"""

    # Add Mermaid to Ch 6
    mermaid_ch6 = """```mermaid
flowchart TD
    subgraph Affine ["Affine Transformation (6 DOF)"]
        A1["3 Point Pairs"] --> A2["cv2.getAffineTransform\n2x3 Matrix M"]
        A2 --> A3["Preserves Parallel Lines\nRotation, Scale, Translation, Shear"]
    end
    subgraph Perspective ["Perspective Homography (8 DOF)"]
        P1["4 Point Pairs"] --> P2["cv2.getPerspectiveTransform\n3x3 Matrix H"]
        P2 --> P3["Preserves Straight Lines\nVanishing Points & Angled Planes"]
    end
    A3 --> WARP["Backward Warping (M^-1 or H^-1)\nSub-pixel Interpolation (INTER_LINEAR / INTER_AREA)"]
    P3 --> WARP
```"""

    # Insert into content
    if "flowchart LR" not in content:
        content = content.replace("### Visual Demonstration & Coordinate Alignment", "### System Architecture & Pipeline Flowchart\n" + mermaid_ch1 + "\n\n### Visual Demonstration & Coordinate Alignment")
        content = content.replace("### Visual Demonstration & Channel Architecture", "### Memory Architecture & Strides Diagram\n" + mermaid_ch2 + "\n\n### Visual Demonstration & Channel Architecture")
        content = content.replace("### Important OpenCV Functions & Syntax\n```python\ncv2.imread", "### Image I/O Processing Flowchart\n" + mermaid_ch3 + "\n\n### Important OpenCV Functions & Syntax\n```python\ncv2.imread")
        content = content.replace("### Visual Demonstration & Color Decomposition", "### Color Space Transformation Graph\n" + mermaid_ch4 + "\n\n### Visual Demonstration & Color Decomposition")
        content = content.replace("### Visual Demonstration & Bitwise Logic Pipeline", "### Bitwise Masking Pipeline Flowchart\n" + mermaid_ch5 + "\n\n### Visual Demonstration & Bitwise Logic Pipeline")
        content = content.replace("### Visual Demonstration & Warp Comparison", "### Geometric Transformation Architecture\n" + mermaid_ch6 + "\n\n### Visual Demonstration & Warp Comparison")
    
    with open("guide_modules/part1_fundamentals.py", "w") as f:
        f.write(content)
    print("Part 1 updated with Mermaid diagrams.")

def update_part2():
    with open("guide_modules/part2_processing.py", "r") as f:
        content = f.read()

    mermaid_ch7 = """```mermaid
flowchart LR
    A["Raw Noisy Image"] --> B["Averaging Blur\nUniform 1/N^2 (Blurs Edges)"]
    A --> C["Gaussian Blur\nSpatial Bell Curve (Softens Edges)"]
    A --> D["Median Blur\nStatistical Rank (Kills Salt & Pepper)"]
    A --> E["Bilateral Filter\nSpatial Gauss x Color Gauss (Keeps Edges Sharp)"]
```"""

    mermaid_ch8 = """```mermaid
flowchart TD
    IMG["Low Contrast Image"] --> TILES["Divide into 8x8 Local Tiles"]
    TILES --> HIST["Compute Tile Histograms"]
    HIST --> CLIP["Clip Spikes > clipLimit (e.g., 3.0)"]
    CLIP --> REDIST["Redistribute Excess Evenly Across Bins"]
    REDIST --> CDF["Equalize via Local CDF"]
    CDF --> MERGE["Bilinear Interpolation Across Tile Borders"]
    MERGE --> OUT["CLAHE Enhanced Output"]
```"""

    mermaid_ch9 = """```mermaid
flowchart TD
    INPUT["Grayscale Image"] --> DECISION{"Illumination Uniform?"}
    DECISION -->|Yes| OTSU["cv2.THRESH_OTSU\nMaximizes Between-Class Variance"]
    DECISION -->|No (Shadows/Gradients)| ADAPTIVE["cv2.adaptiveThreshold\nDynamic Threshold per Pixel = Local Mean - C"]
    OTSU --> BINARY["Clean Binary Mask (0 or 255)"]
    ADAPTIVE --> BINARY
```"""

    mermaid_ch10 = """```mermaid
flowchart TD
    A["Input Image"] --> B["1. Gaussian Smoothing (Noise Removal)"]
    B --> C["2. Sobel Derivatives (Gx, Gy) -> Magnitude & Angle"]
    C --> D["3. Non-Maximum Suppression (1-Pixel Thin Edges)"]
    D --> E["4. Double Thresholding (Strong vs Weak Edges)"]
    E --> F["5. Hysteresis Edge Tracking (Keep Weak if Connected to Strong)"]
    F --> G["Final Canny Edge Map"]
```"""

    mermaid_ch11 = """```mermaid
flowchart TD
    BIN["Binary Mask"] --> ERODE["Erosion (A - B)\nShaves Boundaries & Kills Noise"]
    BIN --> DILATE["Dilation (A + B)\nExpands Boundaries & Fills Holes"]
    ERODE -->|Followed by Dilation| OPEN["Opening (A o B)\nEliminates External White Noise"]
    DILATE -->|Followed by Erosion| CLOSE["Closing (A . B)\nBridges Internal Black Cracks"]
    DILATE -.->|Subtract Eroded| GRAD["Morphological Gradient\nObject Boundary Outline"]
```"""

    mermaid_ch12 = """```mermaid
flowchart LR
    A["Binary Mask"] -->|Suzuki Border Following| B["Contour Points List & Hierarchy Tree"]
    B --> C["Spatial Moments (m00 Area, Centroid cx, cy)"]
    B --> D["Bounding Shapes (BoundingRect, MinAreaRect, ConvexHull)"]
    B --> E["Douglas-Peucker approxPolyDP (Polygon Simplification)"]
```"""

    mermaid_ch13 = """```mermaid
flowchart LR
    EDGE["Canny Edge Pixels (x, y)"] -->|r = x*cos(theta) + y*sin(theta)| ACCUM["2D Accumulator Space (r, theta)"]
    ACCUM -->|Sinusoidal Curves Intersect| PEAK["Local Peak Votes > Threshold"]
    PEAK --> LINES["Detected Parametric Lines & Segments"]
```"""

    if "flowchart LR" not in content:
        content = content.replace("### Visual Demonstration & Filter Comparison", "### Image Filtering Architecture\n" + mermaid_ch7 + "\n\n### Visual Demonstration & Filter Comparison")
        content = content.replace("### Visual Demonstration & Contrast Comparison", "### CLAHE Enhancement Flowchart\n" + mermaid_ch8 + "\n\n### Visual Demonstration & Contrast Comparison")
        content = content.replace("### Visual Demonstration & Thresholding Comparison", "### Thresholding Selection Pipeline\n" + mermaid_ch9 + "\n\n### Visual Demonstration & Thresholding Comparison")
        content = content.replace("### Visual Demonstration & Gradient Stages", "### Canny 5-Stage Edge Detection Pipeline\n" + mermaid_ch10 + "\n\n### Visual Demonstration & Gradient Stages")
        content = content.replace("### Visual Demonstration & Morphological Workflow", "### Morphological Processing Flowchart\n" + mermaid_ch11 + "\n\n### Visual Demonstration & Morphological Workflow")
        content = content.replace("### Visual Demonstration & Contour Decomposition", "### Contour Extraction & Geometric Analysis\n" + mermaid_ch12 + "\n\n### Visual Demonstration & Contour Decomposition")
        content = content.replace("### Visual Demonstration & Accumulator Voting", "### Hough Space Accumulator Voting Flowchart\n" + mermaid_ch13 + "\n\n### Visual Demonstration & Accumulator Voting")

    with open("guide_modules/part2_processing.py", "w") as f:
        f.write(content)
    print("Part 2 updated with Mermaid diagrams.")

def update_part3():
    with open("guide_modules/part3_features_video.py", "r") as f:
        content = f.read()

    mermaid_ch14 = """```mermaid
flowchart TD
    IMG["Grayscale Image"] --> HARRIS["Harris / Shi-Tomasi\nEigenvalues of Second Moment Matrix M"]
    IMG --> SIFT["SIFT (Scale Invariant)\nDoG Scale Space + 128-d Float Gradient Vector"]
    IMG --> ORB["ORB (Real-Time)\nFAST Corner Tests + 256-bit Binary rBRIEF"]
```"""

    mermaid_ch15 = """```mermaid
flowchart LR
    D1["Query Descriptors"] & D2["Scene Descriptors"] --> MATCH["KNN Matcher (k=2)"]
    MATCH --> RATIO["Lowe's Ratio Test: d1 / d2 < 0.75"]
    RATIO -->|Reject Ambiguity| GOOD["Robust Matching Keypoint Pairs"]
```"""

    mermaid_ch16 = """```mermaid
flowchart TD
    PTS["Matched Feature Pairs"] --> RANSAC["RANSAC Loop:\n1. Random 4 Points\n2. Solve H via DLT/SVD\n3. Count Inliers (Reproj Error < 3px)"]
    RANSAC --> BEST_H["Optimal 3x3 Homography Matrix H"]
    BEST_H --> WARP["cv2.warpPerspective -> Metric Planar Registration"]
```"""

    mermaid_ch17 = """```mermaid
flowchart LR
    CAM["Camera Hardware Driver"] -->|Asynchronous Capture| THREAD["Background Grabber Thread"]
    THREAD -->|Overwrites Stale Frames| BUFFER["Single Latest Frame Buffer (Lock-Free)"]
    BUFFER -->|Zero-Lag Read| MAIN["Downstream Vision / ML Worker"]
```"""

    mermaid_ch18 = """```mermaid
flowchart TD
    INIT["Initial Target Box"] --> HIST["Hue Color Histogram / Correlation Filter"]
    NEXT["Next Video Frame"] --> BACK["Backproject / FFT Response Map"]
    BACK --> SHIFT["MeanShift / CSRT Spatial Mode Seeking"]
    SHIFT --> UPDATE["Updated Target Center, Scale & Angle"]
```"""

    mermaid_ch19 = """```mermaid
flowchart TD
    F1["Frame t"] & F2["Frame t+dt"] --> PYR["Build Multi-Level Gaussian Pyramids"]
    PYR --> TOP["Estimate Coarse Motion at Top Level (LK Matrix Inversion)"]
    TOP --> PROP["Propagate Velocity Vectors Downward"]
    PROP --> FINE["Refine Sub-Pixel Optical Flow at Level 0"]
```"""

    if "flowchart TD" not in content:
        content = content.replace("### Visual Demonstration & Feature Extraction", "### Feature Detection Architecture\n" + mermaid_ch14 + "\n\n### Visual Demonstration & Feature Extraction")
        content = content.replace("### Visual Demonstration & Feature Matching", "### Feature Matching & Ratio Test Pipeline\n" + mermaid_ch15 + "\n\n### Visual Demonstration & Feature Matching")
        content = content.replace("### Visual Demonstration & Homography Alignment", "### RANSAC Homography Estimation Pipeline\n" + mermaid_ch16 + "\n\n### Visual Demonstration & Homography Alignment")
        content = content.replace("### Visual Demonstration & Video Pipeline Architecture", "### Zero-Lag Threaded Video Architecture\n" + mermaid_ch17 + "\n\n### Visual Demonstration & Video Pipeline Architecture")
        content = content.replace("### Visual Demonstration & Tracker Trajectory", "### Visual Object Tracking Pipeline\n" + mermaid_ch18 + "\n\n### Visual Demonstration & Tracker Trajectory")
        content = content.replace("### Visual Demonstration & Flow Vector Field", "### Pyramidal Optical Flow Architecture\n" + mermaid_ch19 + "\n\n### Visual Demonstration & Flow Vector Field")

    with open("guide_modules/part3_features_video.py", "w") as f:
        f.write(content)
    print("Part 3 updated with Mermaid diagrams.")

def update_part4():
    with open("guide_modules/part4_3d_geometry.py", "r") as f:
        content = f.read()

    mermaid_ch20 = """```mermaid
flowchart LR
    WORLD["3D World Point Pw"] -->|Extrinsics [R | t]| CAM["3D Camera Frame Coordinates"]
    CAM -->|Intrinsics K| PIXEL["Ideal 2D Image Pixel (u, v)"]
    PIXEL -->|Distortion Model (k1, k2, p1, p2)| DIST["Distorted Observed Pixel (ud, vd)"]
    DIST -->|cv2.undistort| RECT["Metric Undistorted Pixel (u, v)"]
```"""

    mermaid_ch21 = """```mermaid
flowchart TD
    subgraph PnP ["Perspective-n-Point (6-DOF Localization)"]
        PTS3D["Known 3D World Points"] & PTS2D["Detected 2D Image Pixels"] --> SOLVE["cv2.solvePnPRansac"]
        SOLVE --> POSE["Rotation rvec & Translation tvec"]
    end
    subgraph Epipolar ["Two-View Epipolar Geometry"]
        V1["View 1: Point x"] & V2["View 2: Point x'"] --> FUND["x'^T F x = 0\nFundamental / Essential Matrix"]
        FUND --> EPI["Epipolar Line Constraint: l' = F x"]
    end
```"""

    mermaid_ch22 = """```mermaid
flowchart LR
    L["Left Camera Image"] & R["Right Camera Image"] --> RECT["cv2.stereoRectify\nMake Epipolar Lines Horizontal"]
    RECT --> MATCH["StereoSGBM Matching\nCompute Disparity d = xL - xR"]
    MATCH --> DEPTH["Metric Depth Z = (f * B) / d"]
    DEPTH --> CLOUD["cv2.reprojectImageTo3D\n3D Dense Point Cloud"]
```"""

    mermaid_ch23 = """```mermaid
flowchart TD
    FRAME["Camera Frame"] --> DETECT["cv2.aruco.detectMarkers\nFind Black Square Borders"]
    DETECT --> PARITY["Sample Internal Binary Grid & Verify Parity Bits"]
    PARITY --> ID["Extract Unique Marker ID & 4 Sub-Pixel Corners"]
    ID --> PNP["solvePnP with Marker Metric Size L"]
    PNP --> AXES["6-DOF Translation t & Rotation R (cv2.drawFrameAxes)"]
```"""

    mermaid_ch24 = """```mermaid
flowchart TD
    BIN["Touching Binary Blobs"] --> DIST["cv2.distanceTransform\nEuclidean Distance Map"]
    DIST --> SEEDS["Threshold Distance Peaks -> Sure Foreground Seeds"]
    BIN --> BG["Dilate Binary -> Sure Background"]
    SEEDS & BG --> MARKERS["Build Label Markers Matrix"]
    MARKERS --> WATER["cv2.watershed\nTopological Flooding & Boundary Dams"]
```"""

    mermaid_ch25 = """```mermaid
flowchart LR
    BIN["Binary Image"] --> PASS1["Pass 1: Row Scan & Provisional Labeling"]
    PASS1 --> UF["Disjoint-Set Union-Find (Resolve Equivalence)"]
    UF --> PASS2["Pass 2: Canonical Label Assignment"]
    PASS2 --> STATS["cv2.connectedComponentsWithStats\nBounding Box (x, y, w, h), Area A, Centroid (cx, cy)"]
```"""

    if "flowchart LR" not in content:
        content = content.replace("### Visual Demonstration & Calibration Target", "### Camera Calibration & Distortion Flowchart\n" + mermaid_ch20 + "\n\n### Visual Demonstration & Calibration Target")
        content = content.replace("### Visual Demonstration & Epipolar Geometry", "### Camera Pose & Epipolar Geometry Architecture\n" + mermaid_ch21 + "\n\n### Visual Demonstration & Epipolar Geometry")
        content = content.replace("### Visual Demonstration & Stereo Epipolar Rectification", "### Stereo Vision & Disparity Triangulation Pipeline\n" + mermaid_ch22 + "\n\n### Visual Demonstration & Stereo Epipolar Rectification")
        content = content.replace("### Visual Demonstration & 6-DOF Pose Axes", "### ArUco Detection & 6-DOF Localization Flowchart\n" + mermaid_ch23 + "\n\n### Visual Demonstration & 6-DOF Pose Axes")
        content = content.replace("### Visual Demonstration & Watershed Pipeline", "### Marker-Controlled Watershed Segmentation Flowchart\n" + mermaid_ch24 + "\n\n### Visual Demonstration & Watershed Pipeline")
        content = content.replace("### Visual Demonstration & Connected Components Statistics", "### Connected Component Labeling Architecture\n" + mermaid_ch25 + "\n\n### Visual Demonstration & Connected Components Statistics")

    with open("guide_modules/part4_3d_geometry.py", "w") as f:
        f.write(content)
    print("Part 4 updated with Mermaid diagrams.")

def update_part5():
    with open("guide_modules/part5_dnn_systems.py", "r") as f:
        content = f.read()

    mermaid_ch26 = """```mermaid
flowchart LR
    SCENE["Scene Image"] --> EAST["EAST FCN Model\nDetects Word Bounding Boxes"]
    EAST --> SKEW["cv2.minAreaRect\nEstimate Orientation Angle theta"]
    SKEW --> WARP["cv2.warpAffine\nDeskew to Horizontal Text Chip"]
    WARP --> OCR["Tesseract / CRNN Engine\nTranscribe Characters to String"]
```"""

    mermaid_ch27 = """```mermaid
flowchart TD
    YOLO["Neural Network Output\n(Thousands of Multi-Scale Boxes)"] --> CONF["Filter Confidence > score_threshold"]
    CONF --> SORT["Sort Remaining Boxes by Confidence (Descending)"]
    SORT --> NMS["cv2.dnn.NMSBoxes\nGreedy IoU Suppression (> nms_threshold)"]
    NMS --> FINAL["Optimal Unique Bounding Boxes"]
```"""

    mermaid_ch28 = """```mermaid
flowchart LR
    IMG["OpenCV BGR Image\n(H x W x C)"] --> BLOB["cv2.dnn.blobFromImage\nResize, SwapRB, MeanSub, Scalefactor"]
    BLOB --> NCHW["4D Tensor\n(1 x C x H x W)"]
    NCHW --> NET["cv2.dnn Forward Pass\n(CUDA / OpenCL / CPU Engine)"]
    NET --> OUT["Prediction Tensors"]
```"""

    mermaid_ch29 = """```mermaid
flowchart LR
    CAM["Camera V4L2 / GStreamer"] --> PROD["Producer Thread\n(Continuous Frame Capture)"]
    PROD --> RING["Lock-Free Circular Ring Buffer\n(Pre-allocated Shared Memory)"]
    RING --> CONS["Consumer Worker Thread\n(Always Reads Single Latest Frame)"]
```"""

    mermaid_ch30 = """```mermaid
flowchart TD
    ROS["ROS2 Image Topic"] --> BRIDGE["cv_bridge (sensor_msgs -> NumPy)"]
    BRIDGE --> CV["OpenCV Perception Pipeline\n(Object Detection / Pose Estimation)"]
    CV --> TF["tf2 Coordinate Transform (Camera -> Robot base_link)"]
    TF --> SERVO["Visual Servoing Controller (IBVS / PBVS Motor Commands)"]
```"""

    mermaid_ch31 = """```mermaid
flowchart TD
    OPENCV["OpenCV Operations"] --> SIMD["Vectorized SIMD (AVX2 / ARM NEON)\n32 Pixels in 1 CPU Cycle"]
    OPENCV --> TBB["Thread Pools (Intel TBB / OpenMP)\nMultithreaded Row Chunking"]
    OPENCV --> UMAT["OpenCL cv2.UMat\nTransparent GPU Offload"]
    OPENCV --> CUDA["cv2.cuda\nDedicated NVIDIA Hardware Kernels"]
```"""

    mermaid_ch32 = """```mermaid
flowchart LR
    DEV["Python / C++ Perception Code"] --> DOCKER["Minimal Docker Container\n(opencv-python-headless)"]
    DOCKER --> EDGE["Deploy to Edge (Jetson / Robot PC)"]
    EDGE --> MON["Watchdog Health Monitoring\n(FPS, Latency P99, RAM Leaks)"]
```"""

    if "flowchart LR" not in content:
        content = content.replace("### Visual Demonstration & Text Processing Workflow", "### Scene Text Recognition Pipeline\n" + mermaid_ch26 + "\n\n### Visual Demonstration & Text Processing Workflow")
        content = content.replace("### Visual Demonstration & Non-Maximum Suppression", "### Object Detection & NMS Flowchart\n" + mermaid_ch27 + "\n\n### Visual Demonstration & Non-Maximum Suppression")
        content = content.replace("### Visual Demonstration & DNN Inference Pipeline", "### OpenCV DNN Preprocessing & Inference Architecture\n" + mermaid_ch28 + "\n\n### Visual Demonstration & DNN Inference Pipeline")
        content = content.replace("### Visual Demonstration & Low-Latency Architecture", "### Real-Time Zero-Copy Streaming Architecture\n" + mermaid_ch29 + "\n\n### Visual Demonstration & Low-Latency Architecture")
        content = content.replace("### Visual Demonstration & Robot Perception Stack", "### Robotics Perception & Visual Servoing Stack\n" + mermaid_ch30 + "\n\n### Visual Demonstration & Robot Perception Stack")
        content = content.replace("### Visual Demonstration & SIMD Pipeline", "### Hardware Acceleration Hierarchy\n" + mermaid_ch31 + "\n\n### Visual Demonstration & SIMD Pipeline")
        content = content.replace("### Production Dockerfile Example", "### Production Deployment Architecture\n" + mermaid_ch32 + "\n\n### Production Dockerfile Example")

    with open("guide_modules/part5_dnn_systems.py", "w") as f:
        f.write(content)
    print("Part 5 updated with Mermaid diagrams.")

def update_part6():
    with open("guide_modules/part6_advanced_robotics.py", "r") as f:
        content = f.read()

    mermaid_ch33 = """```mermaid
flowchart TD
    FRAMES["Consecutive Video Frames"] --> VO["Visual Odometry (Feature Tracking)"]
    VO --> KF["Keyframe Selection Decision"]
    KF --> TRI["3D Landmark Triangulation (SVD)"]
    TRI --> BA["Local Bundle Adjustment (Reprojection Error Minimization)"]
    BA --> LOOP["Loop Closure Detection (Pose Graph Optimization)"]
```"""

    mermaid_ch34 = """```mermaid
flowchart TD
    subgraph Predict ["1. Predict Step"]
        P1["State Extrapolation: x = F * x"]
        P2["Covariance Extrapolation: P = F * P * F^T + Q"]
    end
    subgraph Update ["2. Update Step"]
        U1["Measurement: z from Detector"]
        U2["Innovation: y = z - H * x"]
        U3["Kalman Gain: K = P * H^T * (H * P * H^T + R)^-1"]
        U4["Updated State: x = x + K * y"]
        U5["Updated Covariance: P = (I - K * H) * P"]
    end
    Predict --> Update --> Predict
```"""

    mermaid_ch35 = """```mermaid
flowchart LR
    DASH["Forward Camera Dash View\n(Converging Lane Lines)"] --> TRAP["Select Ground-Plane Trapezoid (src)"]
    TRAP --> RECT["Define Orthogonal Metric Box (dst)"]
    RECT --> IPM["cv2.getPerspectiveTransform -> Homography H"]
    IPM --> BEV["Bird's-Eye-View (BEV)\nParallel Metric Occupancy Grid"]
```"""

    mermaid_ch36 = """```mermaid
flowchart TD
    BRACKET["Multi-Exposure Bracketed Photos\n(Under, Normal, Over Exposed)"] --> WEIGHTS["Compute Metric Weight Maps:\n1. Contrast 2. Saturation 3. Well-Exposedness"]
    WEIGHTS --> MERTENS["cv2.createMergeMertens\nLaplacian & Gaussian Pyramidal Blending"]
    MERTENS --> HDR["Tone-Mapped HDR Composite Image"]
```"""

    mermaid_ch37 = """```mermaid
flowchart LR
    IMG["Camera Frame"] --> QR["cv2.QRCodeDetector"]
    QR --> CORNERS["Extract 4 Finder Corner Coordinates"]
    CORNERS --> HOM["Perspective Rectification"]
    HOM --> DECODE["Reed-Solomon Error Correction -> String Payload"]
    CORNERS --> POSE["solvePnP -> 6-DOF Camera-to-QR Pose"]
```"""

    mermaid_ch38 = """```mermaid
flowchart TD
    CAM["Dash Camera Stream"] --> IPM["Inverse Perspective Mapping (BEV)"]
    IPM --> HIST["Sliding Window Histogram Peak Tracker"]
    HIST --> POLY["Fit 2nd Order Polynomial: x = ay^2 + by + c"]
    POLY --> CURV["Compute Road Curvature Radius & Lateral Offset"]
    CURV --> STANLEY["Stanley / Pure Pursuit Controller -> Steering Angle"]
```"""

    mermaid_ch39 = """```mermaid
flowchart TD
    CV["OpenCV Master Architecture"] --> PIXELS["Low-Level (Pixels, Color, Filters, Gradients)"]
    PIXELS --> FEATURES["Mid-Level (SIFT, ORB, Contours, Hough, Segments)"]
    FEATURES --> GEOM["3D Geometry (PnP, Calibration, Stereo, Homography)"]
    GEOM --> SYSTEMS["High-Level Systems (SLAM, YOLO DNN, Kalman, ROS2)"]
```"""

    if "flowchart TD" not in content:
        content = content.replace("### Important OpenCV Functions & Syntax", "### Visual SLAM Architecture Flowchart\n" + mermaid_ch33 + "\n\n### Important OpenCV Functions & Syntax")
        content = content.replace("### Important OpenCV Functions & Syntax\n```python\n# Initialize Kalman Filter", "### Kalman Filter Recursive State Machine\n" + mermaid_ch34 + "\n\n### Important OpenCV Functions & Syntax\n```python\n# Initialize Kalman Filter")
        content = content.replace("### Executable Python Example", "### Inverse Perspective Mapping Pipeline\n" + mermaid_ch35 + "\n\n### Executable Python Example")
        content = content.replace("### Important OpenCV Functions & Syntax\n```python\n# Mertens Exposure Fusion", "### Exposure Fusion Architecture\n" + mermaid_ch36 + "\n\n### Important OpenCV Functions & Syntax\n```python\n# Mertens Exposure Fusion")
        content = content.replace("### Executable Python Example\n```python\nimport cv2\nimport numpy as np\n\n# Initialize OpenCV native QR Code Detector", "### QR Code Localization & Pose Pipeline\n" + mermaid_ch37 + "\n\n### Executable Python Example\n```python\nimport cv2\nimport numpy as np\n\n# Initialize OpenCV native QR Code Detector")
        content = content.replace("### Project 1: Autonomous Lane Detection & Steering Angle Controller", "### Autonomous Lane Keeping Architecture\n" + mermaid_ch38 + "\n\n### Project 1: Autonomous Lane Detection & Steering Angle Controller")
        content = content.replace("### Essential Mathematical Formulas Master Reference", "### Computer Vision Conceptual Hierarchy\n" + mermaid_ch39 + "\n\n### Essential Mathematical Formulas Master Reference")

    with open("guide_modules/part6_advanced_robotics.py", "w") as f:
        f.write(content)
    print("Part 6 updated with Mermaid diagrams.")

if __name__ == "__main__":
    update_part1()
    update_part2()
    update_part3()
    update_part4()
    update_part5()
    update_part6()
