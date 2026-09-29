"""
Script to generate high-resolution visual demonstration figures for the OpenCV Engineering Study Guide.
Saves all visual artifacts to the `assets/` directory.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)

def save_fig(filename):
    plt.tight_layout()
    plt.savefig(os.path.join("assets", filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Generated] assets/{filename}")

# -------------------------------------------------------------
# 1. Fundamentals: Pixel Coordinate Grid & Drawing
# -------------------------------------------------------------
canvas = np.full((300, 400, 3), 245, dtype=np.uint8)
cv2.line(canvas, (50, 50), (350, 50), (200, 0, 0), 2, cv2.LINE_AA)
cv2.line(canvas, (50, 50), (50, 250), (0, 150, 0), 2, cv2.LINE_AA)
cv2.putText(canvas, "+X (Columns/Width)", (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1, cv2.LINE_AA)
cv2.putText(canvas, "+Y (Rows/Height)", (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1, cv2.LINE_AA)
cv2.circle(canvas, (200, 150), 50, (0, 0, 220), 2, cv2.LINE_AA)
cv2.rectangle(canvas, (100, 100), (300, 200), (120, 120, 120), 1)

fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
ax.set_title("OpenCV Image Coordinate System & Geometric Primitives", fontsize=11, pad=10)
ax.axis("off")
save_fig("01_opencv_fundamentals.png")

# -------------------------------------------------------------
# 2. NumPy Channel Splitting & ROI Slicing
# -------------------------------------------------------------
img_bgr = np.zeros((200, 300, 3), dtype=np.uint8)
img_bgr[:, :100] = (255, 0, 0)    # Blue
img_bgr[:, 100:200] = (0, 255, 0)  # Green
img_bgr[:, 200:] = (0, 0, 255)    # Red

b, g, r = cv2.split(img_bgr)
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
axs[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original (BGR Canvas)")
axs[1].imshow(b, cmap="Blues")
axs[1].set_title("Blue Channel")
axs[2].imshow(g, cmap="Greens")
axs[2].set_title("Green Channel")
axs[3].imshow(r, cmap="Reds")
axs[3].set_title("Red Channel")
for ax in axs: ax.axis("off")
save_fig("02_numpy_strides_channels.png")

# -------------------------------------------------------------
# 4. Color Spaces: BGR vs HSV vs Lab vs YCrCb
# -------------------------------------------------------------
color_scene = np.zeros((200, 300, 3), dtype=np.uint8)
color_scene[30:170, 30:130] = (0, 220, 220)   # Bright Yellow
color_scene[30:170, 170:270] = (0, 110, 110) # Shadowed Yellow
hsv = cv2.cvtColor(color_scene, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(color_scene, cv2.COLOR_BGR2Lab)
mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))

fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
axs[0].imshow(cv2.cvtColor(color_scene, cv2.COLOR_BGR2RGB))
axs[0].set_title("Input Scene (Light + Shadow)")
axs[1].imshow(hsv[:, :, 0], cmap="hsv")
axs[1].set_title("HSV Hue Plane (H)")
axs[2].imshow(lab[:, :, 0], cmap="gray")
axs[2].set_title("Lab Lightness (L*)")
axs[3].imshow(mask, cmap="gray")
axs[3].set_title("HSV Invariant Segmentation Mask")
for ax in axs: ax.axis("off")
save_fig("04_color_spaces.png")

# -------------------------------------------------------------
# 5. Image Manipulation: Bitwise Operations & Masking
# -------------------------------------------------------------
bg = np.full((200, 200, 3), 180, dtype=np.uint8)
cv2.putText(bg, "BACKGROUND", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
fg = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.circle(fg, (100, 100), 60, (0, 0, 255), -1)

fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(fg_gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
bg_cleared = cv2.bitwise_and(bg, bg, mask=mask_inv)
fg_extracted = cv2.bitwise_and(fg, fg, mask=mask)
blended = cv2.add(bg_cleared, fg_extracted)

fig, axs = plt.subplots(1, 5, figsize=(16, 3))
axs[0].imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)); axs[0].set_title("1. Background")
axs[1].imshow(cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)); axs[1].set_title("2. Foreground")
axs[2].imshow(mask, cmap="gray"); axs[2].set_title("3. Binary Mask")
axs[3].imshow(cv2.cvtColor(bg_cleared, cv2.COLOR_BGR2RGB)); axs[3].set_title("4. Masked BG")
axs[4].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)); axs[4].set_title("5. Seamless Result")
for ax in axs: ax.axis("off")
save_fig("05_bitwise_manipulation.png")

# -------------------------------------------------------------
# 6. Geometric Transformations: Affine & Perspective Warping
# -------------------------------------------------------------
grid = np.zeros((250, 250, 3), dtype=np.uint8)
cv2.rectangle(grid, (40, 40), (210, 210), (0, 255, 0), 3)
cv2.putText(grid, "TARGET", (55, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

M_rot = cv2.getRotationMatrix2D((125, 125), 30, 0.85)
affine_rot = cv2.warpAffine(grid, M_rot, (250, 250))

src_pts = np.float32([[40, 40], [210, 40], [240, 230], [10, 230]])
dst_pts = np.float32([[20, 20], [230, 20], [230, 230], [20, 230]])
H = cv2.getPerspectiveTransform(src_pts, dst_pts)
persp_warp = cv2.warpPerspective(grid, H, (250, 250))

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)); axs[0].set_title("Original Orthogonal Plane")
axs[1].imshow(cv2.cvtColor(affine_rot, cv2.COLOR_BGR2RGB)); axs[1].set_title("Affine (Rotation + Scale)")
axs[2].imshow(cv2.cvtColor(persp_warp, cv2.COLOR_BGR2RGB)); axs[2].set_title("Perspective Warp (Homography)")
for ax in axs: ax.axis("off")
save_fig("06_geometric_transforms.png")

# -------------------------------------------------------------
# 7. Image Filtering & Noise Attenuation
# -------------------------------------------------------------
clean = np.zeros((200, 200), dtype=np.uint8)
clean[:, 100:] = 255
noisy = clean.copy()
# Salt and pepper
for _ in range(300):
    noisy[np.random.randint(0, 199), np.random.randint(0, 199)] = 255
    noisy[np.random.randint(0, 199), np.random.randint(0, 199)] = 0

gauss = cv2.GaussianBlur(noisy, (5, 5), 1.5)
median = cv2.medianBlur(noisy, 5)
bilat = cv2.bilateralFilter(noisy, 9, 75, 75)

fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
axs[0].imshow(noisy, cmap="gray"); axs[0].set_title("Noisy Signal (Impulse)")
axs[1].imshow(gauss, cmap="gray"); axs[1].set_title("Gaussian Blur (Smeared)")
axs[2].imshow(median, cmap="gray"); axs[2].set_title("Median Filter (Restored)")
axs[3].imshow(bilat, cmap="gray"); axs[3].set_title("Bilateral Filter (Edge-Preserving)")
for ax in axs: ax.axis("off")
save_fig("07_filters_noise_reduction.png")

# -------------------------------------------------------------
# 8. Contrast Enhancement: Global HE vs CLAHE
# -------------------------------------------------------------
dark_img = np.full((200, 250), 35, dtype=np.uint8)
cv2.circle(dark_img, (70, 100), 40, 75, -1)
cv2.putText(dark_img, "LOW LIGHT", (120, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 65, 2)

global_he = cv2.equalizeHist(dark_img)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(dark_img)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
axs[0].imshow(dark_img, cmap="gray", vmin=0, vmax=255); axs[0].set_title("Original Underexposed")
axs[1].imshow(global_he, cmap="gray", vmin=0, vmax=255); axs[1].set_title("Global Equalize (Over-amplified)")
axs[2].imshow(clahe_img, cmap="gray", vmin=0, vmax=255); axs[2].set_title("Adaptive CLAHE (Controlled)")
for ax in axs: ax.axis("off")
save_fig("08_contrast_enhancement.png")

# -------------------------------------------------------------
# 9. Thresholding Comparison
# -------------------------------------------------------------
H, W = 200, 300
gradient = np.tile(np.linspace(30, 220, W, dtype=np.uint8), (H, 1))
scene_thresh = gradient.copy()
cv2.putText(scene_thresh, "TEXT", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)
cv2.putText(scene_thresh, "TEXT", (180, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)

_, global_t = cv2.threshold(scene_thresh, 127, 255, cv2.THRESH_BINARY_INV)
_, otsu_t = cv2.threshold(scene_thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
adapt_t = cv2.adaptiveThreshold(scene_thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

fig, axs = plt.subplots(1, 4, figsize=(15, 3.5))
axs[0].imshow(scene_thresh, cmap="gray"); axs[0].set_title("Non-Uniform Illumination")
axs[1].imshow(global_t, cmap="gray"); axs[1].set_title("Fixed Threshold (127)")
axs[2].imshow(otsu_t, cmap="gray"); axs[2].set_title("Otsu Global Bimodal")
axs[3].imshow(adapt_t, cmap="gray"); axs[3].set_title("Adaptive Gaussian (Clean)")
for ax in axs: ax.axis("off")
save_fig("09_thresholding_methods.png")

# -------------------------------------------------------------
# 10. Gradients & Edge Detection: Sobel vs Canny
# -------------------------------------------------------------
shape_img = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(shape_img, (100, 100), 60, 255, -1)
sobel_x = cv2.Sobel(shape_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(shape_img, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(sobel_x**2 + sobel_y**2)
canny_edges = cv2.Canny(shape_img, 50, 150)

fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
axs[0].imshow(np.abs(sobel_x), cmap="gray"); axs[0].set_title("Sobel X Derivative |dI/dx|")
axs[1].imshow(np.abs(sobel_y), cmap="gray"); axs[1].set_title("Sobel Y Derivative |dI/dy|")
axs[2].imshow(mag, cmap="magma"); axs[2].set_title("Gradient Magnitude ||grad I||")
axs[3].imshow(canny_edges, cmap="gray"); axs[3].set_title("Canny 1-Pixel Edges (NMS+Hyst)")
for ax in axs: ax.axis("off")
save_fig("10_gradients_edges.png")

# -------------------------------------------------------------
# 11. Morphology Operations
# -------------------------------------------------------------
morph_in = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(morph_in, (40, 40), (160, 160), 255, -1)
morph_in[20, 20] = 255 # noise speckle
morph_in[100, 100] = 0 # hole
k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
eroded = cv2.erode(morph_in, k)
dilated = cv2.dilate(morph_in, k)
opened = cv2.morphologyEx(morph_in, cv2.MORPH_OPEN, k)
closed = cv2.morphologyEx(morph_in, cv2.MORPH_CLOSE, k)

fig, axs = plt.subplots(1, 5, figsize=(16, 3))
axs[0].imshow(morph_in, cmap="gray"); axs[0].set_title("Original (Hole + Noise)")
axs[1].imshow(eroded, cmap="gray"); axs[1].set_title("Erosion")
axs[2].imshow(dilated, cmap="gray"); axs[2].set_title("Dilation")
axs[3].imshow(opened, cmap="gray"); axs[3].set_title("Opening (Noise Removed)")
axs[4].imshow(closed, cmap="gray"); axs[4].set_title("Closing (Hole Filled)")
for ax in axs: ax.axis("off")
save_fig("11_morphology_ops.png")

# -------------------------------------------------------------
# 12. Contours & Shape Analysis
# -------------------------------------------------------------
cont_img = np.zeros((250, 350, 3), dtype=np.uint8)
cv2.circle(cont_img, (80, 120), 45, (255, 255, 255), -1)
pts = np.array([[200, 60], [310, 100], [270, 200], [180, 150]], np.int32)
cv2.fillPoly(cont_img, [pts], (255, 255, 255))

gray_c = cv2.cvtColor(cont_img, cv2.COLOR_BGR2GRAY)
cnts, _ = cv2.findContours(gray_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vis_cont = cont_img.copy()

for c in cnts:
    M = cv2.moments(c)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(vis_cont, (cx, cy), 5, (0, 0, 255), -1) # Centroid
    # Axis-aligned bounding box
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(vis_cont, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Rotated bounding box
    rect = cv2.minAreaRect(c)
    box = np.intp(cv2.boxPoints(rect))
    cv2.drawContours(vis_cont, [box], 0, (0, 255, 0), 2)

fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(cv2.cvtColor(vis_cont, cv2.COLOR_BGR2RGB))
ax.set_title("Contour Analysis: Bounding Box (Blue), Rotated Grasp Box (Green), Centroid (Red)")
ax.axis("off")
save_fig("12_contours_shape.png")

# -------------------------------------------------------------
# 13. Hough Lines and Circles
# -------------------------------------------------------------
hough_canvas = np.zeros((250, 300, 3), dtype=np.uint8)
cv2.line(hough_canvas, (30, 220), (120, 50), (255, 255, 255), 3)
cv2.line(hough_canvas, (270, 220), (180, 50), (255, 255, 255), 3)
cv2.circle(hough_canvas, (150, 80), 30, (255, 255, 255), 3)

gray_h = cv2.cvtColor(hough_canvas, cv2.COLOR_BGR2GRAY)
edges_h = cv2.Canny(gray_h, 50, 150)
lines = cv2.HoughLinesP(edges_h, 1, np.pi/180, 30, minLineLength=40, maxLineGap=10)
circles = cv2.HoughCircles(gray_h, cv2.HOUGH_GRADIENT, 1.2, 40, param1=100, param2=15, minRadius=15, maxRadius=45)

vis_h = hough_canvas.copy()
if lines is not None:
    for l in lines:
        line_pts = l.ravel()
        if len(line_pts) == 4:
            x1, y1, x2, y2 = line_pts
            cv2.line(vis_h, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for c in circles[0, :]:
        cv2.circle(vis_h, (c[0], c[1]), c[2], (0, 0, 255), 2)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(cv2.cvtColor(hough_canvas, cv2.COLOR_BGR2RGB)); axs[0].set_title("Input Scene")
axs[1].imshow(cv2.cvtColor(vis_h, cv2.COLOR_BGR2RGB)); axs[1].set_title("Detected Lines (Green) & Circles (Red)")
for ax in axs: ax.axis("off")
save_fig("13_hough_detection.png")


# -------------------------------------------------------------
# 14. Feature Detection (Harris, FAST, ORB, SIFT)
# -------------------------------------------------------------
feat_img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(feat_img, (40, 40), (160, 160), 200, -1)
cv2.circle(feat_img, (100, 100), 35, 50, -1)

# Harris
dst_h = cv2.cornerHarris(feat_img, 2, 3, 0.04)
vis_harris = cv2.cvtColor(feat_img, cv2.COLOR_GRAY2BGR)
vis_harris[dst_h > 0.01 * dst_h.max()] = [0, 0, 255]

# FAST
fast = cv2.FastFeatureDetector_create()
kp_fast = fast.detect(feat_img, None)
vis_fast = cv2.drawKeypoints(feat_img, kp_fast, None, color=(0, 255, 0))

# ORB
orb = cv2.ORB_create(nfeatures=50)
kp_orb, _ = orb.detectAndCompute(feat_img, None)
vis_orb = cv2.drawKeypoints(feat_img, kp_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(cv2.cvtColor(vis_harris, cv2.COLOR_BGR2RGB)); axs[0].set_title("Harris Corner Response")
axs[1].imshow(cv2.cvtColor(vis_fast, cv2.COLOR_BGR2RGB)); axs[1].set_title("FAST Keypoints")
axs[2].imshow(cv2.cvtColor(vis_orb, cv2.COLOR_BGR2RGB)); axs[2].set_title("ORB (Scale + Orientation)")
for ax in axs: ax.axis("off")
save_fig("14_feature_detection.png")

# -------------------------------------------------------------
# 19. Optical Flow (Sparse LK and Dense Farneback)
# -------------------------------------------------------------
f1 = np.zeros((150, 150), dtype=np.uint8)
cv2.circle(f1, (50, 75), 20, 255, -1)
f2 = np.zeros((150, 150), dtype=np.uint8)
cv2.circle(f2, (65, 75), 20, 255, -1) # Moved right by 15px

# Dense Farneback
flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag_f, ang_f = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv_flow = np.zeros((150, 150, 3), dtype=np.uint8)
hsv_flow[..., 0] = ang_f * 180 / np.pi / 2
hsv_flow[..., 1] = 255
hsv_flow[..., 2] = cv2.normalize(mag_f, None, 0, 255, cv2.NORM_MINMAX)
rgb_flow = cv2.cvtColor(hsv_flow, cv2.COLOR_HSV2RGB)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
axs[0].imshow(f1, cmap="gray"); axs[0].set_title("Frame t0")
axs[1].imshow(f2, cmap="gray"); axs[1].set_title("Frame t1 (dx = +15px)")
axs[2].imshow(rgb_flow); axs[2].set_title("Farnebäck Optical Flow (HSV Direction)")
for ax in axs: ax.axis("off")
save_fig("19_optical_flow.png")

# -------------------------------------------------------------
# 22. Stereo Disparity & Depth
# -------------------------------------------------------------
left_s = np.full((150, 200), 80, dtype=np.uint8)
right_s = np.full((150, 200), 80, dtype=np.uint8)
tex = np.random.randint(0, 40, (150, 200), dtype=np.uint8)
left_s += tex; right_s += tex

cv2.circle(left_s, (110, 75), 30, 220, -1)
cv2.circle(right_s, (90, 75), 30, 220, -1) # 20px disparity

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=5)
disp = stereo.compute(left_s, right_s).astype(np.float32) / 16.0

fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
axs[0].imshow(left_s, cmap="gray"); axs[0].set_title("Left Rectified Camera")
axs[1].imshow(right_s, cmap="gray"); axs[1].set_title("Right Rectified Camera")
axs[2].imshow(disp, cmap="plasma"); axs[2].set_title("Triangulated Disparity (Depth Map)")
for ax in axs: ax.axis("off")
save_fig("22_stereo_depth.png")

# -------------------------------------------------------------
# 24. Image Segmentation (Watershed)
# -------------------------------------------------------------
w_img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.circle(w_img, (75, 100), 45, (255, 255, 255), -1)
cv2.circle(w_img, (125, 100), 45, (255, 255, 255), -1) # Touching coins

w_gray = cv2.cvtColor(w_img, cv2.COLOR_BGR2GRAY)
_, w_thresh = cv2.threshold(w_gray, 50, 255, cv2.THRESH_BINARY)
dist = cv2.distanceTransform(w_thresh, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
unknown = cv2.subtract(cv2.dilate(w_thresh, np.ones((3,3))), sure_fg)
markers[unknown == 255] = 0

markers_res = cv2.watershed(w_img.copy(), markers)
w_vis = w_img.copy()
w_vis[markers_res == -1] = [0, 0, 255] # Red ridge separation line

fig, axs = plt.subplots(1, 4, figsize=(15, 3.5))
axs[0].imshow(cv2.cvtColor(w_img, cv2.COLOR_BGR2RGB)); axs[0].set_title("Touching Objects")
axs[1].imshow(dist, cmap="magma"); axs[1].set_title("Distance Transform")
axs[2].imshow(sure_fg, cmap="gray"); axs[2].set_title("Isolated Center Seeds")
axs[3].imshow(cv2.cvtColor(w_vis, cv2.COLOR_BGR2RGB)); axs[3].set_title("Watershed Ridge Separation")
for ax in axs: ax.axis("off")
save_fig("24_watershed_coins.png")

# -------------------------------------------------------------
# 25. Connected Components & Blob Statistics
# -------------------------------------------------------------
cc_canvas = np.zeros((200, 300), dtype=np.uint8)
cv2.circle(cc_canvas, (60, 60), 30, 255, -1)
cv2.rectangle(cc_canvas, (150, 40), (260, 140), 255, -1)
cv2.circle(cc_canvas, (80, 150), 20, 255, -1)

num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(cc_canvas)
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[labels == 0] = 0

# Draw bounding boxes and centroids
for i in range(1, num_l):
    x, y, w, h = stats[i, :4]
    cx, cy = centroids[i]
    cv2.rectangle(labeled_img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.circle(labeled_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(cc_canvas, cmap="gray"); axs[0].set_title("Binary Input Mask")
axs[1].imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)); axs[1].set_title(f"Connected Components (N={num_l-1} Blobs)")
for ax in axs: ax.axis("off")
save_fig("25_connected_components.png")

# -------------------------------------------------------------
# 15. Feature Matching & Lowe's Ratio Test
# -------------------------------------------------------------
img_m1 = np.zeros((180, 240), dtype=np.uint8)
cv2.rectangle(img_m1, (40, 40), (140, 140), 200, -1)
cv2.putText(img_m1, "CV", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
M_rot_m = cv2.getRotationMatrix2D((120, 90), 25, 0.9)
img_m2 = cv2.warpAffine(img_m1, M_rot_m, (240, 180))

sift = cv2.SIFT_create()
kp_m1, des_m1 = sift.detectAndCompute(img_m1, None)
kp_m2, des_m2 = sift.detectAndCompute(img_m2, None)
bf = cv2.BFMatcher()
matches_raw = bf.knnMatch(des_m1, des_m2, k=2)
good_m = [m for m, n in matches_raw if m.distance < 0.75 * n.distance]
vis_match = cv2.drawMatches(img_m1, kp_m1, img_m2, kp_m2, good_m, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(cv2.cvtColor(vis_match, cv2.COLOR_BGR2RGB))
ax.set_title("SIFT Feature Matching with Lowe's Ratio Test Filtering")
ax.axis("off")
save_fig("15_feature_matching.png")

# -------------------------------------------------------------
# 21. Camera Pose & 3D Geometry (PnP Axes)
# -------------------------------------------------------------
pnp_canvas = np.full((250, 300, 3), 230, dtype=np.uint8)
# Draw cube in perspective
K_p = np.array([[300.0, 0, 150.0], [0, 300.0, 125.0], [0, 0, 1.0]])
d_p = np.zeros(5)
rvec_p = np.array([0.4, 0.6, 0.1])
tvec_p = np.array([0.0, 0.0, 1.5])
# 3D axes
axis_3d = np.float32([[0,0,0], [0.3,0,0], [0,0.3,0], [0,0,0.3]])
proj_axis, _ = cv2.projectPoints(axis_3d, rvec_p, tvec_p, K_p, d_p)
p_origin = tuple(np.int32(proj_axis[0].ravel()))
p_x = tuple(np.int32(proj_axis[1].ravel()))
p_y = tuple(np.int32(proj_axis[2].ravel()))
p_z = tuple(np.int32(proj_axis[3].ravel()))

cv2.line(pnp_canvas, p_origin, p_x, (0, 0, 255), 3, cv2.LINE_AA) # X-axis Red
cv2.line(pnp_canvas, p_origin, p_y, (0, 255, 0), 3, cv2.LINE_AA) # Y-axis Green
cv2.line(pnp_canvas, p_origin, p_z, (255, 0, 0), 3, cv2.LINE_AA) # Z-axis Blue
cv2.putText(pnp_canvas, "+X (Red)", p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.putText(pnp_canvas, "+Y (Green)", p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
cv2.putText(pnp_canvas, "+Z (Blue)", p_z, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(cv2.cvtColor(pnp_canvas, cv2.COLOR_BGR2RGB))
ax.set_title("Estimated 6-DoF Pose with Projected 3D Metric Coordinate Frame")
ax.axis("off")
save_fig("21_camera_pose_pnp.png")

# -------------------------------------------------------------
# 23. ArUco Marker Detection & Localization
# -------------------------------------------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_sample = cv2.aruco.generateImageMarker(aruco_dict, 12, 120)
aruco_scene = np.full((220, 280), 240, dtype=np.uint8)
aruco_scene[50:170, 80:200] = marker_sample
vis_aruco = cv2.cvtColor(aruco_scene, cv2.COLOR_GRAY2BGR)

detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
corners, ids, _ = detector.detectMarkers(aruco_scene)
if ids is not None:
    cv2.aruco.drawDetectedMarkers(vis_aruco, corners, ids)

fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(cv2.cvtColor(vis_aruco, cv2.COLOR_BGR2RGB))
ax.set_title("ArUco Marker Localization & ID Matrix Decoding (ID #12)")
ax.axis("off")
save_fig("23_aruco_markers.png")

print("All visual assets generated successfully!")

