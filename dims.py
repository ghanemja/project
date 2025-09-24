# # # import os
# # # import sys
# # # import cv2
# # # import numpy as np
# # # import math
# # # import torch
# # # from torch.autograd import Variable
# # # from skimage import io
# # # from skimage.transform import resize

# # # # Add MegaDepth repo to path
# # # sys.path.append('./megadepth')
# # # from options.train_options import TrainOptions
# # # from models.models import create_model

# # # # === CONFIG ===
# # # IMAGE_PATH = 'image.jpg'     # set your image path
# # # INPUT_HEIGHT = 384
# # # INPUT_WIDTH  = 512
# # # PREVIEW_WIDTH = 1280         # GUI preview width (downscaled display)
# # # ASSUME_HFOV_DEG = 63.0       # fallback HFOV if EXIF/VP fail

# # # # ---------------- GUI sanity ----------------
# # # def assert_gui_available():
# # #     try:
# # #         cv2.namedWindow(".__test__", cv2.WINDOW_NORMAL)
# # #         cv2.imshow(".__test__", np.zeros((2,2,3), np.uint8))
# # #         cv2.waitKey(1)
# # #         cv2.destroyWindow(".__test__")
# # #     except Exception as e:
# # #         raise RuntimeError(
# # #             f"OpenCV GUI not available. Error: {e}\n"
# # #             f"Check DISPLAY, OpenCV build, and X11 libs."
# # #         )

# # # # ---------------- MegaDepth ----------------
# # # def load_megadepth():
# # #     print("[INFO] Loading MegaDepth model...")
# # #     opt = TrainOptions().parse()
# # #     model = create_model(opt)
# # #     model.switch_to_eval()
# # #     return model

# # # def run_megadepth(model, image_path):
# # #     img = np.float32(io.imread(image_path)) / 255.0
# # #     H, W = img.shape[:2]
# # #     img_resized = resize(img, (INPUT_HEIGHT, INPUT_WIDTH),
# # #                          order=1, preserve_range=True, anti_aliasing=True)

# # #     input_img = torch.from_numpy(np.transpose(img_resized, (2,0,1))).float().unsqueeze(0)
# # #     if torch.cuda.is_available():
# # #         input_img = input_img.cuda()

# # #     with torch.no_grad():
# # #         pred_log_depth = model.netG.forward(Variable(input_img))
# # #         pred_log_depth = torch.squeeze(pred_log_depth)
# # #         pred_depth = torch.exp(pred_log_depth).cpu().numpy()  # relative depth (unitless scale)

# # #     depth_resized = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_LINEAR)
# # #     return depth_resized  # relative depth z_pred

# # # # ---------------- GUI helpers ----------------
# # # def pick_roi_scaled(image_bgr, window_name="Select ROI", preview_width=PREVIEW_WIDTH):
# # #     scale = preview_width / image_bgr.shape[1]
# # #     preview = cv2.resize(image_bgr, None, fx=scale, fy=scale)
# # #     print(f"[INFO] {window_name}: draw a box, press ENTER/SPACE to confirm.")
# # #     # Use OpenCV's built-in ROI tool on the scaled preview
# # #     roi_scaled = cv2.selectROI(window_name, preview, fromCenter=False, showCrosshair=True)
# # #     cv2.destroyWindow(window_name)
# # #     x, y, w, h = roi_scaled
# # #     if w <= 0 or h <= 0:
# # #         raise RuntimeError("Empty ROI. Please select a non-zero region.")
# # #     # Map back to full-res
# # #     x = int(round(x / scale)); y = int(round(y / scale))
# # #     w = int(round(w / scale)); h = int(round(h / scale))
# # #     return x, y, w, h

# # # def pick_two_points(image_bgr, preview_width=PREVIEW_WIDTH):
# # #     scale = preview_width / image_bgr.shape[1]
# # #     preview = cv2.resize(image_bgr, None, fx=scale, fy=scale)

# # #     pts_scaled = []
# # #     disp = preview.copy()
# # #     print("[INFO] Click TWO points to measure distance. Press any key when done.")

# # #     def cb(event, x, y, flags, param):
# # #         if event == cv2.EVENT_LBUTTONDOWN and len(pts_scaled) < 2:
# # #             pts_scaled.append((x, y))
# # #             cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
# # #             cv2.imshow("Pick 2 Points", disp)

# # #     cv2.namedWindow("Pick 2 Points", cv2.WINDOW_NORMAL)
# # #     cv2.resizeWindow("Pick 2 Points", preview_width, int(preview.shape[0]))
# # #     cv2.imshow("Pick 2 Points", disp)
# # #     cv2.setMouseCallback("Pick 2 Points", cb)
# # #     cv2.waitKey(0)
# # #     cv2.destroyWindow("Pick 2 Points")

# # #     if len(pts_scaled) != 2:
# # #         raise RuntimeError(f"Expected 2 points, got {len(pts_scaled)}")

# # #     pts_full = [(int(x / scale), int(y / scale)) for (x, y) in pts_scaled]
# # #     return pts_full

# # # # ---------------- Calibration pieces ----------------
# # # def roi_median_depth(rel_depth, roi):
# # #     x, y, w, h = roi
# # #     region = rel_depth[y:y+h, x:x+w]
# # #     region = region[np.isfinite(region)]
# # #     if region.size == 0:
# # #         raise RuntimeError("Selected ROI has no valid depth.")
# # #     med = float(np.median(region))
# # #     if med <= 0:
# # #         raise RuntimeError(f"Median relative depth <= 0 in ROI: {med}")
# # #     return med

# # # def exif_fx_from_meta(image_path, image_width_px, sensor_width_mm=None):
# # #     """
# # #     Try to compute fx (pixels) from EXIF.
# # #     - If FocalLengthIn35mmFilm exists: fx = (W/36) * f_35mm
# # #     - Else if FocalLength and sensor_width_mm known: fx = f_mm * (W / sensor_width_mm)
# # #     Returns fx or None.
# # #     """
# # #     try:
# # #         from PIL import Image, ExifTags
# # #     except Exception:
# # #         return None

# # #     try:
# # #         img = Image.open(image_path)
# # #         raw = img._getexif() or {}
# # #         exif = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
# # #     except Exception:
# # #         return None

# # #     f_mm = None
# # #     if 'FocalLength' in exif:
# # #         val = exif['FocalLength']
# # #         if isinstance(val, tuple):
# # #             f_mm = float(val[0]) / float(val[1] if val[1] else 1)
# # #         else:
# # #             f_mm = float(val)

# # #     if 'FocalLengthIn35mmFilm' in exif:
# # #         f35 = float(exif['FocalLengthIn35mmFilm'])
# # #         return (image_width_px / 36.0) * f35

# # #     if sensor_width_mm and f_mm:
# # #         return f_mm * (image_width_px / sensor_width_mm)

# # #     return None

# # # def estimate_fx_from_vanishing_points(image_bgr):
# # #     """
# # #     Rough single-image focal estimate from two orthogonal vanishing points.
# # #     Assumes principal point at image center, square pixels.
# # #     """
# # #     H, W = image_bgr.shape[:2]
# # #     cx, cy = W/2.0, H/2.0

# # #     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# # #     lsd = cv2.createLineSegmentDetector()
# # #     lines = lsd.detect(gray)[0]
# # #     if lines is None or len(lines) < 50:
# # #         raise RuntimeError("Not enough line segments for VP.")

# # #     # homogeneous lines
# # #     L = []
# # #     for x1,y1,x2,y2 in lines[:,0,:]:
# # #         p1 = np.array([x1,y1,1.0]); p2 = np.array([x2,y2,1.0])
# # #         l = np.cross(p1, p2)
# # #         n = np.linalg.norm(l[:2])
# # #         if n < 1e-6: continue
# # #         L.append(l / n)
# # #     L = np.array(L)
# # #     if len(L) < 30:
# # #         raise RuntimeError("Too few normalized lines.")

# # #     # random intersections as VP candidates
# # #     rng = np.random.default_rng(0)
# # #     candidates = []
# # #     for _ in range(3000):
# # #         i, j = rng.integers(0, len(L), size=2)
# # #         if i == j: continue
# # #         v = np.cross(L[i], L[j])  # intersection
# # #         if abs(v[2]) < 1e-9: continue
# # #         vx, vy = v[0]/v[2], v[1]/v[2]
# # #         if -10*W < vx < 11*W and -10*H < vy < 11*H:
# # #             candidates.append((vx, vy))
# # #     candidates = np.array(candidates, dtype=np.float32)
# # #     if len(candidates) < 100:
# # #         raise RuntimeError("Not enough VP candidates.")

# # #     # cluster to 2 main VPs (for two orthogonal directions)
# # #     K = 2
# # #     criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-3)
# # #     attempts = 5
# # #     flags = cv2.KMEANS_PP_CENTERS
# # #     compactness, labels, centers = cv2.kmeans(candidates, K, None, criteria, attempts, flags)
# # #     v1, v2 = centers  # (x,y)

# # #     # f^2 = - (v1 - c) . (v2 - c)  (orthogonality constraint)
# # #     d = -((v1[0]-cx)*(v2[0]-cx) + (v1[1]-cy)*(v2[1]-cy))
# # #     if d <= 0:
# # #         raise RuntimeError("VP orthogonality produced non-positive f^2.")
# # #     f = float(np.sqrt(d))
# # #     return f

# # # def fx_from_hfov(width_px, hfov_deg=ASSUME_HFOV_DEG):
# # #     return width_px / (2.0 * math.tan(math.radians(hfov_deg/2.0)))

# # # def estimate_focal_px(image_bgr, image_path):
# # #     H, W = image_bgr.shape[:2]
# # #     # 1) EXIF
# # #     fx = exif_fx_from_meta(image_path, W, sensor_width_mm=None)
# # #     if fx:
# # #         print(f"[INFO] Using EXIF-based fx ≈ {fx:.1f} px")
# # #         return fx
# # #     # 2) Vanishing points
# # #     try:
# # #         fx_vp = estimate_fx_from_vanishing_points(image_bgr)
# # #         print(f"[INFO] Using VP-based fx ≈ {fx_vp:.1f} px")
# # #         return fx_vp
# # #     except Exception as e:
# # #         print(f"[WARN] VP-based focal failed: {e}")
# # #     # 3) HFOV fallback
# # #     fx_guess = fx_from_hfov(W, hfov_deg=ASSUME_HFOV_DEG)
# # #     print(f"[WARN] Falling back to HFOV guess fx ≈ {fx_guess:.1f} px ({ASSUME_HFOV_DEG}°)")
# # #     return fx_guess

# # # # ---------------- Intrinsics & backprojection ----------------
# # # def build_K(image_shape, fx, fy=None, cx=None, cy=None):
# # #     H, W = image_shape[:2]
# # #     if fy is None: fy = fx
# # #     if cx is None: cx = W/2.0
# # #     if cy is None: cy = H/2.0
# # #     K = np.array([[fx, 0, cx],
# # #                   [0, fy, cy],
# # #                   [0,  0,  1]], dtype=np.float32)
# # #     return K

# # # def backproject_uv_to_xyz(points_uv, depth_m, K):
# # #     fx, fy = K[0,0], K[1,1]
# # #     cx, cy = K[0,2], K[1,2]
# # #     xyz = []
# # #     for (u, v) in points_uv:
# # #         u_i, v_i = int(round(u)), int(round(v))
# # #         if v_i < 0 or u_i < 0 or v_i >= depth_m.shape[0] or u_i >= depth_m.shape[1]:
# # #             raise RuntimeError(f"Point {(u, v)} outside image.")
# # #         Z = float(depth_m[v_i, u_i])
# # #         if not np.isfinite(Z) or Z <= 0:
# # #             raise RuntimeError(f"Invalid depth at {(u_i, v_i)}: {Z}")
# # #         X = (u - cx) * Z / fx
# # #         Y = (v - cy) * Z / fy
# # #         xyz.append((X, Y, Z))
# # #     return np.array(xyz, dtype=np.float32)

# # # # ---------------- Main flow ----------------
# # # def main():
# # #     assert_gui_available()

# # #     image_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
# # #     if image_bgr is None:
# # #         print(f"[ERROR] Image not found: {IMAGE_PATH}")
# # #         return
# # #     H, W = image_bgr.shape[:2]

# # #     model = load_megadepth()
# # #     print("[INFO] Predicting relative depth...")
# # #     rel_depth = run_megadepth(model, IMAGE_PATH)

# # #     # --- Pick TWO reference objects (known real heights) ---
# # #     # ROI 1
# # #     roi1 = pick_roi_scaled(image_bgr, "Select Reference Object 1")
# # #     H1 = float(input("[INPUT] Enter real HEIGHT (meters) for object 1: ").strip())
# # #     z1 = roi_median_depth(rel_depth, roi1)
# # #     hpx1 = roi1[3]  # box height in pixels
# # #     gamma1 = (hpx1 * z1) / max(1e-9, H1)  # gamma = f/k
# # #     print(f"[INFO] Obj1: h_px={hpx1}, z_rel={z1:.6f}, H={H1} m -> gamma1={gamma1:.6f}")

# # #     # ROI 2
# # #     roi2 = pick_roi_scaled(image_bgr, "Select Reference Object 2")
# # #     H2 = float(input("[INPUT] Enter real HEIGHT (meters) for object 2: ").strip())
# # #     z2 = roi_median_depth(rel_depth, roi2)
# # #     hpx2 = roi2[3]
# # #     gamma2 = (hpx2 * z2) / max(1e-9, H2)
# # #     print(f"[INFO] Obj2: h_px={hpx2}, z_rel={z2:.6f}, H={H2} m -> gamma2={gamma2:.6f}")

# # #     # Consistency check: for perfect model gamma1 ≈ gamma2
# # #     gammas = np.array([gamma1, gamma2], dtype=np.float64)
# # #     gamma = float(np.median(gammas))
# # #     print(f"[INFO] gamma estimates: {gammas}  -> gamma(median)={gamma:.6f}  (gamma = f_px / k)")

# # #     # --- Estimate focal length in pixels (EXIF -> VP -> HFOV fallback) ---
# # #     fpx = estimate_focal_px(image_bgr, IMAGE_PATH)
# # #     print(f"[INFO] Estimated focal length fx ≈ {fpx:.2f} px")

# # #     # --- Recover global depth scale k and metric depth ---
# # #     k = fpx / gamma
# # #     print(f"[INFO] Recovered global scale k = f/gamma ≈ {k:.6f} meters per relative-depth unit")
# # #     metric_depth = rel_depth * k

# # #     # --- Build intrinsics and measure distance ---
# # #     K = build_K(image_bgr.shape, fx=fpx)
# # #     pts_uv = pick_two_points(image_bgr, PREVIEW_WIDTH)
# # #     pts_xyz = backproject_uv_to_xyz(pts_uv, metric_depth, K)
# # #     dist_m = float(np.linalg.norm(pts_xyz[0] - pts_xyz[1]))

# # #     print(f"[RESULT] 3D distance: {dist_m:.3f} meters")
# # #     print(f"[DEBUG] P1 (u,v)->(X,Y,Z): {pts_uv[0]} -> {pts_xyz[0]}")
# # #     print(f"[DEBUG] P2 (u,v)->(X,Y,Z): {pts_uv[1]} -> {pts_xyz[1]}")
# # #     print(f"[DEBUG] fx(px)={fpx:.2f}, gamma={gamma:.6f}, k={k:.6f}")

# # # if __name__ == "__main__":
# # #     main()
# # import os
# # import sys
# # import cv2
# # import math
# # import numpy as np
# # import torch
# # from torch.autograd import Variable
# # from skimage import io
# # from skimage.transform import resize

# # # Add MegaDepth repo to path
# # sys.path.append('./megadepth')
# # from options.train_options import TrainOptions
# # from models.models import create_model

# # # === CONFIG ===
# # IMAGE_PATH = 'image.jpg'
# # INPUT_HEIGHT = 384
# # INPUT_WIDTH  = 512
# # PREVIEW_WIDTH = 1280
# # ASSUME_HFOV_DEG = 63.0  # fallback HFOV if EXIF/VP fail
# # ROI_JITTER_FRAC = 0.06  # jitter ROI by ±6% for stability sampling
# # TRIM_PERCENT = 10       # trim depth extremes inside ROI (percent each tail)

# # # ---------------- GUI sanity ----------------
# # def assert_gui_available():
# #     try:
# #         cv2.namedWindow(".__test__", cv2.WINDOW_NORMAL)
# #         cv2.imshow(".__test__", np.zeros((2,2,3), np.uint8))
# #         cv2.waitKey(1)
# #         cv2.destroyWindow(".__test__")
# #     except Exception as e:
# #         raise RuntimeError(
# #             f"OpenCV GUI not available. Error: {e}\n"
# #             f"Check DISPLAY, OpenCV build, and X11 libs."
# #         )

# # # ---------------- MegaDepth ----------------
# # def load_megadepth():
# #     print("[INFO] Loading MegaDepth model...")
# #     opt = TrainOptions().parse()
# #     model = create_model(opt)
# #     model.switch_to_eval()
# #     return model

# # def run_megadepth(model, image_path):
# #     img = np.float32(io.imread(image_path)) / 255.0
# #     H, W = img.shape[:2]
# #     img_resized = resize(img, (INPUT_HEIGHT, INPUT_WIDTH),
# #                          order=1, preserve_range=True, anti_aliasing=True)
# #     input_img = torch.from_numpy(np.transpose(img_resized, (2,0,1))).float().unsqueeze(0)
# #     if torch.cuda.is_available():
# #         input_img = input_img.cuda()
# #     with torch.no_grad():
# #         pred_log = model.netG.forward(Variable(input_img))
# #         pred_log = torch.squeeze(pred_log)
# #         pred_depth = torch.exp(pred_log).cpu().numpy()
# #     depth_resized = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_LINEAR)
# #     return depth_resized  # relative depth

# # # ---------------- GUI helpers ----------------
# # def pick_roi_scaled(image_bgr, window_name="Select ROI", preview_width=PREVIEW_WIDTH):
# #     scale = preview_width / image_bgr.shape[1]
# #     preview = cv2.resize(image_bgr, None, fx=scale, fy=scale)
# #     print(f"[INFO] {window_name}: draw a box, press ENTER/SPACE to confirm.")
# #     roi_scaled = cv2.selectROI(window_name, preview, fromCenter=False, showCrosshair=True)
# #     cv2.destroyWindow(window_name)
# #     x, y, w, h = roi_scaled
# #     if w <= 0 or h <= 0:
# #         raise RuntimeError("Empty ROI. Please select a non-zero region.")
# #     # map back to full-res
# #     x = int(round(x / scale)); y = int(round(y / scale))
# #     w = int(round(w / scale)); h = int(round(h / scale))
# #     # clamp into bounds
# #     H, W = image_bgr.shape[:2]
# #     x = max(0, min(W-1, x))
# #     y = max(0, min(H-1, y))
# #     w = max(1, min(W-x, w))
# #     h = max(1, min(H-y, h))
# #     return (x, y, w, h)

# # def pick_two_points(image_bgr, preview_width=PREVIEW_WIDTH):
# #     scale = preview_width / image_bgr.shape[1]
# #     preview = cv2.resize(image_bgr, None, fx=scale, fy=scale)
# #     pts_scaled = []
# #     disp = preview.copy()
# #     print("[INFO] Click TWO points to measure distance. Press any key when done.")
# #     def cb(event, x, y, flags, param):
# #         if event == cv2.EVENT_LBUTTONDOWN and len(pts_scaled) < 2:
# #             pts_scaled.append((x, y))
# #             cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
# #             cv2.imshow("Pick 2 Points", disp)
# #     cv2.namedWindow("Pick 2 Points", cv2.WINDOW_NORMAL)
# #     cv2.resizeWindow("Pick 2 Points", preview_width, int(preview.shape[0]))
# #     cv2.imshow("Pick 2 Points", disp)
# #     cv2.setMouseCallback("Pick 2 Points", cb)
# #     cv2.waitKey(0)
# #     cv2.destroyWindow("Pick 2 Points")
# #     if len(pts_scaled) != 2:
# #         raise RuntimeError(f"Expected 2 points, got {len(pts_scaled)}")
# #     pts_full = [(int(x / scale), int(y / scale)) for (x, y) in pts_scaled]
# #     return pts_full

# # # ---------------- Pixel-height from ROI (robust, good for curved objects) ----------------
# # def object_pixel_height_from_roi(image_bgr, roi):
# #     """
# #     Estimate object pixel height inside ROI using edges+contours (better than raw box height).
# #     Steps: Canny -> dilate -> largest contour -> vertical extent.
# #     """
# #     x, y, w, h = roi
# #     crop = image_bgr[y:y+h, x:x+w]
# #     if crop.size == 0:
# #         return h  # fallback
# #     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
# #     # adaptive blur to reduce noise
# #     sigma = 0.33
# #     med = np.median(gray)
# #     lower = int(max(0, (1.0 - sigma) * med))
# #     upper = int(min(255, (1.0 + sigma) * med))
# #     edges = cv2.Canny(gray, lower, upper)
# #     # connect fragments
# #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# #     edges = cv2.dilate(edges, kernel, iterations=1)
# #     # find contours
# #     cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     if not cnts:
# #         return h  # fallback to ROI height
# #     # pick largest contour by area
# #     c = max(cnts, key=cv2.contourArea)
# #     ys = c[:,:,1].flatten()
# #     pix_h = int(ys.max() - ys.min() + 1)
# #     # clamp to ROI height (avoid weird overshoots)
# #     pix_h = int(np.clip(pix_h, 1, h))
# #     return pix_h

# # # ---------------- Robust depth from ROI ----------------
# # def local_median_depth(depth, u, v, win=7):
# #     """Median depth in a (win x win) window around (u,v)."""
# #     h, w = depth.shape
# #     x0 = max(0, int(u) - win//2)
# #     y0 = max(0, int(v) - win//2)
# #     x1 = min(w, int(u) + win//2 + 1)
# #     y1 = min(h, int(v) + win//2 + 1)
# #     patch = depth[y0:y1, x0:x1]
# #     vals = patch[np.isfinite(patch) & (patch > 0)]
# #     if vals.size == 0:
# #         return float('nan')
# #     return float(np.median(vals))

# # def measure_two_points_robust(image_bgr, metric_depth, K, preview_width=1280, win=7, reps=3, jitter_px=2):
# #     """
# #     Click two points on a downscaled preview; for each point, take multiple
# #     jittered samples and use median depth for stability.
# #     """
# #     # reuse your pick_two_points
# #     pts_uv = pick_two_points(image_bgr, preview_width)
# #     fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]

# #     def robust_xyz(u, v):
# #         samples = []
# #         # center + small jitters
# #         for du in range(-jitter_px, jitter_px+1, max(1, jitter_px//max(1, reps-1))):
# #             for dv in range(-jitter_px, jitter_px+1, max(1, jitter_px//max(1, reps-1))):
# #                 Z = local_median_depth(metric_depth, u+du, v+dv, win=win)
# #                 if np.isfinite(Z) and Z > 0:
# #                     X = ((u+du) - cx) * Z / fx
# #                     Y = ((v+dv) - cy) * Z / fy
# #                     samples.append((X, Y, Z))
# #         if not samples:
# #             raise RuntimeError(f"No valid depth around ({u},{v})")
# #         P = np.array(samples, dtype=np.float32)
# #         return np.median(P, axis=0)

# #     P1 = robust_xyz(*pts_uv[0])
# #     P2 = robust_xyz(*pts_uv[1])
# #     dist = float(np.linalg.norm(P1 - P2))
# #     print(f"[RESULT] Robust 3D distance: {dist:.3f} meters")
# #     print(f"[DEBUG] P1 (u,v)->(X,Y,Z): {pts_uv[0]} -> {P1}")
# #     print(f"[DEBUG] P2 (u,v)->(X,Y,Z): {pts_uv[1]} -> {P2}")
# #     return dist, pts_uv, P1, P2



# # def robust_depth_from_roi(rel_depth, roi, trim_percent=TRIM_PERCENT, jitter_frac=ROI_JITTER_FRAC):
# #     """
# #     Robust relative depth at ROI using trimmed median and ROI jitter sampling.
# #     Returns (z_med, z_std, samples_count)
# #     """
# #     H, W = rel_depth.shape
# #     x, y, w, h = roi
# #     # base region values
# #     def region_vals(rx, ry, rw, rh):
# #         rx = max(0, min(W-1, rx)); ry = max(0, min(H-1, ry))
# #         rw = max(1, min(W-rx, rw)); rh = max(1, min(H-ry, rh))
# #         r = rel_depth[ry:ry+rh, rx:rx+rw]
# #         r = r[np.isfinite(r)]
# #         return r

# #     samples = []
# #     # nominal patch
# #     vals = region_vals(x, y, w, h)
# #     if vals.size:
# #         vals = np.sort(vals)
# #         k = int(len(vals) * trim_percent / 100.0)
# #         vals = vals[k:len(vals)-k] if len(vals) > 2*k else vals
# #         if vals.size: samples.append(np.median(vals))

# #     # jittered patches
# #     shifts = [-jitter_frac, 0.0, jitter_frac]
# #     for dx in shifts:
# #         for dy in shifts:
# #             if dx == 0.0 and dy == 0.0: continue
# #             rx = int(round(x + dx * w))
# #             ry = int(round(y + dy * h))
# #             vals = region_vals(rx, ry, w, h)
# #             if vals.size:
# #                 vals = np.sort(vals)
# #                 k = int(len(vals) * trim_percent / 100.0)
# #                 vals = vals[k:len(vals)-k] if len(vals) > 2*k else vals
# #                 if vals.size: samples.append(np.median(vals))

# #     if not samples:
# #         raise RuntimeError("Selected ROI has no valid depth.")
# #     z_arr = np.array(samples, dtype=np.float64)
# #     return float(np.median(z_arr)), float(np.std(z_arr)), len(z_arr)

# # # ---------------- Focal estimation helpers ----------------
# # def exif_fx_from_meta(image_path, image_width_px, sensor_width_mm=None):
# #     try:
# #         from PIL import Image, ExifTags
# #     except Exception:
# #         return None
# #     try:
# #         img = Image.open(image_path)
# #         raw = img._getexif() or {}
# #         exif = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
# #     except Exception:
# #         return None
# #     f_mm = None
# #     if 'FocalLength' in exif:
# #         val = exif['FocalLength']
# #         if isinstance(val, tuple):
# #             f_mm = float(val[0]) / float(val[1] if val[1] else 1)
# #         else:
# #             f_mm = float(val)
# #     if 'FocalLengthIn35mmFilm' in exif:
# #         f35 = float(exif['FocalLengthIn35mmFilm'])
# #         return (image_width_px / 36.0) * f35
# #     if sensor_width_mm and f_mm:
# #         return f_mm * (image_width_px / sensor_width_mm)
# #     return None

# # def estimate_fx_from_vanishing_points(image_bgr):
# #     H, W = image_bgr.shape[:2]; cx, cy = W/2.0, H/2.0
# #     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# #     lsd = cv2.createLineSegmentDetector()
# #     lines = lsd.detect(gray)[0]
# #     if lines is None or len(lines) < 50:
# #         raise RuntimeError("Not enough line segments for VP.")
# #     L = []
# #     for x1,y1,x2,y2 in lines[:,0,:]:
# #         p1 = np.array([x1,y1,1.0]); p2 = np.array([x2,y2,1.0])
# #         l = np.cross(p1, p2); n = np.linalg.norm(l[:2])
# #         if n < 1e-6: continue
# #         L.append(l / n)
# #     L = np.array(L); 
# #     if len(L) < 30: raise RuntimeError("Too few normalized lines.")

# #     rng = np.random.default_rng(0); candidates = []
# #     for _ in range(3000):
# #         i, j = rng.integers(0, len(L), size=2)
# #         if i == j: continue
# #         v = np.cross(L[i], L[j])
# #         if abs(v[2]) < 1e-9: continue
# #         vx, vy = v[0]/v[2], v[1]/v[2]
# #         if -10*W < vx < 11*W and -10*H < vy < 11*H:
# #             candidates.append((vx, vy))
# #     candidates = np.array(candidates, dtype=np.float32)
# #     if len(candidates) < 100:
# #         raise RuntimeError("Not enough VP candidates.")

# #     K = 2
# #     criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-3)
# #     compactness, labels, centers = cv2.kmeans(candidates, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
# #     v1, v2 = centers
# #     d = -((v1[0]-cx)*(v2[0]-cx) + (v1[1]-cy)*(v2[1]-cy))
# #     if d <= 0: raise RuntimeError("VP orthogonality produced non-positive f^2.")
# #     return float(np.sqrt(d))

# # def fx_from_hfov(width_px, hfov_deg):
# #     return width_px / (2.0 * math.tan(math.radians(hfov_deg/2.0)))

# # def estimate_focal_px(image_bgr, image_path):
# #     H, W = image_bgr.shape[:2]
# #     fx = exif_fx_from_meta(image_path, W, sensor_width_mm=None)
# #     if fx:
# #         print(f"[INFO] Using EXIF-based fx ≈ {fx:.1f} px")
# #         return fx
# #     try:
# #         fx_vp = estimate_fx_from_vanishing_points(image_bgr)
# #         print(f"[INFO] Using VP-based fx ≈ {fx_vp:.1f} px")
# #         return fx_vp
# #     except Exception as e:
# #         print(f"[WARN] VP-based focal failed: {e}")
# #     fx_guess = fx_from_hfov(W, hfov_deg=ASSUME_HFOV_DEG)
# #     print(f"[WARN] Falling back to HFOV guess fx ≈ {fx_guess:.1f} px ({ASSUME_HFOV_DEG}°)")
# #     return fx_guess

# # # ---------------- Intrinsics & backprojection ----------------
# # def build_K(image_shape, fx, fy=None, cx=None, cy=None):
# #     H, W = image_shape[:2]
# #     if fy is None: fy = fx
# #     if cx is None: cx = W/2.0
# #     if cy is None: cy = H/2.0
# #     return np.array([[fx, 0, cx],[0, fy, cy],[0,  0,  1]], dtype=np.float32)

# # def backproject_uv_to_xyz(points_uv, depth_m, K):
# #     fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
# #     xyz = []
# #     for (u, v) in points_uv:
# #         u_i, v_i = int(round(u)), int(round(v))
# #         if not (0 <= u_i < depth_m.shape[1] and 0 <= v_i < depth_m.shape[0]):
# #             raise RuntimeError(f"Point {(u, v)} outside image.")
# #         Z = float(depth_m[v_i, u_i])
# #         if not np.isfinite(Z) or Z <= 0:
# #             raise RuntimeError(f"Invalid depth at {(u_i, v_i)}: {Z}")
# #         X = (u - cx) * Z / fx; Y = (v - cy) * Z / fy
# #         xyz.append((X, Y, Z))
# #     return np.array(xyz, dtype=np.float32)

# # # ---------------- Robust aggregation ----------------
# # def mad(arr):
# #     med = np.median(arr)
# #     return med, np.median(np.abs(arr - med))

# # def main():
# #     assert_gui_available()

# #     image_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
# #     if image_bgr is None:
# #         print(f"[ERROR] Image not found: {IMAGE_PATH}")
# #         return

# #     model = load_megadepth()
# #     print("[INFO] Predicting relative depth...")
# #     rel_depth = run_megadepth(model, IMAGE_PATH)

# #     # --- Choose how many reference objects ---
# #     try:
# #         N = int(input("[INPUT] How many reference objects (>=2)? ").strip())
# #     except Exception:
# #         N = 2
# #     N = max(2, N)

# #     gammas = []
# #     per_obj_info = []

# #     for i in range(N):
# #         roi = pick_roi_scaled(image_bgr, f"Select Reference Object {i+1}")
# #         # robust pixel height (curved objects handled better by contour extent)
# #         hpx = object_pixel_height_from_roi(image_bgr, roi)
# #         # robust relative depth with trimming + jitter stability
# #         z_med, z_std, n_samp = robust_depth_from_roi(rel_depth, roi)
# #         # ask for true object HEIGHT in meters
# #         while True:
# #             try:
# #                 Hi = float(input(f"[INPUT] Enter real HEIGHT (meters) for object {i+1}: ").strip())
# #                 if Hi > 0: break
# #             except Exception:
# #                 pass
# #             print("  Please enter a positive number (meters).")
# #         gamma_i = (hpx * z_med) / max(1e-9, Hi)
# #         gammas.append(gamma_i)
# #         per_obj_info.append({
# #             'roi': roi, 'hpx': hpx, 'z_med': z_med, 'z_std': z_std, 'n_samp': n_samp, 'H': Hi, 'gamma': gamma_i
# #         })
# #         print(f"[INFO] Obj{i+1}: h_px={hpx}, z_rel_med={z_med:.6f} (±{z_std:.6f}, n={n_samp}), H={Hi} m -> gamma={gamma_i:.6f}")

# #     gammas = np.array(gammas, dtype=np.float64)
# #     g_med, g_mad = mad(gammas)
# #     # outlier rejection: keep within 2.5 * MAD
# #     if g_mad < 1e-9:
# #         mask = np.ones_like(gammas, dtype=bool)
# #     else:
# #         mask = np.abs(gammas - g_med) <= (2.5 * g_mad)
# #     kept = gammas[mask]
# #     print(f"[INFO] gamma raw: {gammas}")
# #     print(f"[INFO] gamma median={g_med:.6f}, MAD={g_mad:.6f}, kept={len(kept)}/{len(gammas)}")

# #     if len(kept) < 2:
# #         print("[WARN] Fewer than 2 inliers after filtering; using all gammas.")
# #         kept = gammas

# #     gamma = float(np.median(kept))
# #     print(f"[INFO] gamma (robust) = {gamma:.6f}  (gamma = f_px / k)")

# #     # --- Estimate focal length in pixels ---
# #     fpx = estimate_focal_px(image_bgr, IMAGE_PATH)
# #     print(f"[INFO] Estimated focal length fx ≈ {fpx:.2f} px")

# #     # --- Recover scale and metric depth ---
# #     k = fpx / gamma
# #     metric_depth = rel_depth * k
# #     measure_two_points_robust(image_bgr, metric_depth, k, preview_width=1280, win=9, reps=3, jitter_px=2)
# #     print(f"[INFO] Global scale k = f/gamma ≈ {k:.6f} m per relative-depth unit")

# #     # --- Build K and measure ---
# #     K = build_K(image_bgr.shape, fx=fpx)
# #     pts_uv = pick_two_points(image_bgr, PREVIEW_WIDTH)
# #     pts_xyz = backproject_uv_to_xyz(pts_uv, metric_depth, K)
# #     dist_m = float(np.linalg.norm(pts_xyz[0] - pts_xyz[1]))
# #     print(f"[RESULT] 3D distance: {dist_m:.3f} meters")
# #     print(f"[DEBUG] P1 (u,v)->(X,Y,Z): {pts_uv[0]} -> {pts_xyz[0]}")
# #     print(f"[DEBUG] P2 (u,v)->(X,Y,Z): {pts_uv[1]} -> {pts_xyz[1]}")
# #     print(f"[DEBUG] fx(px)={fpx:.2f}, gamma={gamma:.6f}, k={k:.6f}")

# #     # --- Stability / sanity report on reference objects ---
# #     print("\n[REPORT] Reference object reprojection check:")
# #     for i, info in enumerate(per_obj_info, 1):
# #         x,y,w,h = info['roi']
# #         H_true = info['H']
# #         z_rel = info['z_med']
# #         Z_metric = z_rel * k
# #         h_pred = (fpx * H_true) / max(1e-9, Z_metric)  # predicted pixel height by pinhole
# #         err_px = h_pred - info['hpx']
# #         err_pct = 100.0 * err_px / max(1e-9, info['hpx'])
# #         print(f"  Obj{i}: h_px_meas={info['hpx']}, h_px_pred={h_pred:.2f}, "
# #               f"err={err_px:.2f}px ({err_pct:+.1f}%), z_rel±std={z_rel:.4f}±{info['z_std']:.4f}")

# # if __name__ == "__main__":
# #     main()
# import os
# import sys
# import cv2
# import math
# import numpy as np
# import torch
# from torch.autograd import Variable
# from skimage import io
# from skimage.transform import resize

# # Add MegaDepth repo to path
# sys.path.append('./megadepth')
# from options.train_options import TrainOptions
# from models.models import create_model

# # === CONFIG ===
# IMAGE_PATH = 'image.jpg'
# INPUT_HEIGHT = 384
# INPUT_WIDTH  = 512
# PREVIEW_WIDTH = 1280
# ASSUME_HFOV_DEG = 63.0  # fallback HFOV if EXIF/VP fail
# ROI_JITTER_FRAC = 0.06  # jitter ROI by ±6% for stability sampling
# TRIM_PERCENT = 10       # trim depth extremes inside ROI (percent each tail)

# # ---------------- GUI sanity ----------------
# def assert_gui_available():
#     try:
#         cv2.namedWindow(".__test__", cv2.WINDOW_NORMAL)
#         cv2.imshow(".__test__", np.zeros((2,2,3), np.uint8))
#         cv2.waitKey(1)
#         cv2.destroyWindow(".__test__")
#     except Exception as e:
#         raise RuntimeError(
#             f"OpenCV GUI not available. Error: {e}\n"
#             f"Check DISPLAY, OpenCV build, and X11 libs."
#         )

# # ---------------- MegaDepth ----------------
# def load_megadepth():
#     print("[INFO] Loading MegaDepth model...")
#     opt = TrainOptions().parse()
#     model = create_model(opt)
#     model.switch_to_eval()
#     return model

# def run_megadepth(model, image_path):
#     img = np.float32(io.imread(image_path)) / 255.0
#     H, W = img.shape[:2]
#     img_resized = resize(img, (INPUT_HEIGHT, INPUT_WIDTH),
#                          order=1, preserve_range=True, anti_aliasing=True)
#     input_img = torch.from_numpy(np.transpose(img_resized, (2,0,1))).float().unsqueeze(0)
#     if torch.cuda.is_available():
#         input_img = input_img.cuda()
#     with torch.no_grad():
#         pred_log = model.netG.forward(Variable(input_img))
#         pred_log = torch.squeeze(pred_log)
#         pred_depth = torch.exp(pred_log).cpu().numpy()
#     depth_resized = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_LINEAR)
#     return depth_resized  # relative depth

# # ---------------- GUI helpers ----------------
# def pick_roi_scaled(image_bgr, window_name="Select ROI", preview_width=PREVIEW_WIDTH):
#     scale = preview_width / image_bgr.shape[1]
#     preview = cv2.resize(image_bgr, None, fx=scale, fy=scale)
#     print(f"[INFO] {window_name}: draw a box, press ENTER/SPACE to confirm.")
#     roi_scaled = cv2.selectROI(window_name, preview, fromCenter=False, showCrosshair=True)
#     cv2.destroyWindow(window_name)
#     x, y, w, h = roi_scaled
#     if w <= 0 or h <= 0:
#         raise RuntimeError("Empty ROI. Please select a non-zero region.")
#     # map back to full-res & clamp
#     x = int(round(x / scale)); y = int(round(y / scale))
#     w = int(round(w / scale)); h = int(round(h / scale))
#     H, W = image_bgr.shape[:2]
#     x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
#     w = max(1, min(W-x, w)); h = max(1, min(H-y, h))
#     return (x, y, w, h)

# def pick_two_points(image_bgr, preview_width=PREVIEW_WIDTH):
#     scale = preview_width / image_bgr.shape[1]
#     preview = cv2.resize(image_bgr, None, fx=scale, fy=scale)
#     pts_scaled = []
#     disp = preview.copy()
#     print("[INFO] Click TWO points to measure distance. Press any key when done.")
#     def cb(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN and len(pts_scaled) < 2:
#             pts_scaled.append((x, y))
#             cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
#             cv2.imshow("Pick 2 Points", disp)
#     cv2.namedWindow("Pick 2 Points", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Pick 2 Points", preview_width, int(preview.shape[0]))
#     cv2.imshow("Pick 2 Points", disp)
#     cv2.setMouseCallback("Pick 2 Points", cb)
#     cv2.waitKey(0)
#     cv2.destroyWindow("Pick 2 Points")
#     if len(pts_scaled) != 2:
#         raise RuntimeError(f"Expected 2 points, got {len(pts_scaled)}")
#     pts_full = [(int(x / scale), int(y / scale)) for (x, y) in pts_scaled]
#     return pts_full

# # ---------------- Pixel-height from ROI (robust, good for curved objects) ----------------
# def object_pixel_height_from_roi(image_bgr, roi):
#     x, y, w, h = roi
#     crop = image_bgr[y:y+h, x:x+w]
#     if crop.size == 0:
#         return h  # fallback
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     sigma = 0.33
#     med = np.median(gray)
#     lower = int(max(0, (1.0 - sigma) * med))
#     upper = int(min(255, (1.0 + sigma) * med))
#     edges = cv2.Canny(gray, lower, upper)
#     edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)
#     cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return h
#     c = max(cnts, key=cv2.contourArea)
#     ys = c[:,:,1].flatten()
#     pix_h = int(ys.max() - ys.min() + 1)
#     return int(np.clip(pix_h, 1, h))

# # ---------------- Robust depth from ROI ----------------
# def robust_depth_from_roi(rel_depth, roi, trim_percent=TRIM_PERCENT, jitter_frac=ROI_JITTER_FRAC):
#     H, W = rel_depth.shape
#     x, y, w, h = roi
#     def region_vals(rx, ry, rw, rh):
#         rx = max(0, min(W-1, rx)); ry = max(0, min(H-1, ry))
#         rw = max(1, min(W-rx, rw)); rh = max(1, min(H-ry, rh))
#         r = rel_depth[ry:ry+rh, rx:rx+rw]
#         r = r[np.isfinite(r)]
#         return r
#     samples = []
#     vals = region_vals(x, y, w, h)
#     if vals.size:
#         vals = np.sort(vals); k = int(len(vals) * trim_percent / 100.0)
#         vals = vals[k:len(vals)-k] if len(vals) > 2*k else vals
#         if vals.size: samples.append(np.median(vals))
#     shifts = [-jitter_frac, 0.0, jitter_frac]
#     for dx in shifts:
#         for dy in shifts:
#             if dx == 0.0 and dy == 0.0: continue
#             rx = int(round(x + dx * w)); ry = int(round(y + dy * h))
#             vals = region_vals(rx, ry, w, h)
#             if vals.size:
#                 vals = np.sort(vals); k = int(len(vals) * trim_percent / 100.0)
#                 vals = vals[k:len(vals)-k] if len(vals) > 2*k else vals
#                 if vals.size: samples.append(np.median(vals))
#     if not samples:
#         raise RuntimeError("Selected ROI has no valid depth.")
#     z_arr = np.array(samples, dtype=np.float64)
#     return float(np.median(z_arr)), float(np.std(z_arr)), len(z_arr)

# # ---------------- Focal estimation helpers ----------------
# def exif_fx_from_meta(image_path, image_width_px, sensor_width_mm=None):
#     try:
#         from PIL import Image, ExifTags
#     except Exception:
#         return None
#     try:
#         img = Image.open(image_path)
#         raw = img._getexif() or {}
#         exif = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
#     except Exception:
#         return None
#     f_mm = None
#     if 'FocalLength' in exif:
#         val = exif['FocalLength']
#         if isinstance(val, tuple):
#             f_mm = float(val[0]) / float(val[1] if val[1] else 1)
#         else:
#             f_mm = float(val)
#     if 'FocalLengthIn35mmFilm' in exif:
#         f35 = float(exif['FocalLengthIn35mmFilm'])
#         return (image_width_px / 36.0) * f35
#     if sensor_width_mm and f_mm:
#         return f_mm * (image_width_px / sensor_width_mm)
#     return None

# def estimate_fx_from_vanishing_points(image_bgr):
#     H, W = image_bgr.shape[:2]; cx, cy = W/2.0, H/2.0
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     lsd = cv2.createLineSegmentDetector()
#     lines = lsd.detect(gray)[0]
#     if lines is None or len(lines) < 50:
#         raise RuntimeError("Not enough line segments for VP.")
#     L = []
#     for x1,y1,x2,y2 in lines[:,0,:]:
#         p1 = np.array([x1,y1,1.0]); p2 = np.array([x2,y2,1.0])
#         l = np.cross(p1, p2); n = np.linalg.norm(l[:2])
#         if n < 1e-6: continue
#         L.append(l / n)
#     L = np.array(L)
#     if len(L) < 30:
#         raise RuntimeError("Too few normalized lines.")
#     rng = np.random.default_rng(0); candidates = []
#     for _ in range(3000):
#         i, j = rng.integers(0, len(L), size=2)
#         if i == j: continue
#         v = np.cross(L[i], L[j])
#         if abs(v[2]) < 1e-9: continue
#         vx, vy = v[0]/v[2], v[1]/v[2]
#         if -10*W < vx < 11*W and -10*H < vy < 11*H:
#             candidates.append((vx, vy))
#     candidates = np.array(candidates, dtype=np.float32)
#     if len(candidates) < 100:
#         raise RuntimeError("Not enough VP candidates.")
#     Kclust = 2
#     criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-3)
#     _, labels, centers = cv2.kmeans(candidates, Kclust, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
#     v1, v2 = centers
#     d = -((v1[0]-cx)*(v2[0]-cx) + (v1[1]-cy)*(v2[1]-cy))
#     if d <= 0:
#         raise RuntimeError("VP orthogonality produced non-positive f^2.")
#     return float(np.sqrt(d))

# def fx_from_hfov(width_px, hfov_deg):
#     return width_px / (2.0 * math.tan(math.radians(hfov_deg/2.0)))

# def estimate_focal_px(image_bgr, image_path):
#     H, W = image_bgr.shape[:2]
#     fx = exif_fx_from_meta(image_path, W, sensor_width_mm=None)
#     if fx:
#         print(f"[INFO] Using EXIF-based fx ≈ {fx:.1f} px")
#         return fx
#     try:
#         fx_vp = estimate_fx_from_vanishing_points(image_bgr)
#         print(f"[INFO] Using VP-based fx ≈ {fx_vp:.1f} px")
#         return fx_vp
#     except Exception as e:
#         print(f"[WARN] VP-based focal failed: {e}")
#     fx_guess = fx_from_hfov(W, hfov_deg=ASSUME_HFOV_DEG)
#     print(f"[WARN] Falling back to HFOV guess fx ≈ {fx_guess:.1f} px ({ASSUME_HFOV_DEG}°)")
#     return fx_guess

# # ---------------- Intrinsics & backprojection ----------------
# def build_K(image_shape, fx, fy=None, cx=None, cy=None):
#     H, W = image_shape[:2]
#     if fy is None: fy = fx
#     if cx is None: cx = W/2.0
#     if cy is None: cy = H/2.0
#     return np.array([[fx, 0, cx],[0, fy, cy],[0,  0,  1]], dtype=np.float32)

# def backproject_uv_to_xyz(points_uv, depth_m, K):
#     fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
#     xyz = []
#     for (u, v) in points_uv:
#         u_i, v_i = int(round(u)), int(round(v))
#         if not (0 <= u_i < depth_m.shape[1] and 0 <= v_i < depth_m.shape[0]):
#             raise RuntimeError(f"Point {(u, v)} outside image.")
#         Z = float(depth_m[v_i, u_i])
#         if not np.isfinite(Z) or Z <= 0:
#             raise RuntimeError(f"Invalid depth at {(u_i, v_i)}: {Z}")
#         X = (u - cx) * Z / fx; Y = (v - cy) * Z / fy
#         xyz.append((X, Y, Z))
#     return np.array(xyz, dtype=np.float32)

# # ---------------- Robust aggregation ----------------
# def mad(arr):
#     med = np.median(arr)
#     return med, np.median(np.abs(arr - med))

# def local_median_depth(depth, u, v, win=7):
#     h, w = depth.shape
#     x0 = max(0, int(u) - win//2)
#     y0 = max(0, int(v) - win//2)
#     x1 = min(w, int(u) + win//2 + 1)
#     y1 = min(h, int(v) + win//2 + 1)
#     patch = depth[y0:y1, x0:x1]
#     vals = patch[np.isfinite(patch) & (patch > 0)]
#     if vals.size == 0:
#         return float('nan')
#     return float(np.median(vals))

# def measure_two_points_robust(image_bgr, metric_depth, K, preview_width=1280, win=9, reps=3, jitter_px=2):
#     pts_uv = pick_two_points(image_bgr, preview_width)
#     fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]

#     def robust_xyz(u, v):
#         samples = []
#         step = max(1, jitter_px // max(1, reps-1))
#         for du in range(-jitter_px, jitter_px+1, step):
#             for dv in range(-jitter_px, jitter_px+1, step):
#                 Z = local_median_depth(metric_depth, u+du, v+dv, win=win)
#                 if np.isfinite(Z) and Z > 0:
#                     X = ((u+du) - cx) * Z / fx
#                     Y = ((v+dv) - cy) * Z / fy
#                     samples.append((X, Y, Z))
#         if not samples:
#             raise RuntimeError(f"No valid depth around ({u},{v})")
#         P = np.array(samples, dtype=np.float32)
#         return np.median(P, axis=0)

#     P1 = robust_xyz(*pts_uv[0])
#     P2 = robust_xyz(*pts_uv[1])
#     dist = float(np.linalg.norm(P1 - P2))
#     print(f"[RESULT] Robust 3D distance: {dist:.3f} meters")
#     print(f"[DEBUG] P1 (u,v)->(X,Y,Z): {pts_uv[0]} -> {P1}")
#     print(f"[DEBUG] P2 (u,v)->(X,Y,Z): {pts_uv[1]} -> {P2}")
#     return dist, pts_uv, P1, P2

# # ---------------- Main flow ----------------
# def main():
#     assert_gui_available()

#     image_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
#     if image_bgr is None:
#         print(f"[ERROR] Image not found: {IMAGE_PATH}")
#         return

#     model = load_megadepth()
#     print("[INFO] Predicting relative depth...")
#     rel_depth = run_megadepth(model, IMAGE_PATH)

#     # --- Choose how many reference objects ---
#     try:
#         N = int(input("[INPUT] How many reference objects (>=2)? ").strip())
#     except Exception:
#         N = 2
#     N = max(2, N)

#     gammas = []
#     per_obj_info = []

#     for i in range(N):
#         roi = pick_roi_scaled(image_bgr, f"Select Reference Object {i+1}")
#         hpx = object_pixel_height_from_roi(image_bgr, roi)
#         z_med, z_std, n_samp = robust_depth_from_roi(rel_depth, roi)
#         while True:
#             try:
#                 Hi = float(input(f"[INPUT] Enter real HEIGHT (meters) for object {i+1}: ").strip())
#                 if Hi > 0: break
#             except Exception:
#                 pass
#             print("  Please enter a positive number (meters).")
#         gamma_i = (hpx * z_med) / max(1e-9, Hi)
#         gammas.append(gamma_i)
#         per_obj_info.append({
#             'roi': roi, 'hpx': hpx, 'z_med': z_med, 'z_std': z_std, 'n_samp': n_samp, 'H': Hi, 'gamma': gamma_i
#         })
#         print(f"[INFO] Obj{i+1}: h_px={hpx}, z_rel_med={z_med:.6f} (±{z_std:.6f}, n={n_samp}), H={Hi} m -> gamma={gamma_i:.6f}")

#     gammas = np.array(gammas, dtype=np.float64)
#     g_med = np.median(gammas)
#     g_mad = np.median(np.abs(gammas - g_med))
#     if g_mad < 1e-9:
#         mask = np.ones_like(gammas, dtype=bool)
#     else:
#         mask = np.abs(gammas - g_med) <= (2.5 * g_mad)
#     kept = gammas[mask]
#     print(f"[INFO] gamma raw: {gammas}")
#     print(f"[INFO] gamma median={g_med:.6f}, MAD={g_mad:.6f}, kept={len(kept)}/{len(gammas)}")
#     if len(kept) < 2:
#         print("[WARN] Fewer than 2 inliers after filtering; using all gammas.")
#         kept = gammas
#     gamma = float(np.median(kept))
#     print(f"[INFO] gamma (robust) = {gamma:.6f}  (gamma = f_px / k)")

#     # --- Estimate focal length in pixels ---
#     fpx = estimate_focal_px(image_bgr, IMAGE_PATH)
#     print(f"[INFO] Estimated focal length fx ≈ {fpx:.2f} px")

#     # --- Recover scale and metric depth ---
#     k = fpx / gamma
#     metric_depth = rel_depth * k
#     print(f"[INFO] Global scale k = f/gamma ≈ {k:.6f} m per relative-depth unit")

#     # --- Build K and measure (robust) ---
#     K = build_K(image_bgr.shape, fx=fpx)
#     measure_two_points_robust(image_bgr, metric_depth, K, preview_width=PREVIEW_WIDTH, win=9, reps=3, jitter_px=2)

#     # --- Optional: also simple measurement with single-pixel depth ---
#     pts_uv = pick_two_points(image_bgr, PREVIEW_WIDTH)
#     pts_xyz = backproject_uv_to_xyz(pts_uv, metric_depth, K)
#     dist_m = float(np.linalg.norm(pts_xyz[0] - pts_xyz[1]))
#     print(f"[RESULT] Simple 3D distance (no robust window): {dist_m:.3f} meters")
#     print(f"[DEBUG] P1 (u,v)->(X,Y,Z): {pts_uv[0]} -> {pts_xyz[0]}")
#     print(f"[DEBUG] P2 (u,v)->(X,Y,Z): {pts_uv[1]} -> {pts_xyz[1]}")
#     print(f"[DEBUG] fx(px)={fpx:.2f}, gamma={gamma:.6f}, k={k:.6f}")

#     # --- Report: reprojection sanity on references ---
#     print("\n[REPORT] Reference object reprojection check:")
#     for i, info in enumerate(per_obj_info, 1):
#         x,y,w,h = info['roi']
#         H_true = info['H']
#         z_rel = info['z_med']
#         Z_metric = z_rel * k
#         h_pred = (fpx * H_true) / max(1e-9, Z_metric)  # predicted pixel height by pinhole
#         err_px = h_pred - info['hpx']
#         err_pct = 100.0 * err_px / max(1e-9, info['hpx'])
#         print(f"  Obj{i}: h_px_meas={info['hpx']}, h_px_pred={h_pred:.2f}, "
#               f"err={err_px:.2f}px ({err_pct:+.1f}%), z_rel±std={z_rel:.4f}±{info['z_std']:.4f}")

# if __name__ == "__main__":
#     main()
import os
import sys
import cv2
import math
import numpy as np
import torch
from torch.autograd import Variable
from skimage import io
from skimage.transform import resize

# Add MegaDepth repo to path
sys.path.append('./megadepth')
from options.train_options import TrainOptions
from models.models import create_model
from point_cloud import export_pointcloud
from axis_symettric_recon import axis_symmetric_mesh, save_mesh_ply
from car_symmetry_pipeline import export_car_symmetric_cloud

# === CONFIG ===
IMAGE_PATH = 'image.jpg'
INPUT_HEIGHT = 384
INPUT_WIDTH  = 512
PREVIEW_WIDTH = 1280
ASSUME_HFOV_DEG = 63.0      # fallback HFOV if EXIF/VP fail
TRIM_PERCENT = 10           # trimming for robust medians (% per tail)
SENSOR_WIDTH_MM = None      # set e.g. 36.0 or 23.5 to print f(mm); leave None to skip


# Put near the top of your script
PREVIEW_MAX_FRAC = 0.85  # use up to 85% of screen width/height

def get_screen_size():
    """Return (screen_w, screen_h). Works on X11/NoMachine via Tkinter. Falls back if not available."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(sw), int(sh)
    except Exception:
        # Safe fallback if Tk is not available
        return 1920, 1080

def make_preview(img, max_frac=PREVIEW_MAX_FRAC):
    """
    Returns (preview_img, scale) where preview_img is resized to fit <= max_frac of screen.
    scale = preview_w / full_w. Use 1/scale to map preview coords -> full-res coords.
    """
    h, w = img.shape[:2]
    sw, sh = get_screen_size()
    max_w = int(sw * max_frac)
    max_h = int(sh * max_frac)
    scale = min(max_w / w, max_h / h, 1.0)  # never upscale
    if scale < 1.0:
        preview = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        preview = img.copy()
    return preview, float(scale)

def show_window(title, img):
    """Create a resizable window sized exactly to the image given."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)  # ensure not fullscreen
    cv2.resizeWindow(title, img.shape[1], img.shape[0])
    cv2.imshow(title, img)

# ---------------- GUI sanity ----------------
def assert_gui_available():
    try:
        cv2.namedWindow(".__test__", cv2.WINDOW_NORMAL)
        cv2.imshow(".__test__", np.zeros((2,2,3), np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow(".__test__")
    except Exception as e:
        raise RuntimeError(
            f"OpenCV GUI not available. Error: {e}\n"
            f"Check DISPLAY, OpenCV build, and X11 libs."
        )

# ---------------- MegaDepth ----------------
def load_megadepth():
    print("[INFO] Loading MegaDepth model...")
    opt = TrainOptions().parse()
    model = create_model(opt)
    model.switch_to_eval()
    return model

def run_megadepth(model, image_path):
    img = np.float32(io.imread(image_path)) / 255.0
    H, W = img.shape[:2]
    img_resized = resize(img, (INPUT_HEIGHT, INPUT_WIDTH),
                         order=1, preserve_range=True, anti_aliasing=True)
    input_img = torch.from_numpy(np.transpose(img_resized, (2,0,1))).float().unsqueeze(0)
    if torch.cuda.is_available():
        input_img = input_img.cuda()
    with torch.no_grad():
        pred_log = model.netG.forward(Variable(input_img))
        pred_log = torch.squeeze(pred_log)
        pred_depth = torch.exp(pred_log).cpu().numpy()
    depth_resized = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_LINEAR)
    return depth_resized  # relative depth (unitless scale)

# ---------------- GUI helpers ----------------
def pick_two_points(image_bgr, preview_width=PREVIEW_WIDTH, window_name="Pick 2 Points"):
    scale = preview_width / image_bgr.shape[1]
    preview = cv2.resize(image_bgr, None, fx=scale, fy=scale)
    pts_scaled = []
    disp = preview.copy()
    print(f"[INFO] {window_name}: Click TWO points. Press any key when done.")
    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts_scaled) < 2:
            pts_scaled.append((x, y))
            cv2.circle(disp, (x, y), 5, (0,255,0), -1)
            cv2.imshow(window_name, disp)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, preview_width, int(preview.shape[0]))
    cv2.imshow(window_name, disp)
    cv2.setMouseCallback(window_name, cb)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    if len(pts_scaled) != 2:
        raise RuntimeError("Expected TWO points.")
    (u1s, v1s), (u2s, v2s) = pts_scaled
    u1, v1 = int(u1s/scale), int(v1s/scale)
    u2, v2 = int(u2s/scale), int(v2s/scale)
    return (u1, v1), (u2, v2)

def pick_segment(image_bgr, preview_width=PREVIEW_WIDTH, window_name="Click segment endpoints"):
    (u1, v1), (u2, v2) = pick_two_points(image_bgr, preview_width, window_name)
    l_px = float(np.hypot(u2 - u1, v2 - v1))
    return (u1, v1), (u2, v2), l_px

def get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(sw), int(sh)
    except Exception:
        return 1920, 1080

def make_preview(img, max_frac=0.85):
    h, w = img.shape[:2]
    sw, sh = get_screen_size()
    max_w = int(sw * max_frac)
    max_h = int(sh * max_frac)
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        preview = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        preview = img.copy()
    return preview, float(scale)

def show_window(title, img):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, img.shape[1], img.shape[0])
    cv2.imshow(title, img)

def pick_roi_scaled(image_bgr, window_name="Select ROI", preview_width=None):
    # preview_width is optional now; we size by screen instead
    preview, scale = make_preview(image_bgr, max_frac=0.85)
    print(f"[INFO] {window_name}: draw a box, ENTER/SPACE to confirm.")
    show_window(window_name, preview)
    roi_scaled = cv2.selectROI(window_name, preview, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    x, y, w, h = map(int, roi_scaled)
    if w <= 0 or h <= 0:
        raise RuntimeError("Empty ROI.")
    inv = 1.0 / max(scale, 1e-9)
    x = int(round(x * inv)); y = int(round(y * inv))
    w = int(round(w * inv)); h = int(round(h * inv))
    H, W = image_bgr.shape[:2]
    x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
    w = max(1, min(W - x, w)); h = max(1, min(H - y, h))
    return (x, y, w, h)


def sanity_fx(fx, W, fallback_hfov_deg=63.0):
    # Typical fx is ~[0.6*W, 3.0*W] pixels for most photos.
    if not np.isfinite(fx) or fx < 0.6*W or fx > 3.0*W:
        fx_fb = fx_from_hfov(W, fallback_hfov_deg)
        print(f"[WARN] VP/EXIF fx={fx:.1f}px out of sane range for width={W}. "
              f"Falling back to HFOV {fallback_hfov_deg}° -> fx={fx_fb:.1f}px")
        return fx_fb
    return fx

# ---------------- Depth sampling ----------------
def local_median_depth(depth, u, v, win=7):
    h, w = depth.shape
    x0 = max(0, int(u) - win//2)
    y0 = max(0, int(v) - win//2)
    x1 = min(w, int(u) + win//2 + 1)
    y1 = min(h, int(v) + win//2 + 1)
    patch = depth[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch) & (patch > 0)]
    if vals.size == 0:
        return float('nan')
    return float(np.median(vals))

def sample_relative_depth_along_segment(rel_depth, p1, p2, stripe_half=2, samples=50, trim_percent=TRIM_PERCENT):
    """
    Sample rel depth in a thin stripe perpendicular to the segment p1->p2.
    Robust trimmed median returned.
    """
    h, w = rel_depth.shape
    (u1, v1), (u2, v2) = p1, p2
    du, dv = (u2 - u1), (v2 - v1)
    seg_len = math.hypot(du, dv)
    if seg_len < 1e-6:
        raise RuntimeError("Zero-length segment.")
    # perpendicular unit normal
    nx, ny = -dv/seg_len, du/seg_len

    vals = []
    for t in np.linspace(0.0, 1.0, samples):
        uc = u1 + t * du
        vc = v1 + t * dv
        # stripe across normal
        for s in range(-stripe_half, stripe_half+1):
            uu = int(round(uc + s * nx))
            vv = int(round(vc + s * ny))
            if 0 <= uu < w and 0 <= vv < h:
                z = rel_depth[vv, uu]
                if np.isfinite(z) and z > 0:
                    vals.append(z)
    if not vals:
        raise RuntimeError("No valid relative depth along segment.")
    vals = np.sort(np.asarray(vals, dtype=np.float32))
    k = int(len(vals) * trim_percent / 100.0)
    if len(vals) > 2*k:
        vals = vals[k:len(vals)-k]
    return float(np.median(vals))

def gamma_from_segment(rel_depth, p1, p2, known_metric_length, stripe_half=2):
    """
    gamma = (l_px * z_rel_med) / H  (H is known metric length of the clicked segment).
    No assumption about same depth at endpoints; uses robust median depth along segment.
    """
    l_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    Hm = float(known_metric_length)
    if Hm <= 0:
        raise ValueError("Known length must be > 0.")
    z_med = sample_relative_depth_along_segment(rel_depth, p1, p2, stripe_half=stripe_half)
    gamma = (l_px * z_med) / Hm
    return gamma, l_px, z_med

# ---------------- Focal estimation ----------------
def exif_fx_from_meta(image_path, image_width_px, sensor_width_mm=None):
    try:
        from PIL import Image, ExifTags
    except Exception:
        return None
    try:
        img = Image.open(image_path)
        raw = img._getexif() or {}
        exif = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        return None
    f_mm = None
    if 'FocalLength' in exif:
        val = exif['FocalLength']
        if isinstance(val, tuple):
            f_mm = float(val[0]) / float(val[1] if val[1] else 1)
        else:
            f_mm = float(val)
    if 'FocalLengthIn35mmFilm' in exif:
        f35 = float(exif['FocalLengthIn35mmFilm'])
        return (image_width_px / 36.0) * f35
    if sensor_width_mm and f_mm:
        return f_mm * (image_width_px / sensor_width_mm)
    return None

def estimate_fx_from_vanishing_points(image_bgr):
    H, W = image_bgr.shape[:2]; cx, cy = W/2.0, H/2.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)[0]
    if lines is None or len(lines) < 50:
        raise RuntimeError("Not enough line segments for VP.")
    L = []
    for x1,y1,x2,y2 in lines[:,0,:]:
        p1 = np.array([x1,y1,1.0]); p2 = np.array([x2,y2,1.0])
        l = np.cross(p1, p2); n = np.linalg.norm(l[:2])
        if n < 1e-6: continue
        L.append(l / n)
    L = np.array(L)
    if len(L) < 30:
        raise RuntimeError("Too few normalized lines.")
    rng = np.random.default_rng(0); candidates = []
    for _ in range(3000):
        i, j = rng.integers(0, len(L), size=2)
        if i == j: continue
        v = np.cross(L[i], L[j])
        if abs(v[2]) < 1e-9: continue
        vx, vy = v[0]/v[2], v[1]/v[2]
        if -10*W < vx < 11*W and -10*H < vy < 11*H:
            candidates.append((vx, vy))
    candidates = np.array(candidates, dtype=np.float32)
    if len(candidates) < 100:
        raise RuntimeError("Not enough VP candidates.")
    Kclust = 2
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-3)
    _, labels, centers = cv2.kmeans(candidates, Kclust, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    v1, v2 = centers
    d = -((v1[0]-cx)*(v2[0]-cx) + (v1[1]-cy)*(v2[1]-cy))
    if d <= 0:
        raise RuntimeError("VP orthogonality produced non-positive f^2.")
    return float(np.sqrt(d))

def fx_from_hfov(width_px, hfov_deg):
    return width_px / (2.0 * math.tan(math.radians(hfov_deg/2.0)))

def estimate_focal_px(image_bgr, image_path):
    H, W = image_bgr.shape[:2]
    fx = exif_fx_from_meta(image_path, W, sensor_width_mm=None)
    if fx:
        print(f"[INFO] Using EXIF-based fx ≈ {fx:.1f} px")
        return fx
    try:
        fx_vp = estimate_fx_from_vanishing_points(image_bgr)
        print(f"[INFO] Using VP-based fx ≈ {fx_vp:.1f} px")
        return fx_vp
    except Exception as e:
        print(f"[WARN] VP-based focal failed: {e}")
    fx_guess = fx_from_hfov(W, hfov_deg=ASSUME_HFOV_DEG)
    print(f"[WARN] Falling back to HFOV guess fx ≈ {fx_guess:.1f} px ({ASSUME_HFOV_DEG}°)")
    return fx_guess

# ---------------- Intrinsics & backprojection ----------------
def build_K(image_shape, fx, fy=None, cx=None, cy=None):
    H, W = image_shape[:2]
    if fy is None: fy = fx
    if cx is None: cx = W/2.0
    if cy is None: cy = H/2.0
    return np.array([[fx, 0, cx],[0, fy, cy],[0,  0,  1]], dtype=np.float32)

def deproject_uv_with_Z(u, v, Z, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)

def backproject_uv_to_xyz(points_uv, depth_m, K):
    fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
    xyz = []
    for (u, v) in points_uv:
        u_i, v_i = int(round(u)), int(round(v))
        if not (0 <= u_i < depth_m.shape[1] and 0 <= v_i < depth_m.shape[0]):
            raise RuntimeError(f"Point {(u, v)} outside image.")
        Z = float(depth_m[v_i, u_i])
        if not np.isfinite(Z) or Z <= 0:
            raise RuntimeError(f"Invalid depth at {(u_i, v_i)}: {Z}")
        X = (u - cx) * Z / fx; Y = (v - cy) * Z / fy
        xyz.append((X, Y, Z))
    return np.array(xyz, dtype=np.float32)

# ---------------- Plane fit & projected measurement ----------------
def fit_plane_svd(points_xyz):
    C = points_xyz.mean(axis=0)
    Q = points_xyz - C
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    return C, n

def project_point_to_plane(P, C, n):
    return P - n * np.dot(n, (P - C))

def measure_two_points_on_plane(image_bgr, metric_depth, K,
                                preview_width=PREVIEW_WIDTH, grid=10, win=7):
    """
    1) Select a planar ROI (e.g., windshield).
    2) Sample 3D points in ROI; fit plane.
    3) Click 2 points; project them onto plane; measure distance on the plane.
    """
    roi = pick_roi_scaled(image_bgr, "Select planar surface ROI", preview_width)
    x, y, w, h = roi
    xs = np.linspace(x, x + w - 1, grid).astype(int)
    ys = np.linspace(y, y + h - 1, grid).astype(int)
    pts3d = []
    for vv in ys:
        for uu in xs:
            Z = local_median_depth(metric_depth, uu, vv, win=win)
            if np.isfinite(Z) and Z > 0:
                pts3d.append(deproject_uv_with_Z(uu, vv, Z, K))
    pts3d = np.array(pts3d, dtype=np.float32)
    if pts3d.shape[0] < 6:
        raise RuntimeError("Not enough valid 3D points in ROI to fit a plane.")
    C, n = fit_plane_svd(pts3d)
    (u1, v1), (u2, v2) = pick_two_points(image_bgr, preview_width, "Pick 2 Points (plane-projected)")
    Z1 = local_median_depth(metric_depth, u1, v1, win=win)
    Z2 = local_median_depth(metric_depth, u2, v2, win=win)
    if not (np.isfinite(Z1) and Z1 > 0 and np.isfinite(Z2) and Z2 > 0):
        raise RuntimeError("Invalid depth at one of the clicked points.")
    P1 = deproject_uv_with_Z(u1, v1, Z1, K)
    P2 = deproject_uv_with_Z(u2, v2, Z2, K)
    P1p = project_point_to_plane(P1, C, n)
    P2p = project_point_to_plane(P2, C, n)
    d = float(np.linalg.norm(P1p - P2p))
    print(f"[RESULT] Plane-projected distance: {d:.3f} meters")
    print(f"[DEBUG] Plane C={C}, n={n}")
    print(f"[DEBUG] P1->P1p: {P1} -> {P1p}")
    print(f"[DEBUG] P2->P2p: {P2} -> {P2p}")
    return d

# ---------------- Same-depth chord (optional quick measure) ----------------
def measure_two_points_same_depth(image_bgr, metric_depth, K, preview_width=PREVIEW_WIDTH, win=9):
    (u1, v1), (u2, v2) = pick_two_points(image_bgr, preview_width, "Pick 2 Points (same-depth)")
    Z1 = local_median_depth(metric_depth, u1, v1, win=win)
    Z2 = local_median_depth(metric_depth, u2, v2, win=win)
    Zs = [z for z in (Z1, Z2) if np.isfinite(z) and z > 0]
    if not Zs:
        raise RuntimeError("No valid depth at clicked points.")
    Z_shared = float(np.median(Zs))
    P1 = deproject_uv_with_Z(u1, v1, Z_shared, K)
    P2 = deproject_uv_with_Z(u2, v2, Z_shared, K)
    d = float(np.linalg.norm(P1 - P2))
    print(f"[RESULT] Same-depth distance: {d:.3f} meters (Z_shared={Z_shared:.3f})")
    return d

# ---------------- Robust aggregation ----------------
def mad(arr):
    med = np.median(arr)
    return med, np.median(np.abs(arr - med))

# ---------------- Main flow ----------------
def main():
    assert_gui_available()

    image_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"[ERROR] Image not found: {IMAGE_PATH}")
        return

    model = load_megadepth()
    print("[INFO] Predicting relative depth...")
    rel_depth = run_megadepth(model, IMAGE_PATH)

    # --- Collect N reference segments (known metric lengths) ---
    try:
        N = int(input("[INPUT] How many reference segments (>=2)? ").strip())
    except Exception:
        N = 2
    N = max(2, N)

    gammas = []
    for i in range(N):
        p1, p2, l_px = pick_segment(image_bgr, PREVIEW_WIDTH, window_name=f"Ref segment {i+1}")
        while True:
            try:
                H_i = float(input(f"[INPUT] Enter real LENGTH (meters) for segment {i+1}: ").strip())
                if H_i > 0: break
            except Exception:
                pass
            print("  Please enter a positive number (meters).")
        gamma_i, l_px_i, z_med_i = gamma_from_segment(rel_depth, p1, p2, H_i, stripe_half=2)
        gammas.append(gamma_i)
        print(f"[INFO] Ref{i+1}: l_px={l_px_i:.1f}, z_rel_med={z_med_i:.6f}, H={H_i} m -> gamma={gamma_i:.6f}")

    gammas = np.asarray(gammas, dtype=np.float64)
    g_med, g_mad = mad(gammas)
    mask = np.abs(gammas - g_med) <= (2.5 * g_mad) if g_mad > 1e-9 else np.ones_like(gammas, dtype=bool)
    kept = gammas[mask]
    if len(kept) < 2:
        print("[WARN] Too few inliers after filtering; using all.")
        kept = gammas
    gamma = float(np.median(kept))
    print(f"[INFO] gamma raw: {gammas}")
    print(f"[INFO] gamma median={g_med:.6f}, MAD={g_mad:.6f}, kept={len(kept)}/{len(gammas)}")
    print(f"[INFO] gamma (robust) = {gamma:.6f}  (gamma = f_px / k)")

    # --- Estimate focal length in pixels ---
    fpx = estimate_focal_px(image_bgr, IMAGE_PATH)
    Himg, Wimg = image_bgr.shape[:2]
    fpx = sanity_fx(fpx, Wimg, fallback_hfov_deg=ASSUME_HFOV_DEG)
    print(f"[INFO] Estimated focal length fx ≈ {fpx:.2f} px")

    # Optional: report f in mm if sensor width known
    H, W = image_bgr.shape[:2]
    if SENSOR_WIDTH_MM:
        f_mm = fpx * (SENSOR_WIDTH_MM / W)
        print(f"[INFO] Focal length ≈ {f_mm:.2f} mm (sensor width {SENSOR_WIDTH_MM} mm, image width {W} px)")

    # --- Recover global scale and metric depth ---
    k = fpx / gamma
    metric_depth = rel_depth * k
    print(f"[INFO] Global scale k = f/gamma ≈ {k:.6f} m per relative-depth unit")

    # --- Build K ---
    K = build_K(image_bgr.shape, fx=fpx)

    D = _preprocess_depth(metric_depth, z_clip=(0.2, 80.0), median_ksize=3)
    pts, cols, used_stride = _backproject_dense_auto_stride(D, K, rgb_bgr=image_bgr, mask=None, max_points=800_000)
    print(f"[DEBUG] HxW={D.shape[::-1]}  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}  points={len(pts)}")
    print(f"[DEBUG] depth median (maskless) = {np.nanmedian(D):.2f} m")
    _save_ply_binary_little("original_only.ply", pts, cols)

    # tweak these to taste onlpyear
    
    export_car_symmetric_cloud(
    image_bgr=image_bgr,
    metric_depth=metric_depth,
    K=K,
    out_path="car_sym_refined_vox15mm.ply",
    z_clip=(0.2, 80.0),       # adjust to your scene
    median_ksize=3,           # mild denoise
    stride=1,                 # 1=dense; 2/3 to thin pre-voxel
    plane_mode="midline_plus_optical_z",
    refine=True,              # ICP-like plane refinement
    pair_radius=0.25,         # m; tweak 0.15–0.35
    iters=5,
    use_grabcut=True,         # box-select car to mask background
    voxel=0.015,              # 1.5cm voxels → stable & compact
    combine=True              # original + mirrored
)

    # --- Measure on plane (recommended for windshields/panels) ---
    try:
        measure_two_points_on_plane(image_bgr, metric_depth, K, preview_width=PREVIEW_WIDTH, grid=10, win=7)
    except Exception as e:
        print(f"[WARN] Plane-projected measurement failed: {e}")

    # --- Optional: same-depth chord quick measurement ---
    try:
        measure_two_points_same_depth(image_bgr, metric_depth, K, preview_width=PREVIEW_WIDTH, win=9)
    except Exception as e:
        print(f"[WARN] Same-depth measurement failed: {e}")

    # --- Optional: naive 3D (will inflate if ΔZ != 0) ---
    try:
        (u1, v1), (u2, v2) = pick_two_points(image_bgr, PREVIEW_WIDTH, "Pick 2 Points (naive 3D)")
        P = backproject_uv_to_xyz([(u1, v1), (u2, v2)], metric_depth, K)
        d = float(np.linalg.norm(P[0] - P[1]))
        print(f"[RESULT] Naive 3D distance (includes ΔZ): {d:.3f} meters")
        print(f"[DEBUG] P1: {P[0]}  P2: {P[1]}")
    except Exception as e:
        print(f"[WARN] Naive 3D measurement failed: {e}")

if __name__ == "__main__":
    main()
