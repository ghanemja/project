# car_symmetry_pointcloud.py
import numpy as np
import cv2
import time

# ---------- screen-aware GUI helpers ----------
def _get_screen_size():
    try:
        import tkinter as tk
        r = tk.Tk(); r.withdraw()
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy(); return int(sw), int(sh)
    except Exception:
        return 1920, 1080

def _make_preview(img, max_frac=0.85):
    h, w = img.shape[:2]; sw, sh = _get_screen_size()
    max_w, max_h = int(sw*max_frac), int(sh*max_frac)
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img, float(s)

def _show_window(title, img):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, img.shape[1], img.shape[0])
    cv2.imshow(title, img)

def _pick_two_points(image_bgr, title="Car midline (2 pts)", mask=None,
                     timeout_sec=45, overlay=True):
    """
    Non-blocking GUI point picker with timeout + auto-fallback to midline from mask.
    Returns two points in FULL-RES pixel coords.
    Keys: ENTER/SPACE to finish (once 2 clicks), 'u' undo, 'c' clear, ESC to cancel.
    """
    prev, s = _make_preview(image_bgr, max_frac=0.85)
    Hf, Wf = image_bgr.shape[:2]
    pts = []
    disp = prev.copy()

    def draw():
        disp[:] = prev
        if overlay:
            txt = f"Clicks: {len(pts)}/2   (ENTER/SPACE to accept, 'u' undo, 'c' clear)"
            cv2.putText(disp, txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20,255,20), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(disp, p, 5, (0,255,0), -1)
        if len(pts) == 2:
            cv2.line(disp, pts[0], pts[1], (0,255,0), 2)
        cv2.imshow(title, disp)

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y)); draw()

    _show_window(title, disp)
    cv2.moveWindow(title, 60, 60)  # ensure it’s on-screen top-left
    cv2.setMouseCallback(title, cb)
    draw()

    t0 = time.time()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32):  # ENTER/SPACE
            if len(pts) == 2:
                break
        elif k in (27,):   # ESC
            break
        elif k == ord('u') and pts:
            pts.pop(); draw()
        elif k == ord('c'):
            pts = []; draw()

        # countdown overlay
        if overlay:
            remaining = max(0, int(round(timeout_sec - (time.time() - t0))))
            disp2 = disp.copy()
            cv2.putText(disp2, f"Timeout in {remaining}s", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2, cv2.LINE_AA)
            cv2.imshow(title, disp2)

        if time.time() - t0 > timeout_sec:
            break

    cv2.destroyWindow(title)

    # If got two clicks → map to full-res and return
    if len(pts) == 2:
        inv = 1.0 / max(s, 1e-9)
        (u1s, v1s), (u2s, v2s) = pts
        u1, v1 = int(round(u1s * inv)), int(round(v1s * inv))
        u2, v2 = int(round(u2s * inv)), int(round(v2s * inv))
        # clamp
        u1 = np.clip(u1, 0, Wf-1); u2 = np.clip(u2, 0, Wf-1)
        v1 = np.clip(v1, 0, Hf-1); v2 = np.clip(v2, 0, Hf-1)
        return (u1, v1), (u2, v2)

    # ---- Auto-fallback: derive midline from mask or center column ----
    print("[WARN] No clicks received; using auto midline fallback.")
    if mask is not None and mask.shape[:2] == (Hf, Wf):
        # choose column with max foreground pixels
        col_sum = mask.sum(axis=0)
        cx = int(np.argmax(col_sum))
        band = mask[:, max(cx-5,0):min(cx+6,Wf)]
        ys = np.where(band.any(axis=1))[0]
        if ys.size >= 2:
            u1, v1 = cx, int(ys[0])
            u2, v2 = cx, int(ys[-1])
            return (u1, v1), (u2, v2)

    # fallback to image center column
    cx = Wf // 2
    u1, v1 = cx, int(0.25 * Hf)
    u2, v2 = cx, int(0.75 * Hf)
    return (u1, v1), (u2, v2)

# ---------- depth utilities ----------
def _local_median_depth(depth, u, v, win=7):
    h,w = depth.shape
    x0=max(0,u-win//2); y0=max(0,v-win//2); x1=min(w,u+win//2+1); y1=min(h,v+win//2+1)
    patch = depth[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch) & (patch>0)]
    return float(np.median(vals)) if vals.size else float('nan')

def _preprocess_depth(depth_m, z_clip=(0.05, None), median_ksize=3):
    D = depth_m.astype(np.float32).copy()
    zmin, zmax = z_clip
    if zmin is not None: D[D<zmin] = np.nan
    if zmax is not None: D[D>zmax] = np.nan
    if median_ksize and median_ksize>=3:
        F = D.copy(); F[~np.isfinite(F)] = 0.0
        F = cv2.medianBlur(F, int(median_ksize))
        D = np.where(np.isfinite(D), F, np.nan)
    return D

# ---------- backprojection ----------
def _backproject_dense(depth_m, K, rgb_bgr=None, mask=None, stride=1):
    H,W = depth_m.shape
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    us = np.arange(0,W,stride,dtype=np.float32)
    vs = np.arange(0,H,stride,dtype=np.float32)
    uu,vv = np.meshgrid(us,vs)
    Z = depth_m[::stride, ::stride]
    valid = np.isfinite(Z) & (Z>0)
    if mask is not None: valid &= (mask[::stride, ::stride] > 0)
    if not np.any(valid): return np.empty((0,3),np.float32), None
    uu,vv,Z = uu[valid], vv[valid], Z[valid]
    X = (uu - cx) * Z / fx; Y = (vv - cy) * Z / fy
    pts = np.vstack((X,Y,Z)).T.astype(np.float32)
    cols = None
    if rgb_bgr is not None and rgb_bgr.size:
        C = rgb_bgr[::stride, ::stride][valid][:, ::-1]  # BGR->RGB
        cols = C.astype(np.uint8)
    return pts, cols

# ---------- quick car mask (GrabCut box) ----------
def _select_roi_scaled(image_bgr, title="Draw a box"):
    """Screen-fit ROI picker. Returns ROI in FULL-RES coords."""
    prev, s = _make_preview(image_bgr, max_frac=0.85)
    print(f"[INFO] {title}: draw a box, then press ENTER/SPACE to confirm (or 'c' to cancel).")
    _show_window(title, prev)
    roi_scaled = cv2.selectROI(title, prev, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    x, y, w, h = map(int, roi_scaled)
    if w <= 0 or h <= 0:
        return None
    inv = 1.0 / max(s, 1e-9)
    x = int(round(x * inv)); y = int(round(y * inv))
    w = int(round(w * inv)); h = int(round(h * inv))
    H, W = image_bgr.shape[:2]
    x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
    w = max(1, min(W-x, w)); h = max(1, min(H-y, h))
    return (x, y, w, h)

def _grabcut_mask(image_bgr,
                  title="Draw a box around the CAR",
                  fast=True,
                  target_w=1024,          # downscale width for fast GrabCut
                  gc_iters=3,             # iters on low-res
                  refine_hr_iters=1):     # quick refine at full-res using INIT_WITH_MASK
    """
    Returns uint8 mask (1=keep/FG, 0=BG). Fast path does GrabCut at low-res, upscales mask,
    and optionally refines once at full-res.
    """
    roi = _select_roi_scaled(image_bgr, title)
    if roi is None:
        print("[WARN] ROI cancelled; using full image.")
        roi = (0, 0, image_bgr.shape[1], image_bgr.shape[0])

    H, W = image_bgr.shape[:2]
    x, y, w, h = roi

    if not fast:
        # Full-res GrabCut (can be slow on large images)
        print("[INFO] Running full-res GrabCut...")
        t0 = time.time()
        mask = np.zeros((H, W), np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        cv2.grabCut(image_bgr, mask, (x,y,w,h), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        keep = (mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD)
        print(f"[INFO] GrabCut done in {time.time()-t0:.2f}s")
        return keep.astype(np.uint8)

    # ---- FAST path: run GrabCut on a downscaled copy ----
    s = min(1.0, target_w / float(W))
    if s < 1.0:
        small = cv2.resize(image_bgr, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
        rx, ry, rw, rh = [int(round(v * s)) for v in (x,y,w,h)]
    else:
        small = image_bgr.copy()
        rx, ry, rw, rh = x, y, w, h

    print(f"[INFO] Running low-res GrabCut at scale {s:.3f} (iters={gc_iters})...")
    t0 = time.time()
    mask_s = np.zeros(small.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(small, mask_s, (rx,ry,rw,rh), bgdModel, fgdModel, gc_iters, cv2.GC_INIT_WITH_RECT)
    keep_s = ((mask_s==cv2.GC_FGD) | (mask_s==cv2.GC_PR_FGD)).astype(np.uint8)

    # Upscale to full-res
    keep = cv2.resize(keep_s, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    print(f"[INFO] Low-res GrabCut done in {time.time()-t0:.2f}s; upscaled to full-res.")
    def _bbox_from_mask(bin_mask):
        ys, xs = np.where(bin_mask > 0)
        if xs.size == 0: return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _expand_bbox(x0, y0, x1, y1, W, H, margin_frac=0.08):
        w = x1 - x0 + 1; h = y1 - y0 + 1
        mx = int(round(w * margin_frac)); my = int(round(h * margin_frac))
        x0 = max(0, x0 - mx); y0 = max(0, y0 - my)
        x1 = min(W-1, x1 + mx); y1 = min(H-1, y1 + my)
        return x0, y0, x1, y1

    def _refine_with_mask_fast(image_bgr, keep_bin,
                            iters=1, margin_frac=0.08, scale=0.50,
                            morph_ksize=5):
        """
        Fast refinement:
        - take bbox of current mask (+margin)
        - downscale crop
        - run GC_INIT_WITH_MASK for a few iters
        - upscale & paste
        - optional morphology to clean edges
        keep_bin: uint8 0/1
        returns refined uint8 0/1 mask
        """
        H, W = keep_bin.shape[:2]
        box = _bbox_from_mask(keep_bin)
        if box is None:
            return keep_bin
        x0, y0, x1, y1 = _expand_bbox(*box, W, H, margin_frac=margin_frac)

        crop_img = image_bgr[y0:y1+1, x0:x1+1]
        crop_mask_bin = keep_bin[y0:y1+1, x0:x1+1]

        # Downscale
        if scale < 1.0:
            new_w = max(64, int(round(crop_img.shape[1] * scale)))
            new_h = max(64, int(round(crop_img.shape[0] * scale)))
            img_s = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_s = cv2.resize(crop_mask_bin, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            img_s = crop_img
            mask_s = crop_mask_bin

        # Build GC labels from binary mask (probable FG/BG)
        gc_mask = np.where(mask_s > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)

        # Run fast GrabCut on the small crop
        if iters and iters > 0:
            bgdModel = np.zeros((1,65), np.float64)
            fgdModel = np.zeros((1,65), np.float64)
            t0 = time.time()
            cv2.grabCut(img_s, gc_mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
            # convert to binary
            keep_s = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8)
            # Upscale back to crop size
            if scale < 1.0:
                keep_crop = cv2.resize(keep_s, (crop_img.shape[1], crop_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                keep_crop = keep_s
            # Paste back
            keep_refined = keep_bin.copy()
            keep_refined[y0:y1+1, x0:x1+1] = keep_crop
            # Optional morphology to clean edges (close -> open)
            if morph_ksize and morph_ksize >= 3:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
                keep_refined = cv2.morphologyEx(keep_refined, cv2.MORPH_CLOSE, k, iterations=1)
                keep_refined = cv2.morphologyEx(keep_refined, cv2.MORPH_OPEN,  k, iterations=1)
            dt = time.time() - t0
            print(f"[INFO] Fast refine (iters={iters}, scale={scale:.2f}, box={(x0,y0,x1,y1)}) in {dt:.2f}s")
            return keep_refined.astype(np.uint8)

        # No refinement requested; just return cleaned mask
        return keep_bin.astype(np.uint8)

    if refine_hr_iters and refine_hr_iters > 0:
        print(f"[INFO] Refining at full-res with INIT_WITH_MASK (iters={refine_hr_iters})...")
        t1 = time.time()
        # Build GrabCut label mask from binary keep
        mask_hr = np.where(keep>0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        # Strongly mark the ROI interior as probable FG to guide refinement
        mask_hr[y:y+h, x:x+w] = cv2.GC_PR_FGD
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        keep = _refine_with_mask_fast(
        image_bgr=image_bgr,
        keep_bin=keep.astype(np.uint8),
        iters=refine_hr_iters,
        margin_frac=0.08,   # expand bbox by 8%
        scale=0.5,          # run refine at 50% scale
        morph_ksize=5
    )
        print(f"[INFO] Full-res refine done in {time.time()-t1:.2f}s.")

    return keep

# ---------- symmetry plane estimation & refinement ----------
def _normalize(v): n=np.linalg.norm(v); return v/(n+1e-12)

def _pixel_to_ray_dir(u,v,K):
    invK = np.linalg.inv(K).astype(np.float32)
    vec = invK @ np.array([u,v,1.0], np.float32)
    return _normalize(vec)

def _estimate_plane_from_clicks(image_bgr, metric_depth, K,
                                mode="midline_plus_optical_z", win=9, mask=None):
    """
    Returns (n, d) for plane n·X + d = 0. Uses robust picker with timeout+fallback.
    """
    (u1, v1), (u2, v2) = _pick_two_points(image_bgr, "Car midline (2 pts)", mask=mask, timeout_sec=45)

    if mode == "origin_rays":
        r1 = _pixel_to_ray_dir(u1, v1, K); r2 = _pixel_to_ray_dir(u2, v2, K)
        n = _normalize(np.cross(r1, r2)); d = 0.0
        return n.astype(np.float32), float(d)

    # midline_plus_optical_z (depth-aware)
    Z1 = _local_median_depth(metric_depth, u1, v1, win=win)
    Z2 = _local_median_depth(metric_depth, u2, v2, win=win)
    if not (np.isfinite(Z1) and Z1 > 0 and np.isfinite(Z2) and Z2 > 0):
        r1 = _pixel_to_ray_dir(u1, v1, K); r2 = _pixel_to_ray_dir(u2, v2, K)
        n = _normalize(np.cross(r1, r2)); return n.astype(np.float32), 0.0

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    P1 = np.array([(u1 - cx) * Z1 / fx, (v1 - cy) * Z1 / fy, Z1], np.float32)
    P2 = np.array([(u2 - cx) * Z2 / fx, (v2 - cy) * Z2 / fy, Z2], np.float32)

    l_dir = _normalize(P2 - P1)
    z_cam = np.array([0, 0, 1], np.float32)
    n = np.cross(l_dir, z_cam)
    if np.linalg.norm(n) < 1e-6:
        r1 = _pixel_to_ray_dir(u1, v1, K); r2 = _pixel_to_ray_dir(u2, v2, K)
        n = np.cross(r1, r2)
    n = _normalize(n)
    C = 0.5 * (P1 + P2)
    d = -float(np.dot(n, C))
    return n.astype(np.float32), d

def _mirror_points(points, n, d=0.0):
    # reflect across plane n·X + d = 0
    proj = (points @ n) + d
    return points - 2.0 * proj[:,None] * n[None,:]

def _refine_plane(points, n0, d0, pair_radius=0.25, iters=5, subsample=200000):
    n, d = n0.astype(np.float32), float(d0)
    pts = points
    if pts.shape[0] > subsample:
        idx = np.random.default_rng(0).choice(pts.shape[0], subsample, replace=False)
        pts = pts[idx]
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points.astype(np.float32))
    except Exception:
        tree = None

    for _ in range(iters):
        Pm = _mirror_points(pts, n, d)
        if tree is None:
            # brute-force tiny subset fallback
            idx = np.random.default_rng(1).choice(points.shape[0], min(20000, points.shape[0]), replace=False)
            Q = points[idx]
            dists = ((Pm[:,None,:]-Q[None,:,:])**2).sum(-1)
            nn = dists.argmin(1)
            pairs_ok = np.arange(Pm.shape[0])
            Qn = Q[nn]
        else:
            dist, nn = tree.query(Pm, k=1, distance_upper_bound=pair_radius)
            pairs_ok = np.where(np.isfinite(dist))[0]
            if pairs_ok.size < 1000: break
            Qn = points[nn[pairs_ok]]

        Pm_ok = Pm[pairs_ok]
        M = 0.5*(Pm_ok + Qn)
        V = Qn - Pm_ok
        Vc = V - V.mean(0)
        _,_,Vt = np.linalg.svd(Vc, full_matrices=False)
        n = _normalize(Vt[0].astype(np.float32))
        d = -float(np.median(M @ n))
    return n, d

# ---------- voxel downsample & binary PLY ----------
def _voxel_downsample(points, colors=None, voxel=0.015):
    if points.shape[0] == 0: return points, colors
    g = np.floor(points / voxel).astype(np.int64)
    key = (g[:,0] << 42) ^ (g[:,1] << 21) ^ g[:,2]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    n = uniq.shape[0]
    P = np.zeros((n,3), np.float64)
    for i in range(3):
        P[:,i] = np.bincount(inv, weights=points[:,i], minlength=n) / cnt
    P = P.astype(np.float32)
    if colors is None: return P, None
    C = np.zeros((n,3), np.float64)
    for i in range(3):
        C[:,i] = np.bincount(inv, weights=colors[:,i], minlength=n) / cnt
    C = np.clip(np.round(C), 0, 255).astype(np.uint8)
    return P, C

def _save_ply_binary_little(path, points, colors=None):
    points = points.astype(np.float32, copy=False)
    n = points.shape[0]
    has_color = colors is not None
    if has_color:
        colors = colors.astype(np.uint8, copy=False)
        assert colors.shape[0]==n and colors.shape[1]==3
    header = [
        "ply","format binary_little_endian 1.0",f"element vertex {n}",
        "property float x","property float y","property float z"
    ]
    if has_color:
        header += ["property uchar red","property uchar green","property uchar blue"]
    header += ["end_header\n"]
    with open(path,"wb") as f:
        f.write(("\n".join(header)).encode("ascii"))
        if has_color:
            rec = np.empty(n, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                                     ("r","u1"),("g","u1"),("b","u1")])
            rec["x"],rec["y"],rec["z"]=points[:,0],points[:,1],points[:,2]
            rec["r"],rec["g"],rec["b"]=colors[:,0],colors[:,1],colors[:,2]
            rec.tofile(f)
        else:
            points.astype("<f4").tofile(f)

# ---------- public entrypoint ----------
def export_car_symmetric_cloud(image_bgr, metric_depth, K,
                               out_path="car_sym_refined_vox15mm.ply",
                               z_clip=(0.2, 80.0),
                               median_ksize=3,
                               stride=1,
                               plane_mode="midline_plus_optical_z",
                               refine=True,
                               pair_radius=0.25,
                               iters=5,
                               use_grabcut=True,
                               voxel=0.015,
                               combine=True):
    """
    Build a bilateral-symmetric car point cloud.
      - image_bgr: original BGR image (H,W,3)
      - metric_depth: depth in meters (H,W), already scaled
      - K: 3x3 intrinsics
    Steps:
      1) optional mask (GrabCut box)
      2) clean depth and backproject (dense, stride)
      3) estimate symmetry plane from 2 midline clicks
      4) optional plane refinement (align mirrored pairs)
      5) mirror, combine, voxel downsample
      6) save binary PLY
    """
    if use_grabcut:
        mask = _grabcut_mask(image_bgr)
    else:
        mask = np.ones(image_bgr.shape[:2], np.uint8)

    D = _preprocess_depth(metric_depth, z_clip=z_clip, median_ksize=median_ksize)
    pts, cols = _backproject_dense(D, K, rgb_bgr=image_bgr, mask=mask, stride=stride)
    if pts.shape[0] == 0:
        raise RuntimeError("No valid points to export (check z_clip/mask/stride).")

    n0, d0 = _estimate_plane_from_clicks(image_bgr, D, K, mode=plane_mode, win=9, mask=mask)

    n, d = (n0, d0)
    if refine:
        n, d = _refine_plane(pts, n0, d0, pair_radius=pair_radius, iters=iters, subsample=250000)

    pts_m = _mirror_points(pts, n, d)
    if combine:
        pts_all = np.vstack([pts, pts_m])
        cols_all = np.vstack([cols, cols]) if cols is not None else None
    else:
        pts_all, cols_all = pts_m, (cols.copy() if cols is not None else None)

    pts_ds, cols_ds = _voxel_downsample(pts_all, cols_all, voxel=voxel)
    _save_ply_binary_little(out_path, pts_ds, cols_ds)
    print(f"[INFO] Saved: {out_path}  points={len(pts_ds)}  plane n={n}, d={d:.4f}")
    return pts_ds, cols_ds, (n, d)
