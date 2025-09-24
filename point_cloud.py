# ---- pointcloud_export.py ----
import numpy as np
import cv2

def preprocess_depth_for_pc(depth_m,
                            z_clip=(0.05, None),
                            median_ksize=3,
                            joint_bilateral=False,
                            guide_rgb=None,
                            jb_d=9, jb_sigma_color=12.0, jb_sigma_space=7.0):
    """
    Clean depth a bit before backprojection.
      - z_clip=(zmin, zmax): clip invalid too-near/too-far values
      - median_ksize: small median filter to kill salt-and-pepper noise (0/None to disable)
      - joint_bilateral: if True and guide_rgb provided, run edge-preserving smoothing guided by RGB
    Returns cleaned depth (float32) with NaN at invalid pixels.
    """
    D = depth_m.astype(np.float32).copy()

    # clip
    zmin, zmax = z_clip
    if zmin is not None:
        D[D < zmin] = np.nan
    if zmax is not None:
        D[D > zmax] = np.nan

    # median filter on valid values only (simple inpainting approach)
    if median_ksize and median_ksize >= 3:
        D_filled = D.copy()
        nan_mask = ~np.isfinite(D_filled)
        if np.any(nan_mask):
            # fill NaNs with local median for filtering
            # quick fill: nearest valid via cv2.inpaint needs 8U mask; use simple fallback
            D_filled[nan_mask] = 0.0
        D_med = cv2.medianBlur(D_filled, ksize=int(median_ksize))
        # keep original invalids
        D = np.where(np.isfinite(D), D_med, np.nan)

    # optional joint bilateral (only if guide and ximgproc available)
    if joint_bilateral and guide_rgb is not None:
        try:
            import cv2.ximgproc as xip
            guide = cv2.cvtColor(guide_rgb, cv2.COLOR_BGR2RGB) if guide_rgb.shape[2] == 3 else guide_rgb
            guide = guide.astype(np.uint8)
            # normalize depth to 0..255 for filter, then bring back
            dmin = np.nanmin(D); dmax = np.nanmax(D)
            if np.isfinite(dmin) and np.isfinite(dmax) and dmax > dmin:
                Dn = (D - dmin) / (dmax - dmin)
                Dn[~np.isfinite(Dn)] = 0
                D8 = (Dn * 255).astype(np.uint8)
                D8f = xip.jointBilateralFilter(guide, D8, d=jb_d,
                                               sigmaColor=jb_sigma_color, sigmaSpace=jb_sigma_space)
                D = (D8f.astype(np.float32) / 255.0) * (dmax - dmin) + dmin
        except Exception:
            pass

    return D

def backproject_depth_to_points(depth_m, K, rgb_bgr=None, mask=None, stride=1):
    """
    Vectorized backprojection:
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy
      Z = depth
    - stride: sample every Nth pixel to reduce point count (1 = dense)
    - mask: boolean same size as depth, True = keep
    Returns (Nx3 points, Nx3 uint8 colors or None)
    """
    H, W = depth_m.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # grid
    us = np.arange(0, W, stride, dtype=np.float32)
    vs = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    Z = depth_m[::stride, ::stride]

    valid = np.isfinite(Z) & (Z > 0)
    if mask is not None:
        valid &= mask[::stride, ::stride].astype(bool)

    if not np.any(valid):
        return np.empty((0,3), np.float32), None

    uu = uu[valid]; vv = vv[valid]; Z = Z[valid]
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    pts = np.vstack((X, Y, Z)).T.astype(np.float32)

    cols = None
    if rgb_bgr is not None and rgb_bgr.size:
        C = rgb_bgr[::stride, ::stride, :]
        C = C[valid]
        cols = C[:, ::-1].copy()  # convert BGR->RGB for PLY
        cols = cols.astype(np.uint8)
    return pts, cols

def mirror_pointcloud(points, colors=None, plane='yz', combine=True):
    """
    Mirror across a canonical plane through the camera origin:
      - 'yz' plane: X -> -X (left-right symmetry)
      - 'xz' plane: Y -> -Y (up-down)
      - 'xy' plane: Z -> -Z (front-back; rarely useful)
    If combine=True, returns concatenated original + mirrored.
    """
    pts_m = points.copy()
    if plane == 'yz':
        pts_m[:,0] *= -1
    elif plane == 'xz':
        pts_m[:,1] *= -1
    elif plane == 'xy':
        pts_m[:,2] *= -1
    else:
        raise ValueError("plane must be one of {'yz','xz','xy'}")
    if colors is None:
        if combine:
            return np.vstack([points, pts_m]), None
        return pts_m, None
    else:
        cols_m = colors.copy()
        if combine:
            return np.vstack([points, pts_m]), np.vstack([colors, cols_m])
        return pts_m, cols_m

def save_ply_ascii(path, points, colors=None):
    """
    Save PLY (ASCII). points: Nx3 float, colors: Nx3 uint8 (RGB) or None.
    """
    n = points.shape[0]
    has_color = colors is not None and colors.shape[0] == n
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if has_color:
            for (x,y,z), (r,g,b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x,y,z) in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

def export_pointcloud(image_bgr, metric_depth, K,
                      out_path="cloud.ply",
                      stride=1,
                      z_clip=(0.05, None),
                      median_ksize=3,
                      joint_bilateral=False,
                      assume_symmetry=False,
                      symmetry_plane='yz'):
    """
    High-level one-call export:
      - cleans depth
      - backprojects to 3D with color
      - optional mirroring across symmetry plane (combine original + mirrored)
      - saves ASCII PLY
    """
    D = preprocess_depth_for_pc(metric_depth,
                                z_clip=z_clip,
                                median_ksize=median_ksize,
                                joint_bilateral=joint_bilateral,
                                guide_rgb=image_bgr)
    pts, cols = backproject_depth_to_points(D, K, rgb_bgr=image_bgr, stride=stride)
    if pts.shape[0] == 0:
        raise RuntimeError("No valid 3D points after preprocessing.")

    if assume_symmetry:
        pts, cols = mirror_pointcloud(pts, cols, plane=symmetry_plane, combine=True)

    save_ply_ascii(out_path, pts, cols)
    return pts, cols
