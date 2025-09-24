# axis_symmetric_recon.py
import numpy as np
import cv2

def pick_roi(image_bgr, title="Select bottle ROI"):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, int(1280*image_bgr.shape[0]/image_bgr.shape[1]))
    roi = cv2.selectROI(title, image_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    x,y,w,h = map(int, roi)
    if w <= 0 or h <= 0: return None
    return (x,y,w,h)

def backproject_depth_to_points(depth_m, K, rgb_bgr=None, mask=None):
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    Z = depth_m
    valid = np.isfinite(Z) & (Z > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    uu = uu[valid]; vv = vv[valid]; Z = Z[valid]
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)
    cols = None
    if rgb_bgr is not None and rgb_bgr.size:
        C = rgb_bgr.reshape(-1,3)[np.ravel(valid)]
        cols = C[:, ::-1].astype(np.uint8)  # to RGB
    return pts, cols

def gaussian1d(x, sigma):
    if sigma <= 0: return np.array([1.0], dtype=np.float32)
    k = int(max(3, round(3*sigma)))
    t = np.arange(-k, k+1, dtype=np.float32)
    g = np.exp(-(t*t)/(2*sigma*sigma))
    g /= g.sum()
    return g

def orthonormal_basis_from_axis(a):
    a = a / (np.linalg.norm(a)+1e-12)
    # pick any vector not parallel to a
    t = np.array([1.0,0,0], np.float32)
    if abs(np.dot(t,a)) > 0.9: t = np.array([0,1.0,0], np.float32)
    b1 = np.cross(a, t); b1 /= (np.linalg.norm(b1)+1e-12)
    b2 = np.cross(a, b1); b2 /= (np.linalg.norm(b2)+1e-12)
    return a, b1, b2

def estimate_axis_pca(points):
    C = points.mean(axis=0)
    Q = points - C
    # largest variance direction
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    a = Vt[0]  # principal axis
    # ensure "up"-ish (optional)
    if a[1] < 0: a = -a
    return C.astype(np.float32), a.astype(np.float32)

def cylindrical_coords(points, C, a, b1, b2):
    """
    express p as: p = C + a*z + r*cos(theta)*b1 + r*sin(theta)*b2
    returns (z, r, theta)
    """
    Q = points - C
    z = Q @ a
    x1 = Q @ b1
    x2 = Q @ b2
    r = np.sqrt(x1*x1 + x2*x2)
    theta = np.arctan2(x2, x1)
    return z, r, theta

def build_surface_of_revolution(C, a, b1, b2, z_vals, r_vals, n_theta=180):
    """
    Generate a triangle mesh by revolving r(z) around axis a through C.
    C,a,b1,b2: float32 (3,)
    z_vals, r_vals: (nz,)
    """
    # ensure types
    C  = np.asarray(C,  dtype=np.float32)
    a  = np.asarray(a,  dtype=np.float32)
    b1 = np.asarray(b1, dtype=np.float32)
    b2 = np.asarray(b2, dtype=np.float32)
    z_vals = np.asarray(z_vals, dtype=np.float32)
    r_vals = np.asarray(r_vals, dtype=np.float32)

    # re-orthonormalize basis
    a = a / (np.linalg.norm(a) + 1e-12)
    b1 = b1 - a * np.dot(a, b1); b1 /= (np.linalg.norm(b1) + 1e-12)
    b2 = np.cross(a, b1);        b2 /= (np.linalg.norm(b2) + 1e-12)

    nz = len(z_vals)
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False, dtype=np.float32)
    # shapes: (1, nt, 1) and (1,1,3)
    cosT = np.cos(thetas)[None, :, None]
    sinT = np.sin(thetas)[None, :, None]
    b1v  = b1[None, None, :]
    b2v  = b2[None, None, :]

    # ring directions for one unit radius: (1, nt, 3)
    ring_dirs = cosT * b1v + sinT * b2v  # (1, nt, 3)

    # centers along axis: (nz, 1, 3)
    centers = C[None, None, :] + z_vals[:, None, None] * a[None, None, :]

    # radii as (nz, 1, 1)
    rr = r_vals[:, None, None]

    # full ring points: (nz, nt, 3)
    ring = centers + rr * ring_dirs

    verts = ring.reshape(-1, 3).astype(np.float32)

    # faces
    faces = []
    nt = n_theta
    for i in range(nz - 1):
        base0 = i * nt
        base1 = (i + 1) * nt
        for j in range(nt):
            jn = (j + 1) % nt
            v00 = base0 + j
            v01 = base0 + jn
            v10 = base1 + j
            v11 = base1 + jn
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.asarray(faces, dtype=np.int32)
    return verts, faces


def save_mesh_ply(path, V, F, color=None):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(V)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if color is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(F)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        if color is not None:
            c = np.clip(color, 0, 255).astype(np.uint8)
            for (x,y,z), (r,g,b) in zip(V, c):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x,y,z) in V:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for (i,j,k) in F:
            f.write(f"3 {i} {j} {k}\n")

def axis_symmetric_mesh(image_bgr, metric_depth, K,
                        roi=None,
                        bins_z=200,
                        smooth_sigma=2.0,
                        n_theta=240,
                        assume_axis="pca"):  # or "vertical"
    """
    Build surface of revolution from a single metric depth map.
    Returns vertex array (Nx3) and face array (Mx3).
    """
    H, W = metric_depth.shape
    # ROI mask (optional but strongly recommended)
    if roi is None:
        roi = pick_roi(image_bgr)
    mask = None
    if roi is not None:
        x,y,w,h = roi
        mask = np.zeros((H,W), np.uint8)
        mask[y:y+h, x:x+w] = 1

    # backproject points inside ROI
    pts, _ = backproject_depth_to_points(metric_depth, K, rgb_bgr=None, mask=mask)
    if pts.shape[0] < 500:
        raise RuntimeError("Too few valid points in ROI.")

    # estimate symmetry axis
    if assume_axis == "vertical":
        C = pts.mean(axis=0).astype(np.float32)
        a = np.array([0,1,0], np.float32)  # camera Y
    else:
        C, a = estimate_axis_pca(pts)

    a, b1, b2 = orthonormal_basis_from_axis(a)

    # cylindrical coordinates
    z, r, _ = cylindrical_coords(pts, C, a, b1, b2)

    # robust r(z): median in bins
    zmin, zmax = np.percentile(z, [1, 99])
    z_edges = np.linspace(zmin, zmax, bins_z+1)
    z_centers = 0.5*(z_edges[:-1] + z_edges[1:])
    r_profile = np.zeros_like(z_centers, dtype=np.float32)
    for i in range(bins_z):
        m = (z >= z_edges[i]) & (z < z_edges[i+1])
        if np.any(m):
            r_profile[i] = np.median(r[m])
        else:
            r_profile[i] = np.nan
    # inpaint small gaps by linear interp
    valid = np.isfinite(r_profile)
    if np.sum(valid) < 5:
        raise RuntimeError("Not enough valid bins for radius profile.")
    r_profile[~valid] = np.interp(z_centers[~valid], z_centers[valid], r_profile[valid])
    # smooth
    ker = gaussian1d(r_profile, smooth_sigma)
    r_sm = np.convolve(r_profile, ker, mode='same')

    # revolve to mesh
    V, F = build_surface_of_revolution(C, a, b1, b2, z_centers.astype(np.float32), r_sm, n_theta=n_theta)
    return V, F
