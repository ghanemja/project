#!/usr/bin/env python3
# server.py — Rover CAD viewer/editor (GLB pipeline with real “add” support)

import io, os, sys, json, re, base64, threading, mimetypes, ast
from typing import Dict, Any, Optional, List, Tuple

# freecad_bootstrap.py
import sys, importlib.util

def load_freecad():
    path = '/home/ec2-user/Documents/cad-optimizer/squashfs-root/usr/lib/FreeCAD.so'
    spec = importlib.util.spec_from_file_location("FreeCAD", path)
    FreeCAD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FreeCAD)
    sys.modules["FreeCAD"] = FreeCAD  # ← so others can import it later
    return FreeCAD

FreeCAD = load_freecad()

import requests
import cadquery as cq
import trimesh
from flask import (
    Flask,
    Response,
    request,
    jsonify,
    send_file,
    send_from_directory,
    abort,
    render_template,
)

# ----------------- keep your existing HTML block unchanged -----------------
# Reuse the exact HTML you already have in your file.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from existing_html_module import HTML  # placeholder for type-checkers
# If your file already defines HTML = """...""", leave it as-is:
# ----------------- Rover & parts from your repo -----------------
from robot_base import Rover
from electronics import type1 as _Electronics
from pan_tilt import PanTilt as _PanTilt
from wheel import BuiltWheel as _ThisWheel
from cqparts_motors.stepper import Stepper as _Stepper

# Example new part with mates (your file sensor_fork.py)
from sensor_fork import SensorFork  # expects .mate_mount() and .mate_sensor()

# MIME fix for ESM
mimetypes.add_type("application/javascript", ".js")

# ----------------- VLM config -----------------
OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
).rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava-llama3:latest")
LLAVA_URL = os.environ.get("LLAVA_URL")  # optional alternative

# ---------- robust request parsing & normalization ----------

TARGET_ALIASES = {
    "motor_controllerboard": "motor_controller_board",
    "motorcontrollerboard": "motor_controller_board",
    "motor controller board": "motor_controller_board",
    "motorcontroller": "motor_controller",
    "motor": "motor_controller",     # if you don't have this component, it'll be ignored later
    "sensorsbase": "sensor_fork",
    "sensor": "sensor_fork",
    "sensors": "sensor_fork",
    "wheels": "wheel",
}

ACTION_ALIASES = {
    "move": "translate",
    "position": "translate",
    "pos": "translate",
    "orientation": "rotate",
    "orient": "rotate",
    "size": "modify",
    "count": "add",                 # we'll keep parameters.count if present
    "wheels_per_side": "modify",    # action misused as key → treat as modify
}

def _strip_units_to_float(val):
    # turns "200mm" -> 200.0, " 45 deg " -> 45.0
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    s = re.sub(r"[^0-9eE+\-\.]", "", s)
    try:
        return float(s)
    except Exception:
        return None

def _normalize_params(target: str, action: str, params: dict) -> dict:
    """Fix common param name issues and convert units. Returns a new dict."""
    p = {}
    params = params or {}

    # normalize keys
    for k, v in params.items():
        kk = str(k).strip().lower().replace(" ", "_")
        # map basic translations
        if kk in ("x", "dx", "x_mm"):
            p["dx_mm"] = _strip_units_to_float(v)
        elif kk in ("y", "dy", "y_mm"):
            p["dy_mm"] = _strip_units_to_float(v)
        elif kk in ("z", "dz", "z_mm"):
            p["dz_mm"] = _strip_units_to_float(v)
        elif kk in ("diameter", "diameter_mm"):
            # for wheels, your param_map uses wheel_diameter -> diameter
            p["wheel_diameter"] = _strip_units_to_float(v)
        elif kk in ("width_mm", "height_mm", "depth_mm", "wall_mm", "hole_diam_mm",
                    "axle_spacing_mm", "wheelbase_span_mm", "wheels_per_side",
                    "wheel_diameter", "wheel_width"):
            p[kk] = _strip_units_to_float(v)
        elif kk in ("position_mm", "orientation_deg"):
            # keep arrays as-is but clean numbers if list-like
            if isinstance(v, (list, tuple)) and len(v) in (2,3):
                p[kk] = [ _strip_units_to_float(x) for x in v ]
            else:
                p[kk] = v
        else:
            # default: try to coerce numeric, else pass through
            num = _strip_units_to_float(v)
            p[kk] = num if num is not None else v

    # Special cases:
    # If action was an alias 'wheels_per_side' (treated as modify), value
    # might sit in parameters already; nothing else to do here.

    # If this is pan_tilt translate, map dx/dy/dz into pan_tilt offsets
    tgt = (target or "").lower()
    if ACTION_ALIASES.get(action, action) == "translate" and tgt == "pan_tilt":
        if "dx_mm" in p: p["pan_tilt_offset_x"] = p.pop("dx_mm")
        if "dy_mm" in p: p["pan_tilt_offset_y"] = p.pop("dy_mm")
        if "dz_mm" in p: p["pan_tilt_offset_z"] = p.pop("dz_mm")

    # For sensor_fork, map dx/dy/dz into a position list the add() path expects
    if tgt == "sensor_fork":
        if any(k in p for k in ("dx_mm","dy_mm","dz_mm")):
            px = p.pop("dx_mm", 0.0) or 0.0
            py = p.pop("dy_mm", 0.0) or 0.0
            pz = p.pop("dz_mm", 0.0) or 0.0
            p.setdefault("position_mm", [px, py, pz])

    return p

def _normalize_change(ch: dict) -> dict | None:
    """Normalize a single change dict: action/target aliases, params, sanity."""
    if not isinstance(ch, dict):
        return None

    action = (ch.get("action") or "").strip().lower()
    action = ACTION_ALIASES.get(action, action)

    target = (ch.get("target_component") or "").strip().lower()
    target = TARGET_ALIASES.get(target, target)

    params = _normalize_params(target, action, ch.get("parameters") or {})

    # Some VLMs misuse action for 'wheels_per_side'—keep as modify, ensure param present
    if action == "modify" and "wheels_per_side" in ch.get("action", "").lower():
        params.setdefault("wheels_per_side", _strip_units_to_float(ch.get("parameters", {}).get("wheels_per_side")))

    # If after normalization we still have no target or action, drop the change
    if not target or not action:
        return None

    out = dict(ch)
    out["action"] = action
    out["target_component"] = target
    out["parameters"] = params
    return out

def _extract_all_json_blocks(text: str) -> list:
    """
    From a noisy text (like your 'recommend raw' dump), extract ALL lists/dicts,
    parse them, and flatten to a single list of change dicts.
    """
    changes = []
    if not text:
        return changes

    # Find every {...} or [...] block (non-greedy)
    for m in re.finditer(r"(\[[\s\S]*?\]|\{[\s\S]*?\})", text):
        block = (m.group(1) or "").strip()
        if not block:
            continue
        parsed = None
        # Try JSON then literal_eval
        try:
            parsed = json.loads(block)
        except Exception:
            try:
                parsed = ast.literal_eval(block)
            except Exception:
                parsed = None
        if parsed is None:
            continue
        # Normalize to list
        seq = parsed if isinstance(parsed, list) else [parsed]
        for ch in seq:
            norm = _normalize_change(ch)
            if norm:
                changes.append(norm)
    return changes

def _extract_json_loose(text: str):
    """(kept from earlier answer) single-block best-effort extraction."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])\s*$', text.strip())
        if m:
            block = m.group(1)
            try:
                return json.loads(block)
            except Exception:
                try:
                    return ast.literal_eval(block)
                except Exception:
                    return None
    return None

def _parse_apply_request():
    """
    Accepts:
      - application/json (optionally with {"excerpt": "...", "actions"/"changes": [...]})
      - multipart/form-data fields: json=..., excerpt=...
      - raw text with many JSON blocks + trailing prose
      - VLM wrapper: {"response":{"raw": "...", "json": <obj>}}
    Returns: (changes_list, excerpt_str)
    """
    excerpt = None

    # 1) Try JSON body
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        # VLM wrapper case
        if "response" in data and isinstance(data["response"], dict):
            inner = data["response"].get("json")
            raw_text = data["response"].get("raw")
            if inner is not None:
                payload = inner.get("actions") or inner.get("changes") or inner
                items = _coerce_changes(payload)
                # normalize each
                changes = []
                for ch in items:
                    norm = _normalize_change(ch)
                    if norm:
                        changes.append(norm)
                if not changes and raw_text:
                    # fall back to scanning raw
                    changes = _extract_all_json_blocks(raw_text)
                excerpt = data.get("excerpt") or data.get("summary")
                return changes, excerpt

        excerpt = data.get("excerpt") or data.get("summary") or None
        payload = data.get("actions") or data.get("changes") or data
        items = _coerce_changes(payload)
        changes = []
        for ch in items:
            norm = _normalize_change(ch)
            if norm:
                changes.append(norm)
        return changes, excerpt

    if isinstance(data, list):
        changes = []
        for ch in data:
            norm = _normalize_change(ch)
            if norm:
                changes.append(norm)
        return changes, None

    # 2) Multipart form
    if request.form:
        raw_json = (
            request.form.get("json")
            or request.form.get("changes")
            or request.form.get("actions")
            or ""
        )
        excerpt = request.form.get("excerpt") or request.form.get("summary") or None
        # Try single payload first
        payload = _extract_json_loose(raw_json)
        items = _coerce_changes(payload)
        changes = []
        for ch in items:
            norm = _normalize_change(ch)
            if norm:
                changes.append(norm)
        # If empty, scan for all blocks
        if not changes:
            changes = _extract_all_json_blocks(raw_json)
        return changes, excerpt

    # 3) Raw text (like your 'recommend raw' dump)
    raw_text = request.get_data(as_text=True) or ""
    changes = _extract_all_json_blocks(raw_text)
    return changes, excerpt


VLM_SYSTEM_PROMPT = """You are a visual large language model assisting with 3D CAD model editing.

Context:
- The model is a parametric robot rover built in CadQuery / cqparts.
- The UI provides a list of component classes and the optional selected class.
- A reference image may be provided; propose precise, conservative param updates or true 'add' operations.

# OUTPUT CONTRACT (STRICT)
First, output ONLY a strict JSON payload (either a single object or a list of objects), matching this schema exactly:
[
  {
    "target_component": "<class_or_type>",
    "action": "<modify|replace|resize|rotate|translate|delete|add>",
    "parameters": { "...": ... },
    "rationale": "one brief sentence",
    "title": "optional short label for UI (<= 8 words)",
    "confidence": 0.0
  }
]

Immediately AFTER the JSON, add a blank line, then output a human-readable summary line that starts with:
SUMMARY:
…and concisely explains all proposed changes in 1–3 sentences.

No markdown fences, no commentary before the JSON, and do not wrap the JSON in any other envelope.

# RULES AND GUIDELINES

General:
- Assume the CAD snapshot represents the model produced by the given CAD state JSON.
- Identify mismatches between the reference image and the current CAD (sizes, positions, counts, orientations, presence/absence).
- Propose changes conservatively; when unsure, prefer small deltas.
- Include a short 'rationale' for each change and a 'confidence' field in [0.0–1.0].

'add' rules:
- When action = "add", target_component is the new part type (e.g., "sensor_fork", "wheel", etc.).
- For wheels, support both "count" and "wheels_per_side".
- Parameters may include:
  - "count" or "wheels_per_side"
  - "position_mm": [x, y, z]
  - "orientation_deg": [rx, ry, rz]
  - other geometric or dimensional attributes (width_mm, height_mm, etc.)
- If unsure of a numeric value, include null and briefly explain in 'rationale'.
"""


# ----------------- cqparts v1→v2 tiny shims (safe no-ops) -----------------
try:
    from cqparts.utils.geometry import CoordSystem
except Exception:

    class CoordSystem:
        def __sub__(self, other):
            return self

        def __rsub__(self, other):
            return other


cq.Workplane.world_coords = property(lambda self: CoordSystem())
cq.Workplane.local_coords = property(lambda self: CoordSystem())


# ----------------- Component registry & adapter -----------------
class ComponentSpec:
    def __init__(self, cls, add_fn=None, param_map=None):
        self.cls = cls
        self.add_fn = add_fn  # callable(adapter_or_model, **params)
        self.param_map = param_map or {}  # {"json_param": "class_attr"}


COMPONENT_REGISTRY: Dict[str, ComponentSpec] = {}


def register_component(key: str, spec: ComponentSpec):
    COMPONENT_REGISTRY[key.lower()] = spec


def get_component_spec(key: str) -> Optional[ComponentSpec]:
    return COMPONENT_REGISTRY.get(key.lower())


# queued ops consumed at build time (true geometry adds)
PENDING_ADDS: List[dict] = []  # e.g. {"kind":"sensor_fork","params":{...}}


class ModelAdapter:
    def __init__(self, model_cls):
        self.model_cls = model_cls

    def add(self, kind: str, **params):
        PENDING_ADDS.append({"kind": kind.lower(), "params": params})


ADAPTER = ModelAdapter(Rover)

# Register built-ins
register_component(
    "wheel",
    ComponentSpec(
        cls=_ThisWheel,
        add_fn=lambda model, **p: (
            "wheel",
            None,
        ),  # wheels added via model param (wheels_per_side)
        param_map={"wheel_diameter": "diameter", "wheel_width": "width"},
    ),
)
register_component(
    "pan_tilt",
    ComponentSpec(
        cls=_PanTilt,
        add_fn=None,
        param_map={
            "pan_tilt_offset_x": "pan_tilt_offset_x",
            "pan_tilt_offset_y": "pan_tilt_offset_y",
            "pan_tilt_offset_z": "pan_tilt_offset_z",
        },
    ),
)
register_component(
    "sensor_fork",
    ComponentSpec(
        cls=SensorFork,
        add_fn=lambda adapter, **p: ADAPTER.add("sensor_fork", **p),
        param_map={
            "width_mm": "width",
            "depth_mm": "depth",
            "height_mm": "height",
            "wall_mm": "wall",
            "hole_diam_mm": "hole_diam",
        },
    ),
)

# ----------------- App, state, config -----------------
PORT = int(os.environ.get("PORT", "5160"))
BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
ROVER_GLB_PATH = os.path.join(ASSETS_DIR, "rover.glb")
USE_CQPARTS = os.environ.get("USE_CQPARTS", "1") == "1"

app = Flask(__name__, static_folder="static")
STATE: Dict[str, Any] = {"selected_parts": []}

# ---------- live params (also adds terrain context) ----------
CURRENT_PARAMS: Dict[str, Optional[float]] = {
    "wheel_diameter": None,
    "wheel_width": None,
    "pan_tilt_offset_x": None,
    "pan_tilt_offset_y": None,
    "pan_tilt_offset_z": None,
    "wheels_per_side": None,
    "axle_spacing_mm": None,
    "wheelbase_span_mm": None,
}
# non-numeric context
CONTEXT: Dict[str, Any] = {"terrain_mode": "flat"}  # or "uneven"

HISTORY: List[Dict[str, Optional[float]]] = []
H_PTR: int = -1


def _snapshot() -> Dict[str, Optional[float]]:
    return {k: (float(v) if v is not None else None) for k, v in CURRENT_PARAMS.items()}


def _push_history():
    global H_PTR, HISTORY
    if H_PTR < len(HISTORY) - 1:
        HISTORY = HISTORY[: H_PTR + 1]
    HISTORY.append(_snapshot())
    H_PTR = len(HISTORY) - 1


def _restore(snapshot: Dict[str, Optional[float]]):
    for k in CURRENT_PARAMS.keys():
        CURRENT_PARAMS[k] = snapshot.get(k, CURRENT_PARAMS[k])


def _clean_num(v):
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def apply_params_to_rover(rv, params: Dict[str, Any] | None):
    if params:
        for k, v in params.items():
            if k in CURRENT_PARAMS:
                CURRENT_PARAMS[k] = _clean_num(v)

    # wheel size
    if CURRENT_PARAMS["wheel_diameter"] is not None:
        setattr(_ThisWheel, "diameter", float(CURRENT_PARAMS["wheel_diameter"]))
    if CURRENT_PARAMS["wheel_width"] is not None:
        setattr(_ThisWheel, "width", float(CURRENT_PARAMS["wheel_width"]))

    # wheel layout
    if CURRENT_PARAMS["wheels_per_side"] is not None:
        setattr(Rover, "wheels_per_side", int(CURRENT_PARAMS["wheels_per_side"]))
    for k in ("axle_spacing_mm", "wheelbase_span_mm"):
        if CURRENT_PARAMS[k] is not None:
            setattr(Rover, k, float(CURRENT_PARAMS[k]))

    # pan-tilt translation
    for axis in ("x", "y", "z"):
        key = f"pan_tilt_offset_{axis}"
        if CURRENT_PARAMS[key] is not None:
            setattr(_PanTilt, key, float(CURRENT_PARAMS[key]))


# ----------------- helpers: uploads and VLM -----------------
def _data_url_from_upload(file_storage) -> Optional[str]:
    if not file_storage:
        return None
    raw = file_storage.read()
    mime = file_storage.mimetype or "application/octet-stream"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def call_vlm(
    final_prompt: str, image_data_urls: Optional[List[str] | str]
) -> Dict[str, Any]:
    def _normalize(imgs):
        if not imgs:
            return None
        if isinstance(imgs, str):
            imgs = [imgs]
        out = []
        for u in imgs:
            if not u:
                continue
            out.append(u.split(",", 1)[1] if u.startswith("data:") else u)
        return out or None

    images_payload = _normalize(image_data_urls)
    err = None

    if OLLAMA_URL:
        try:
            payload = {"model": OLLAMA_MODEL, "prompt": final_prompt, "stream": False}
            if images_payload:
                payload["images"] = images_payload
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
            r.raise_for_status()
            return {"provider": "ollama", "raw": r.json().get("response", "")}
        except Exception as e:
            err = f"Ollama error: {e}"

    if LLAVA_URL:
        try:
            payload = {"prompt": final_prompt}
            if images_payload:
                payload["image"] = images_payload[0]
            r = requests.post(LLAVA_URL, json=payload, timeout=120)
            r.raise_for_status()
            try:
                js = r.json()
                if isinstance(js, dict) and "response" in js:
                    return {"provider": "llava_url", "raw": js["response"]}
                return {"provider": "llava_url", "raw": json.dumps(js)}
            except Exception:
                return {"provider": "llava_url", "raw": r.text}
        except Exception as e:
            err = (err or "") + f" ; LLAVA_URL error: {e}"

    raise RuntimeError(err or "No VLM endpoint configured")

# ---------- robust request parsing for /apply ----------
def _extract_json_loose(text: str):
    """Try strict JSON, else extract the last {...} or [...] block, else literal_eval."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])\s*$', text.strip())
        if m:
            block = m.group(1)
            # try JSON then Python literal
            try:
                return json.loads(block)
            except Exception:
                try:
                    return ast.literal_eval(block)
                except Exception:
                    return None
    return None


def _parse_apply_request():
    """
    Accepts:
      - application/json  (with optional {"excerpt": "...", "changes"/"actions": [...]})
      - multipart/form-data fields: json=..., excerpt=...
      - raw text that has JSON + trailing human excerpt
      - VLM wrapper: {"response":{"raw": "...", "json": <obj>}}
    Returns: (changes_list, excerpt_str)
    """
    excerpt = None

    # 1) Try normal JSON body
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        # common wrappers
        if "response" in data and isinstance(data["response"], dict):
            inner = data["response"].get("json")
            if inner is not None:
                data = inner
        else:
            excerpt = data.get("excerpt") or data.get("summary") or excerpt

        payload = data.get("changes") or data.get("actions") or data
        return _coerce_changes(payload), excerpt

    if isinstance(data, list):
        return _coerce_changes(data), None

    # 2) Try form fields (multipart)
    if request.form:
        raw_json = (
            request.form.get("json")
            or request.form.get("changes")
            or request.form.get("actions")
            or ""
        )
        excerpt = request.form.get("excerpt") or request.form.get("summary") or None
        payload = _extract_json_loose(raw_json)
        return _coerce_changes(payload), excerpt

    # 3) Fallback: raw text with JSON + trailing prose
    raw_text = request.get_data(as_text=True) or ""
    payload = _extract_json_loose(raw_text)
    return _coerce_changes(payload), excerpt


# put near your other helpers
def _cad_state_json():
    return {
        "current_params": _snapshot(),          # live numeric params
        "context": CONTEXT,                     # e.g. {"terrain_mode": "flat"}
        "known_classes": sorted(list(COMPONENT_REGISTRY.keys())),
        "selected_parts": list(STATE.get("selected_parts", [])),
        "history": HISTORY[:H_PTR+1],           # applied snapshots up to current
        "pending_adds": list(PENDING_ADDS),     # queued 'add' ops not yet consumed
        # (Optional) Introspected class defaults for extra grounding:
        # "introspected": {
        #     "wheel": _introspect_params_from_cls(_ThisWheel),
        #     "pan_tilt": _introspect_params_from_cls(_PanTilt),
        # }
    }



# ----------------- routes: page + UI helpers -----------------
@app.get("/")
def index():
    return render_template("viewer.html")


@app.get("/mode")
def mode():
    mode = (
        "GLB: assets/rover.glb"
        if os.path.exists(ROVER_GLB_PATH)
        else ("cqparts" if USE_CQPARTS else "fallback")
    )
    return jsonify({"mode": mode})


@app.post("/label")
def label():
    data = request.get_json(force=True, silent=True) or {}
    part = (data.get("part_name") or "").strip()
    if part:
        STATE["selected_parts"].append(part)
        return jsonify(
            {"ok": True, "part": part, "count": len(STATE["selected_parts"])}
        )
    return jsonify({"ok": False, "error": "no part_name"})


@app.get("/labels")
def labels():
    return jsonify({"ok": True, "selected_parts": STATE["selected_parts"]})

def _find_last_balanced_json_block(text: str):
    """
    Return (start_idx, end_idx_inclusive) of the LAST balanced {...} or [...]
    block in `text`, respecting JSON strings and escapes. If none, return (None, None).
    """
    if not text:
        return None, None

    in_string = False
    escape = False
    stack = []          # holds opening chars '{' or '['
    start_idx = None
    last_segment = (None, None)

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            # ignore everything else while inside a string
            continue

        # not in string:
        if ch == '"':
            in_string = True
            continue

        if ch == '{' or ch == '[':
            stack.append(ch)
            if len(stack) == 1:
                start_idx = i
        elif ch == '}' or ch == ']':
            if stack:
                opening = stack[-1]
                if (opening == '{' and ch == '}') or (opening == '[' and ch == ']'):
                    stack.pop()
                    if not stack and start_idx is not None:
                        # we just closed a top-level JSON block
                        last_segment = (start_idx, i)
                        start_idx = None
                else:
                    # mismatched closers reset (be forgiving)
                    stack.clear()
                    start_idx = None

    return last_segment


def _split_json_and_summary(raw_text: str):
    """
    Expected model output:
      <JSON (object or array)>
      
      SUMMARY: <one concise line>

    Returns: (parsed_json, summary_str_or_None, json_block_text_or_None)
    """
    if not raw_text:
        return None, None, None

    text = raw_text.strip()
    s, e = _find_last_balanced_json_block(text)
    if s is None or e is None:
        # No JSON found; try to pull a SUMMARY line anyway
        summary = None
        for ln in text.splitlines():
            sline = ln.strip()
            if sline.upper().startswith("SUMMARY:"):
                summary = sline[len("SUMMARY:"):].strip()
                break
        return None, summary, None

    json_block = text[s:e+1]
    tail = text[e+1:].strip()

    # Extract SUMMARY: the first non-empty line starting with it
    summary = None
    if tail:
        for ln in tail.splitlines():
            sline = ln.strip()
            if not sline:
                continue
            if sline.upper().startswith("SUMMARY:"):
                summary = sline[len("SUMMARY:"):].strip()
                break

    # Try to parse the json_block robustly
    parsed = None
    try:
        parsed = json.loads(json_block)
    except Exception:
        try:
            # Python-literal tolerance (single quotes etc.)
            parsed = ast.literal_eval(json_block)
        except Exception:
            try:
                # Heuristic repair: single→double, booleans/None
                fixed = (json_block
                         .replace("'", '"')
                         .replace(" None", " null")
                         .replace(": None", ": null")
                         .replace(" True", " true").replace(": True", ": true")
                         .replace(" False", " false").replace(": False", ": false"))
                parsed = json.loads(fixed)
            except Exception as e:
                print("[recommend][parser] could not parse JSON:", e)
                print("[recommend][parser] json_block candidate:\n", json_block)
                parsed = None

    return parsed, summary, json_block



# ----------------- VLM endpoints -----------------
@app.post("/recommend")
def recommend():
    try:
        prompt_text = (request.form.get("prompt") or "").strip()
        classes = json.loads(request.form.get("classes") or "[]")
        if not isinstance(classes, list):
            classes = []

        # Images
        ref_url = _data_url_from_upload(request.files.get("reference"))
        if not ref_url:
            return jsonify({"ok": False, "error": "no reference image"}), 400
        snapshot_url = _data_url_from_upload(request.files.get("snapshot"))

        # Build grounding with explicit comparison to CAD state/sequence
        cad_state = _cad_state_json()

        grounding_lines = [
            "Goal: Compare the REFERENCE image (photo/render) to the CURRENT CAD and propose precise, conservative changes that align CAD to the image.",
            "",
            "You are given:",
            "1) REFERENCE image (index 0).",
        ]
        if snapshot_url:
            grounding_lines.append("2) CURRENT CAD SNAPSHOT image (index 1).")
        grounding_lines += [
            "3) CURRENT CAD STATE JSON (below) which includes: parameters, context, known classes, selection history, applied history, and pending adds.",
            "",
            "CURRENT CAD STATE JSON:",
            json.dumps(cad_state, indent=2),
            "",
            "Known classes (from client):",
            *[f"- {c}" for c in classes],
            "",
            "Instructions:",
            "- Assume the CAD snapshot represents the model produced by the state JSON.",
            "- Identify mismatches between the reference image and the current CAD (sizes, positions, counts, orientations, presence/absence).",
            "- Propose changes conservatively; when unsure, prefer small deltas and include a brief 'rationale' and 'confidence'.",
            "",
            "Schema for each change object:",
            """{
  "target_component": "<class_or_type>",
  "action": "<modify|replace|resize|rotate|translate|delete|add>",
  "parameters": { "...": ... },
  "rationale": "one brief sentence",
  "title": "optional short label for UI (<= 8 words)",
  "confidence": 0.0
}""",
            "",
            "'add' rules:",
            "- When action = \"add\", target_component is the new part type (e.g., \"sensor_fork\", \"wheel\").",
            "- For wheels, support either \"count\" or \"wheels_per_side\".",
            "- Parameters may include:",
            "  - \"count\" or \"wheels_per_side\"",
            "  - \"position_mm\": [x, y, z]",
            "  - \"orientation_deg\": [rx, ry, rz]",
            "  - other dimensional attributes (e.g., width_mm, height_mm, etc.)",
            "- If a numeric value is uncertain, you may set it to null and explain briefly in 'rationale'.",
            "",
            "OUTPUT RULES:",
            "1) First output STRICT JSON: either a list of change objects or a single object as defined (no code fences, no preamble).",
            "2) Then output one blank line.",
            "3) Then output a single line beginning with 'SUMMARY: ' that briefly explains the proposed changes for end users.",
        ]

        if prompt_text:
            grounding_lines += ["", "User prompt:", prompt_text]

        images = [ref_url, snapshot_url] if snapshot_url else [ref_url]
        final_prompt = f"{VLM_SYSTEM_PROMPT}\n\n---\n" + "\n".join(grounding_lines)

        provider_out = call_vlm(final_prompt, images)
        raw = provider_out.get("raw", "")
        # --- DEBUG LOGGING ---
        print("\n" + "=" * 80)
        print("[recommend] Raw VLM output:\n")
        print(raw)
        print("=" * 80 + "\n")

        parsed, summary, json_block = _split_json_and_summary(raw)

        print("[recommend] Parsed JSON:", parsed)
        print("[recommend] Extracted summary:", summary)
        if json_block and not parsed:
            print("[recommend] JSON block candidate (unparsed):\n", json_block)

        return jsonify({
            "ok": True,
            "response": {
                "raw": raw,           # full text (JSON + SUMMARY)
                "json": parsed,       # parsed changes (object or list)
                "summary": summary,   # user-facing summary string (or None)
                "json_block": json_block  # exact JSON text block we parsed (debug aid)
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500



@app.post("/vlm")
def vlm():
    try:
        prompt_text = (request.form.get("prompt") or "").strip()
        selected = (request.form.get("selected_class") or "").strip() or None
        classes = json.loads(request.form.get("classes") or "[]")
        if not isinstance(classes, list):
            classes = []
        data_url = _data_url_from_upload(request.files.get("image"))

        grounding = ["Known component classes:", *[f"- {c}" for c in classes]]
        if selected:
            grounding.append(f"\nUser highlighted class: {selected}")
        grounding.append("\nUser prompt:\n" + prompt_text)

        final_prompt = f"{VLM_SYSTEM_PROMPT}\n\n---\n" + "\n".join(grounding)
        resp = call_vlm(final_prompt, data_url)
        raw = resp.get("raw", "")

        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}\s*$", raw.strip())
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

        return jsonify(
            {
                "ok": True,
                "provider": resp.get("provider"),
                "response": {"raw": raw, "json": parsed},
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ----------------- param introspection (unchanged) -----------------
def _introspect_params_from_cls(cls) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            val = getattr(cls, name)
            mod = getattr(getattr(val, "__class__", object), "__module__", "")
            if "cqparts.params" in mod:
                d[name] = str(val)
        except Exception:
            pass
    for name in dir(cls):
        if name in d or name.startswith("_"):
            continue
        try:
            val = getattr(cls, name)
            if isinstance(val, (int, float)):
                d[name] = val
        except Exception:
            pass
    return d


@app.get("/params")
def get_params():
    info = {"current": _snapshot(), "introspected": {}}
    if USE_CQPARTS:
        try:
            info["introspected"]["wheel"] = _introspect_params_from_cls(_ThisWheel)
            info["introspected"]["pan_tilt"] = _introspect_params_from_cls(_PanTilt)
        except Exception:
            pass
    info["context"] = {"terrain_mode": CONTEXT["terrain_mode"]}
    return jsonify({"ok": True, "params": info})


# ----------------- APPLY (key path) -----------------
def _coerce_changes(payload: Any) -> List[dict]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        if "response" in payload and isinstance(payload["response"], dict):
            payload = payload["response"].get("json") or payload
        if "actions" in payload or "changes" in payload:
            payload = payload.get("actions") or payload.get("changes")
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    return []


@app.post("/apply")
def apply_change():

    try:
        changes, excerpt = _parse_apply_request()

        if not changes:
            return jsonify({"ok": False, "error": "No change objects supplied"}), 400

        _push_history()
        rv = Rover(
            stepper=_Stepper,
            electronics=_Electronics,
            sensors=_PanTilt,
            wheel=_ThisWheel,
        )

        highlight_key = None

        for change in changes:
            action = (change.get("action") or "").strip().lower()
            target = (change.get("target_component") or "").strip().lower()
            params = change.get("parameters") or {}
            if "terrain_mode" in params:
                CONTEXT["terrain_mode"] = str(params["terrain_mode"]).lower()

            comp = get_component_spec(target) or get_component_spec(target.split()[0])

            # Normalize wheel add via count→wheels_per_side
            if action == "add" and (
                "wheel" in target or comp is get_component_spec("wheel")
            ):
                cnt = params.get("count")
                if params.get("wheels_per_side") is None and cnt is not None:
                    try:
                        params["wheels_per_side"] = max(1, (int(cnt) + 1) // 2)
                    except Exception:
                        pass

            # Map incoming param names → class attributes
            if comp and action in (
                "modify",
                "resize",
                "replace",
                "translate",
                "rotate",
                "add",
            ):
                for jkey, attr in (comp.param_map or {}).items():
                    if jkey in params and params[jkey] is not None:
                        try:
                            setattr(comp.cls, attr, float(params[jkey]))
                        except Exception:
                            pass

            # True add: queue geometry creation
            if action == "add" and comp and callable(comp.add_fn):
                comp.add_fn(adapter=ADAPTER, **params)

            # Model-level params
            apply_params_to_rover(rv, params)

            if not highlight_key:
                highlight_key = target

        _rebuild_and_save_glb()
        return jsonify({"ok": True, "highlight_key": highlight_key or "wheel", "excerpt": excerpt})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/undo")
def undo_change():
    global H_PTR
    if H_PTR <= 0:
        return jsonify({"ok": False, "error": "Nothing to undo"}), 400
    H_PTR -= 1
    _restore(HISTORY[H_PTR])
    try:
        _rebuild_and_save_glb()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/redo")
def redo_change():
    global H_PTR
    if H_PTR >= len(HISTORY) - 1:
        return jsonify({"ok": False, "error": "Nothing to redo"}), 400
    H_PTR += 1
    _restore(HISTORY[H_PTR])
    try:
        _rebuild_and_save_glb()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ----------------- Build & GLB export -----------------
def _rebuild_and_save_glb():
    setattr(Rover, "_pending_adds", list(PENDING_ADDS))  # expose queue for build
    glb = build_rover_scene_glb_cqparts()
    with open(ROVER_GLB_PATH, "wb") as f:
        f.write(glb)


def patch_cqparts_brittleness():
    from cqparts.utils.geometry import CoordSystem

    # import your classes
    from electronics import OtherBatt, OtherController, MotorController, type1
    from robot_base import Rover

    roots = (Rover, type1, OtherController, MotorController, OtherBatt)

    # 1) Ensure world coords exist (silences the warning and stabilizes solve)
    for cls in roots:
        try:
            if not hasattr(cls, "world_coords"):
                cls.world_coords = CoordSystem()
        except Exception:
            pass

    # 2) Make constraints no-ops for fragile assemblies (dev mode)
    def _no_constraints(self):
        return []

    for cls in (OtherController, MotorController, OtherBatt, type1, Rover):
        try:
            cls.make_constraints = _no_constraints
        except Exception:
            pass

    # 3) Guarantee geometry on parts that sometimes fail to generate
    import cadquery as cq

    def _ob_box(self, x=60, y=30, z=15):
        self.local_obj = cq.Workplane("XY").box(x, y, z)
        return {}

    def _oc_box(self, x=65, y=55, z=12):
        self.local_obj = cq.Workplane("XY").box(x, y, z)
        return {}

    try:
        OtherBatt.make_components = _ob_box
    except Exception:
        pass
    try:
        OtherController.make_components = _oc_box
    except Exception:
        pass


def constraints():
    # ---- PATCH: disable constraints & guarantee geometry on brittle parts ----
    from cqparts.utils.geometry import CoordSystem
    from electronics import OtherBatt as _OtherBatt
    from controller import OtherController as _OtherController
    from electronics import type1 as _Type1
    from robot_base import Rover as _Rover

    # 1) Ensure world coords exist on roots to silence solver warnings
    for cls in (_Rover, _Type1, _OtherController, _OtherBatt):
        try:
            if not hasattr(cls, "world_coords"):
                cls.world_coords = CoordSystem()  # Z-up default
        except Exception:
            pass

    # 2) NOP out constraints to avoid failing solves
    def _no_constraints(self):
        return []

    for cls in (_Rover, _Type1, _OtherController, _OtherBatt):
        try:
            cls.make_constraints = _no_constraints
        except Exception:
            pass

    # 3) Provide guaranteed geometry for battery (avoids 'fV' bug)
    def _otherbatt_make_components(self):
        self.local_obj = cq.Workplane("XY").box(60, 30, 15)
        return {}

    _OtherBatt.make_components = _otherbatt_make_components

    # 4) (Optional) Provide trivial geometry for controller if it lacks one
    try:

        def _ctrl_make_components(self):
            self.local_obj = cq.Workplane("XY").box(65, 55, 12)
            return {}

        _OtherController.make_components = _ctrl_make_components
    except Exception:
        pass


# -------------------------------------------------------------------------


def build_rover_scene_glb_cqparts() -> bytes:
    print("Generating GLB via cqparts…")
    rv = Rover(
        stepper=_Stepper, electronics=_Electronics, sensors=_PanTilt, wheel=_ThisWheel
    )

    # Ensure required attributes exist
    for name, cls in (
        ("stepper", _Stepper),
        ("electronics", _Electronics),
        ("sensors", _PanTilt),
        ("wheel", _ThisWheel),
    ):
        if not getattr(rv, name, None):
            setattr(rv, name, cls)

    # Build in a thread (some cqparts builds can stall)
    built = False
    build_err = [None]

    def _run_build():
        try:
            patch_cqparts_brittleness()
            rv.build()
        except Exception as e:
            build_err[0] = e

    t = threading.Thread(target=_run_build, daemon=True)
    t.start()
    t.join(25.0)
    if not t.is_alive() and build_err[0] is None:
        built = True
        print("Assembly build OK.")
    else:
        print("[warn] build timed out or errored:", build_err[0])

    # Convert CQ → Trimesh Scene
    scene = trimesh.Scene()

    def _cq_to_trimesh(obj, tol=0.6):
        from cadquery import exporters

        try:
            stl_txt = exporters.toString(obj, "STL", tolerance=tol).encode("utf-8")
            m = trimesh.load(io.BytesIO(stl_txt), file_type="stl")
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(tuple(m.geometry.values()))
            return m
        except Exception as e:
            print("[mesh] STL export failed:", e)
            return None

    def _get_shape(component):
        for attr in (
            "world_obj",
            "toCompound",
            "obj",
            "to_cadquery",
            "shape",
            "local_obj",
            "make",
        ):
            if hasattr(component, attr):
                try:
                    v = getattr(component, attr)
                    shp = v() if callable(v) else v
                    if shp is not None:
                        return shp
                except Exception as e:
                    print(
                        f"[get_shape] {component.__class__.__name__}.{attr} failed:", e
                    )
        return None

    def _iter_components(root):
        comps = getattr(root, "components", None)
        if isinstance(comps, dict):
            return comps.items()
        if comps:
            try:
                return list(comps)
            except Exception:
                pass
        return []

    def _walk(node, prefix=""):
        shp = _get_shape(node)
        if shp is not None:
            tm = _cq_to_trimesh(shp, tol=0.6)
            if tm and not getattr(tm, "is_empty", False):
                nm = prefix.rstrip("/") or node.__class__.__name__
                try:
                    scene.add_geometry(tm, node_name=nm)
                except Exception as e:
                    print(f"[scene] add {nm} failed:", e)
        for child_name, child in _iter_components(node):
            _walk(child, f"{prefix}{child_name}/")

    whole = None
    for attr in ("world_obj", "toCompound", "obj", "to_cadquery"):
        if hasattr(rv, attr):
            try:
                v = getattr(rv, attr)
                whole = v() if callable(v) else v
                if whole is not None:
                    break
            except Exception as e:
                print(f"[asm] rv.{attr} failed:", e)

    if whole is not None:
        mesh = _cq_to_trimesh(whole, tol=0.6)
        if mesh and not getattr(mesh, "is_empty", False):
            scene.add_geometry(mesh, node_name="Rover")
        else:
            _walk(rv, "")
    else:
        _walk(rv, "")

    # ---- consume queued adds: make real geometry and place it ----
    def _add_sensor_fork(params: Dict[str, Any]):
        # make the part
        part = SensorFork(
            width=float(params.get("width_mm", 40) or 40.0),
            depth=float(params.get("depth_mm", 25) or 25.0),
            height=float(params.get("height_mm", 30) or 30.0),
            wall=float(params.get("wall_mm", 3) or 3.0),
            hole_diam=float(params.get("hole_diam_mm", 3.2) or 3.2),
        )
        # default placement: front upper deck; terrain mode pitches slightly up
        pos = params.get("position_mm") or [220.0, 0.0, 160.0]
        rx, ry, rz = params.get("orientation_deg") or [0.0, 0.0, 0.0]

        if CONTEXT["terrain_mode"] == "uneven":
            rx = rx or 6.0  # pitch up to avoid ground strikes
            pos = [pos[0], pos[1], pos[2] + 10.0]  # small clearance bump

        # build CQ object and transform
        shp = part.make() if hasattr(part, "make") else getattr(part, "shape", None)
        if shp is None:
            return

        # rotation then translation (degrees)
        w = shp.rotate((0, 0, 0), (1, 0, 0), float(rx or 0.0))
        w = w.rotate((0, 0, 0), (0, 1, 0), float(ry or 0.0))
        w = w.rotate((0, 0, 0), (0, 0, 1), float(rz or 0.0))
        w = w.translate(tuple(float(x) for x in pos))

        tm = _cq_to_trimesh(w, tol=0.5)
        if tm and not getattr(tm, "is_empty", False):
            scene.add_geometry(tm, node_name="sensor_fork")

    for op in list(PENDING_ADDS):
        kind = (op.get("kind") or "").lower()
        params = op.get("params") or {}
        if kind == "sensor_fork":
            _add_sensor_fork(params)
        elif kind == "wheel":
            # wheels handled via class params (already applied)
            pass
        else:
            print(f"[adds] unknown kind '{kind}', skipping")

    # Export GLB
    if not scene.geometry:
        raise RuntimeError("No geometry exported")
    glb_bytes = scene.export(file_type="glb")
    return glb_bytes


# ----------------- GLB route & static -----------------
def build_rover_scene_glb(_: Optional[Dict[str, Any]] = None) -> bytes:
    if USE_CQPARTS:
        return build_rover_scene_glb_cqparts()
    if os.path.exists(ROVER_GLB_PATH):
        with open(ROVER_GLB_PATH, "rb") as f:
            return f.read()
    raise FileNotFoundError("cqparts disabled, and assets/rover.glb not found")


@app.get("/model.glb")
def model_glb():
    try:
        glb = build_rover_scene_glb({})
        return send_file(io.BytesIO(glb), mimetype="model/gltf-binary")
    except Exception as e:
        import traceback

        return Response(
            "model.glb build failed:\n" + traceback.format_exc(),
            status=500,
            mimetype="text/plain",
        )


@app.route("/static/<path:filename>")
def custom_static(filename):
    root = app.static_folder
    full = os.path.join(root, filename)
    if not os.path.exists(full):
        abort(404)
    if filename.endswith(".js"):
        return send_from_directory(root, filename, mimetype="application/javascript")
    return send_from_directory(root, filename)


# ----------------- warm build & main -----------------
def _warm_build():
    try:
        print("[warm] starting initial build…")
        glb = build_rover_scene_glb({})
        with open(ROVER_GLB_PATH, "wb") as f:
            f.write(glb)
        print("[warm] initial GLB ready.")
    except Exception as e:
        print("[warm] initial build failed:", e)


if __name__ == "__main__":
    os.makedirs(ASSETS_DIR, exist_ok=True)
    threading.Thread(target=_warm_build, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
