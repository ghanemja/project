#!/usr/bin/env python3
# viewer_bucket_like.py  (exact-geometry via GLB, with optional cqparts fallback)

import io
import os
import sys
import mimetypes
from typing import Dict, Any, Optional
import json, time, pathlib, requests, re, threading, base64

# ---- VLM CONFIG ----
# Falls back to OLLAMA_HOST if set. You can also set LLAVA_URL to a custom server.
OLLAMA_URL = os.environ.get("OLLAMA_URL", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava-llama3:latest")
LLAVA_URL = os.environ.get("LLAVA_URL")  # optional, e.g. "http://localhost:8000/infer"
VLM_SYSTEM_PROMPT = """You are a visual large language model assisting with 3D CAD model editing.

Context:
- The model is a parametric robot rover built in CadQuery / cqparts.
- The UI provides a list of component classes and the optional selected class.
- The user may provide a reference image; if present, propose changes that make the CAD better match that image (be conservative, but concrete).

Response format (strict JSON only). You may return a SINGLE change object, or a LIST of change objects:
{
  "target_component": "<component_class_or_specific_name_or_new_type>",
  "action": "<modify|replace|resize|rotate|translate|delete|add>",
  "parameters": { "field": "value", "field2": "value2" },
  "rationale": "one brief sentence"
}

Rules for 'add':
- Use "target_component": the class/type to add, e.g., "wheel".
- Put necessary fields in "parameters" such as:
  - "count": integer (e.g., 2 to add one wheel on each side, or 6 total),
  - "wheels_per_side": integer (e.g., 3),
  - "positions_mm": optional array of X/Y/Z triplets (mm) if you want explicit placements,
  - or "axle_count": integer plus "axle_spacing_mm".
- If unsure about exact values, set them to null but keep the key so the UI can prompt later.

General Rules:
- Output STRICT JSON (no prose).
- If you return multiple proposals, respond with a JSON array of objects.
- If unsure, set ambiguous fields to null and explain briefly in 'rationale'.
"""



# --------------------------- CQ v1 → v2 SHIMS ---------------------------
import cadquery as cq
try:
    from cqparts.utils.geometry import CoordSystem
except Exception:
    class CoordSystem:
        def __sub__(self, other): return self
        def __rsub__(self, other): return other

def _world_cs(self): return CoordSystem()
cq.Workplane.world_coords = property(_world_cs)
cq.Workplane.local_coords  = property(_world_cs)

def _wp_cut_out(self, other=None, *_, **__):
    if other is None: return self
    items = other if isinstance(other, (list, tuple)) else [other]
    res = self
    for it in items:
        if hasattr(it, "val"):
            try: it = it.val()
            except Exception: pass
        res = res.cut(it)
    return res
cq.Workplane.cut_out = _wp_cut_out

# --------------------------- LOCAL PKG PATH ---------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "cqparts_bucket"))

# --------------------------- IMPORTS ---------------------------
import trimesh
from flask import Flask, Response, send_file, request, jsonify, send_from_directory, abort

# ---- Rover + deps (top-level imports to match your repo layout) ----
from robot_base import Rover
from electronics import type1 as _Electronics
from pan_tilt import PanTilt as _PanTilt
from wheel import BuiltWheel as _ThisWheel
from cqparts_motors.stepper import Stepper as _Stepper  # only for signature compatibility

# Ensure correct MIME for ESM
mimetypes.add_type('application/javascript', '.js')

# --------------------------- CONFIG ---------------------------
PORT = int(os.environ.get("PORT", "5160"))
ROVER_GLB_PATH = os.path.join(os.path.dirname(__file__), "assets", "rover.glb")
USE_CQPARTS = os.environ.get("USE_CQPARTS", "1") == "1"

# --------------------------- PARAMS & HISTORY ---------------------------
CURRENT_PARAMS: Dict[str, Optional[float]] = {
    "wheel_diameter": None,
    "wheel_width": None,
    "pan_tilt_offset_x": None,
    "pan_tilt_offset_y": None,
    "pan_tilt_offset_z": None,
    "wheels_per_side": None,       # e.g., 3  (=> total wheels = 2 * wheels_per_side)
    "axle_spacing_mm": None,       # spacing between wheel centers along X or Y
    "wheelbase_span_mm": None,   
}
HISTORY: list[Dict[str, Optional[float]]] = []
H_PTR: int = -1  # -1 means empty

def _snapshot() -> Dict[str, Optional[float]]:
    return {k: (float(v) if v is not None else None) for k, v in CURRENT_PARAMS.items()}

def _push_history():
    global H_PTR, HISTORY
    if H_PTR < len(HISTORY) - 1:  # truncate redo branch
        HISTORY = HISTORY[:H_PTR+1]
    HISTORY.append(_snapshot())
    H_PTR = len(HISTORY) - 1

def _restore(snapshot: Dict[str, Optional[float]]):
    for k in CURRENT_PARAMS.keys():
        CURRENT_PARAMS[k] = snapshot.get(k, CURRENT_PARAMS[k])

def _clean_num(v):
    try:
        if v is None: return None
        if isinstance(v, (int, float)): return float(v)
        s = str(v).strip()
        if s == "": return None
        return float(s)
    except Exception:
        return None

def apply_params_to_rover(rv, params: Dict[str, Any] | None):
    """Map CURRENT_PARAMS (updated by params) to your classes/parts as needed."""
    if params:
        for k, v in params.items():
            if k in CURRENT_PARAMS:
                CURRENT_PARAMS[k] = _clean_num(v)

    # Example mappings (adjust to your parts’ actual attributes)
    try:
        if CURRENT_PARAMS["wheel_diameter"] is not None:
            setattr(_ThisWheel, "diameter", float(CURRENT_PARAMS["wheel_diameter"]))
    except Exception:
        pass

    try:
        if CURRENT_PARAMS["wheel_width"] is not None:
            setattr(_ThisWheel, "width", float(CURRENT_PARAMS["wheel_width"]))
    except Exception:
        pass

    # Wheels per side (adds wheels)
    try:
        if CURRENT_PARAMS["wheels_per_side"] is not None:
            setattr(Rover, "wheels_per_side", int(CURRENT_PARAMS["wheels_per_side"]))
    except Exception:
        pass

    # Spacing controls
    for k in ("axle_spacing_mm","wheelbase_span_mm"):
        try:
            if CURRENT_PARAMS[k] is not None:
                setattr(Rover, k, float(CURRENT_PARAMS[k]))
        except Exception:
            pass


    try:
        for axis in ("x", "y", "z"):
            key = f"pan_tilt_offset_{axis}"
            if CURRENT_PARAMS[key] is not None:
                setattr(_PanTilt, key, float(CURRENT_PARAMS[key]))
    except Exception:
        pass

# --------------------------- HTML ---------------------------
HTML = """
<!doctype html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Bucket-like CAD Viewer</title>
<style>
  :root{--bar:54px;--sidebar:320px;--right:380px;--console:150px}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  /* Collapse whole panels by zeroing the CSS vars already used by layout */
  body.left-collapsed  { --sidebar: 0px; }
  body.right-collapsed { --right:   0px; }

  /* Optional: remove borders when hidden */
  body.left-collapsed  #left  { border-right: none; }
  body.right-collapsed #right { border-left:  none; }
  #bar{position:fixed;top:0;left:0;right:0;height:var(--bar);display:flex;gap:10px;align-items:center;padding:8px 12px;border-bottom:1px solid #eee;background:#fff;z-index:10}
  #left{position:fixed;top:var(--bar);left:0;bottom:var(--console);width:var(--sidebar);border-right:1px solid #eee;background:#fafafa;overflow:auto}
  #left h3,#right h3{margin:10px 12px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#666}
  #compList{list-style:none;margin:0;padding:6px 6px 12px}
  #compList li{display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:8px;margin:4px 6px;cursor:pointer}
  #compList li:hover{background:#f1f5f9}
  #compList li.active{outline:2px solid #0ea5e9;background:#e0f2fe}
  .swatch{width:14px;height:14px;border-radius:4px;border:1px solid #ddd;flex:none}
  .count{margin-left:auto;font-size:12px;color:#475569;background:#e2e8f0;border-radius:999px;padding:1px 6px}
  #wrap{position:fixed;top:var(--bar);left:var(--sidebar);right:var(--right);bottom:var(--console)}
  #canvas{width:100%;height:100%;display:block}
  #right{position:fixed;top:var(--bar);right:0;bottom:var(--console);width:var(--right);border-left:1px solid #eee;background:#fff;display:flex;flex-direction:column;overflow:auto}
  .section{padding:10px 12px;border-bottom:1px solid #f0f0f0}
  .section header{display:flex;align-items:center;gap:8px}
  .section header h3{margin:0;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#666}
  .section header .toggle{margin-left:auto;border:1px solid #ddd;background:#fff;border-radius:8px;padding:4px 8px;cursor:pointer;font-size:12px}
  .section header .toggle:hover{background:#f8fafc}
  .section.collapsed .section-body{display:none}
  .row{display:flex;align-items:center;gap:10px;margin:6px 0;flex-wrap:wrap}
  .row label{font-size:12px;color:#334155;min-width:90px}
  .row input[type=range]{flex:1 1 160px}
  .row input[type=number]{width:110px;padding:4px 6px;border:1px solid #e5e7eb;border-radius:6px}
  .row button,.btn{padding:7px 10px;border:1px solid #ddd;background:#fff;border-radius:8px;cursor:pointer}
  .row button:hover,.btn:hover{background:#f8fafc}
  .pill{font-size:11px;padding:2px 8px;border-radius:999px;background:#f3f4f6;color:#555}
  #name{font-weight:600;color:#333}
  #log{font-size:12px;color:#666;white-space:pre}
  #prompt{width:100%;min-height:90px;padding:8px;border:1px solid #e5e7eb;border-radius:8px;resize:vertical}
  .chip{display:inline-flex;align-items:center;gap:6px;font-size:12px;background:#eef2ff;color:#4338ca;border-radius:999px;padding:3px 8px;margin:3px 4px 0 0;cursor:pointer}
  /* Always show preview area even without image */
  #imgPreview{display:block; width:100%; min-height:120px; max-height:160px; border:1px dashed #cbd5e1;border-radius:8px;padding:6px;object-fit:contain;background:#fafafa}
  #tools{display:flex;gap:8px;margin-left:8px}
  #tools label{display:flex;align-items:center;gap:6px;font-size:12px;color:#444}
  #console{position:fixed;left:0;right:0;bottom:0;height:var(--console);border-top:1px solid #e5e7eb;background:#0b1020;color:#e5e7eb;font:12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;display:flex;flex-direction:column}
  #console header{padding:6px 10px;border-bottom:1px solid #1f2937;display:flex;align-items:center;gap:8px}
  #console .stream{flex:1;overflow:auto;padding:8px 10px}
  .logline{white-space:pre-wrap;margin:0 0 4px}
  .ok{color:#86efac}
  .warn{color:#facc15}
  .err{color:#fca5a5}
</style></head><body>
<div id="bar">
  <button id="reload">Reload Model</button>
  <button id="clear">Clear Selection</button>
  <button id="suggestFromImage" class="btn">Suggest from image</button>
  <span>Selected:</span> <span id="name">—</span>
  <span id="log" style="margin-left:12px"></span>
  <span id="tools" style="margin-left:auto">
      <!-- inside #bar, anywhere convenient -->
    <label><input type="checkbox" id="toggleLeftPanel" title="Show/Hide Camera panel" checked> Show Camera panel</label>
    <label><input type="checkbox" id="toggleRightPanel" title="Show/Hide Components panel" checked> Show Components panel</label>
    <label><input type="checkbox" id="toggleLabels" checked> Show labels</label>
    <button id="fitAll" class="btn">Fit All</button>
  </span>
  <span class="pill" id="modeHint" style="margin-left:8px"></span>
</div>

<!-- LEFT: Camera (moved here) -->
<div id="left">
  <div class="section" id="sectionCamera">
    <header>
      <h3>Camera</h3>
      <button class="toggle" data-target="sectionCamera">Collapse</button>
    </header>
    <div class="section-body">
      <div class="row"><label>FOV</label><input id="fov" type="range" min="20" max="90" value="50"><span id="fovVal" class="pill">50°</span></div>
      <div class="row"><label>Near</label><input id="near" type="number" value="0.01" step="0.01"><label>Far</label><input id="far" type="number" value="5000" step="10"></div>
      <div class="row"><label>Damping</label><input id="damping" type="range" min="0" max="100" value="8"><span id="dampVal" class="pill">0.08</span></div>
      <div class="row"><label>Rotate Spd</label><input id="rotSpd" type="range" min="10" max="300" value="100"><span id="rotVal" class="pill">1.00</span></div>
      <div class="row"><label>Zoom Spd</label><input id="zoomSpd" type="range" min="10" max="300" value="100"><span id="zoomVal" class="pill">1.00</span></div>
      <div class="row"><label>Pan Spd</label><input id="panSpd" type="range" min="10" max="300" value="100"><span id="panVal" class="pill">1.00</span></div>
      <div class="row">
        <button id="viewIso">Iso</button>
        <button id="viewTop">Top</button>
        <button id="viewFront">Front</button>
        <button id="viewRight">Right</button>
        <button id="rotLeft">⟲ 90°</button>
        <button id="rotRight">⟳ 90°</button>
        <button id="resetCam">Reset</button>
        <label style="margin-left:auto"><input type="checkbox" id="lockTarget"> Lock to selection</label>
      </div>
      <div class="row">
        <label><input type="checkbox" id="gridToggle" checked> Grid</label>
        <label><input type="checkbox" id="axesToggle" checked> Axes</label>
      </div>
    </div>
  </div>
</div>

<div id="wrap"><canvas id="canvas"></canvas></div>

<!-- RIGHT: Components (moved here) + VLM Prompt -->
<div id="right">
  <div class="section" id="sectionComponents">
    <header>
      <h3>Components</h3>
      <button class="toggle" data-target="sectionComponents">Collapse</button>
    </header>
    <div class="section-body">
      <ul id="compList"></ul>
    </div>
  </div>

  <div class="section" id="vlmSection">
    <header>
      <h3>VLM Prompt</h3>
      <button class="toggle" data-target="vlmSection">Collapse</button>
    </header>
    <div class="section-body">
      <div class="row" style="flex-direction:column; align-items:stretch">
        <textarea id="prompt" placeholder="Describe the change you want (e.g., 'Increase wheel diameter by 10% and move the pan-tilt 20mm forward')"></textarea>
      </div>
      <div class="row">
        <input id="imgFile" type="file" accept="image/*">
        <button id="clearImg">Clear Image</button>
      </div>
      <div class="row"><img id="imgPreview" alt="Reference preview (optional)"></div>
      <div class="row">
        <button id="insertSelected">Insert selected</button>
        <button id="sendVLM" style="margin-left:auto">Send to VLM</button>
      </div>
      <!-- chips live here now -->
      <div id="chips" style="margin-top:6px"></div>
      <div id="vlmNotice" style="font-size:12px;color:#64748b"></div>
    </div>
  </div>
</div>

<div id="console">
  <header>
    <strong>Console</strong>
    <button id="btnUndo">Undo</button>
    <button id="btnRedo">Redo</button>
    <span id="paramHint" style="margin-left:auto;opacity:.8"></span>
  </header>
  <div class="stream" id="consoleStream"></div>
</div>

<script type="module">
  import * as THREE from '/static/jsm/three.module.js';
  import { OrbitControls } from '/static/jsm/controls/OrbitControls.js';
  import { GLTFLoader } from '/static/jsm/loaders/GLTFLoader.js';

  // ---- Collapsible helpers ----
  function restoreCollapsedState(){
    document.querySelectorAll('.section').forEach(sec=>{
      const id = sec.id; if(!id) return; const collapsed = localStorage.getItem('sec:'+id)==='1';
      sec.classList.toggle('collapsed', collapsed);
      const btn = sec.querySelector('.toggle'); if(btn) btn.textContent = collapsed ? 'Expand' : 'Collapse';
    });
  }
  function setupToggles(){
    document.querySelectorAll('.section .toggle').forEach(btn=>{
      btn.addEventListener('click',()=>{
        const target = btn.dataset.target; const sec = document.getElementById(target);
        if(!sec) return; const now = !sec.classList.contains('collapsed');
        sec.classList.toggle('collapsed', now);
        btn.textContent = now ? 'Expand' : 'Collapse';
        localStorage.setItem('sec:'+target, now ? '1':'0');
      });
    });
    restoreCollapsedState();
  }

  const suggestBtn = document.getElementById('suggestFromImage');

  async function snapshotCanvasToBlob() {
    const canvas = document.getElementById('canvas');
    return await new Promise(res => canvas.toBlob(b => res(b), 'image/png', 0.9));
  }

  suggestBtn.onclick = async () => {
    try {
      const data = new FormData();
      if (imgFile.files?.[0]) data.append('reference', imgFile.files[0]);
      else { vlmNotice.textContent = 'Select a reference image first.'; vlmNotice.style.color='#b45309'; return; }

      const snapBlob = await snapshotCanvasToBlob();
      if (snapBlob) data.append('snapshot', new File([snapBlob], 'snapshot.png', { type: 'image/png' }));

      data.append('classes', JSON.stringify([...classMap.keys()]));
      data.append('prompt', promptEl.value || '');

      const r = await fetch('/recommend', { method:'POST', body:data });
      if(!r.ok) throw new Error('recommend HTTP '+r.status);
      const js = await r.json();
      const recs = js?.response?.json;

      if (!recs) { logLine('No structured suggestions returned.', 'warn'); return; }

      const changes = Array.isArray(recs) ? recs : [recs];
      for (const ch of changes) {
        await applyVLMJson(ch);
      }
    } catch(e){
      logLine(String(e), 'err');
    }
  };

  function sceneBasis(){
    const up = new THREE.Vector3(0,1,0);
    const camDir = new THREE.Vector3(); camera.getWorldDirection(camDir);
    const right = new THREE.Vector3().crossVectors(camDir, up).normalize();
    const upFace = new THREE.Vector3().crossVectors(right, camDir).normalize(); // camera-facing up
    return {right, up: upFace};
  }

  function project2(basis, v, origin){
    const p = v.clone().sub(origin);
    return { x: p.dot(basis.right), y: p.dot(basis.up) };
  }


  // --- tiny text label sprite helper ---
  // replace old makeTextSprite
  function makeTextSprite(text, {fontSize=128, pad=16, worldScale=0.5}={}){
    const cvs=document.createElement('canvas'); const ctx=cvs.getContext('2d');
    ctx.font = `${fontSize}px system-ui,-apple-system,Segoe UI,Roboto,sans-serif`;
    const w=Math.ceil(ctx.measureText(text).width)+pad*2, h=fontSize+pad*2;
    cvs.width=w*2; cvs.height=h*2;
    const g=cvs.getContext('2d'); g.scale(2,2);
    g.fillStyle='rgba(255,255,255,0.96)'; g.strokeStyle='rgba(0,0,0,0.18)';
    g.lineWidth=1.2; g.beginPath();
    const r=8; g.roundRect?g.roundRect(0,0,w,h,r):g.rect(0,0,w,h); g.fill(); g.stroke();
    g.fillStyle='#0f172a'; g.font=`${fontSize}px system-ui,-apple-system,Segoe UI,Roboto,sans-serif`;
    g.textBaseline='middle'; g.fillText(text,pad,h/2);

    const tex=new THREE.CanvasTexture(cvs); tex.needsUpdate=true;
    const mat=new THREE.SpriteMaterial({map:tex, depthTest:false});
    const spr=new THREE.Sprite(mat);
    spr.scale.set(w*worldScale,h*worldScale,1);
    spr.renderOrder=999;
    return spr;
  }
    function buildLabelCallout(key, center, bboxDiag){
    const group = new THREE.Group();

    // offset diagonally (up + camera-right) proportional to model size
    const up = new THREE.Vector3(0,1,0);
    const camDir = new THREE.Vector3(); camera.getWorldDirection(camDir);
    const right = new THREE.Vector3().crossVectors(camDir, up).normalize();
    const offLen = Math.max(0.8*bboxDiag, 12);     // world units
    const labelPos = center.clone()
      .add(up.clone().multiplyScalar(offLen*0.9))
      .add(right.clone().multiplyScalar(offLen*0.8));

    // sprite (bigger)
    const spr = makeTextSprite(key, {fontSize:50, worldScale:0.18});
    spr.position.copy(labelPos);
    group.add(spr);

    // leader line
    const geom = new THREE.BufferGeometry().setFromPoints([labelPos, center]);
    const line = new THREE.Line(geom, new THREE.LineBasicMaterial({color:0x0f172a}));
    group.add(line);

    // arrow head at the center pointing toward label
    const dir = labelPos.clone().sub(center).normalize();
    const len = Math.max(0.06*bboxDiag, 6);
    const arrow = new THREE.ArrowHelper(dir, center, len, 0x0f172a, /*headLength*/ len*0.55, /*headWidth*/ len*0.35);
    group.add(arrow);

    return {group, sprite:spr, line, arrow};
  }



  const canvas=document.getElementById('canvas');
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
  const scene=new THREE.Scene(); scene.background=new THREE.Color(0xf7f8fb);
  const camera=new THREE.PerspectiveCamera(50,2,0.01,5000); camera.position.set(150,110,180);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true; controls.dampingFactor=0.08;

  const hemi=new THREE.HemisphereLight(0xffffff,0x444444,1.1); scene.add(hemi);
  const dir=new THREE.DirectionalLight(0xffffff,1.2); dir.position.set(120,220,160); scene.add(dir);

  // ---- PIVOT: rotate model + grid + axes together
  const pivot = new THREE.Group(); scene.add(pivot);
  const grid=new THREE.GridHelper(1000,100); grid.position.y=0; pivot.add(grid);
  const axes=new THREE.AxesHelper(120); pivot.add(axes);

  // materials
  const defaultMat=new THREE.MeshStandardMaterial({color:0x9aa3af,metalness:0.0,roughness:0.9});
  const hoverEmissive=new THREE.Color(0x2b6cb0);
  const selectEmissive=new THREE.Color(0xd97706);

  // UI elements
  const rl=document.getElementById('reload'), cl=document.getElementById('clear');
  const nameEl=document.getElementById('name'), logEl=document.getElementById('log'), modeHint=document.getElementById('modeHint');
  const compList=document.getElementById('compList'), toggleLabels=document.getElementById('toggleLabels');
  const fitAllBtn=document.getElementById('fitAll');
  const fov=document.getElementById('fov'), fovVal=document.getElementById('fovVal');
  const near=document.getElementById('near'), far=document.getElementById('far');
  const damping=document.getElementById('damping'), dampVal=document.getElementById('dampVal');
  const rotSpd=document.getElementById('rotSpd'), rotVal=document.getElementById('rotVal');
  const zoomSpd=document.getElementById('zoomSpd'), zoomVal=document.getElementById('zoomVal');
  const panSpd=document.getElementById('panSpd'), panVal=document.getElementById('panVal');
  const viewIso=document.getElementById('viewIso'), viewTop=document.getElementById('viewTop');
  const viewFront=document.getElementById('viewFront'), viewRight=document.getElementById('viewRight'), resetCam=document.getElementById('resetCam');
  const lockTarget=document.getElementById('lockTarget'), gridToggle=document.getElementById('gridToggle'), axesToggle=document.getElementById('axesToggle');
  const rotLeft=document.getElementById('rotLeft'), rotRight=document.getElementById('rotRight');

  // console
  const stream = document.getElementById('consoleStream');
  const paramHint = document.getElementById('paramHint');
  const btnUndo = document.getElementById('btnUndo');
  const btnRedo = document.getElementById('btnRedo');

  // VLM panel
  const promptEl=document.getElementById('prompt');
  const chips=document.getElementById('chips');
  const imgFile=document.getElementById('imgFile'), imgPreview=document.getElementById('imgPreview'), clearImg=document.getElementById('clearImg');
  const insertSelected=document.getElementById('insertSelected'), sendVLM=document.getElementById('sendVLM'), vlmNotice=document.getElementById('vlmNotice');

  // state
  let group=null, baselineCam=null;
  const classMap=new Map(); // key -> { color:THREE.Color, nodes:Set<Object3D>, label:Sprite, count:number }
  let hovered=null, selectedClass=null;
  const origMats=new Map();

  // ---- Dynamic column sizing (fit content, within viewport limits)
  function adjustColumns(){
    const left = document.getElementById('left');
    const right = document.getElementById('right');
    const docStyle = document.documentElement.style;
    const maxRight = Math.min(Math.max(380, right.scrollWidth + 24), window.innerWidth - 360); // leave canvas visible
    const maxLeft  = Math.min(Math.max(320, left.scrollWidth + 24),  Math.floor(window.innerWidth * 0.45));
    docStyle.setProperty('--right', maxRight + 'px');
    docStyle.setProperty('--sidebar', maxLeft + 'px');
  }
  const resizeObserver = new ResizeObserver(()=>adjustColumns());
  resizeObserver.observe(document.body);
  window.addEventListener('resize', adjustColumns);

  function logLine(msg, kind='ok'){
    const p=document.createElement('div');
    p.className='logline '+kind;
    p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    stream.appendChild(p);
    stream.scrollTop = stream.scrollHeight;
  }

  async function refreshParamsHint(){
    try{
      const r = await fetch('/params'); const js = await r.json();
      if(js.ok){
        const cur = js.params?.current || {};
        const parts = Object.entries(cur).filter(([_,v])=>v!=null).map(([k,v])=>`${k}=${v}`);
        paramHint.textContent = parts.length ? parts.join('  ·  ') : 'No live params set';
      }
    }catch{}
  }

  function saveBaselineCam(){
    baselineCam={pos:camera.position.clone(), target:controls.target.clone(), fov:camera.fov, near:camera.near, far:camera.far};
  }
  function restoreBaselineCam(){
    if(!baselineCam) return;
    camera.position.copy(baselineCam.pos); controls.target.copy(baselineCam.target);
    camera.fov=baselineCam.fov; camera.near=baselineCam.near; camera.far=baselineCam.far; camera.updateProjectionMatrix();
    fov.value=String(Math.round(camera.fov)); fovVal.textContent=`${Math.round(camera.fov)}°`;
    near.value=String(camera.near); far.value=String(camera.far);
  }

  async function getMode(){ try{ const r=await fetch('/mode'); const js=await r.json(); modeHint.textContent=js.mode; }catch{ modeHint.textContent='mode: unknown'; } }
  function resize(){ const w=renderer.domElement.clientWidth, h=renderer.domElement.clientHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
  window.addEventListener('resize', resize);

  function fit(){
    const box=new THREE.Box3().setFromObject(pivot);
    const len=box.getSize(new THREE.Vector3()).length();
    const c=box.getCenter(new THREE.Vector3());
    camera.near=Math.max(0.01,len/200); camera.far=len*15; camera.updateProjectionMatrix();
    camera.position.copy(c).add(new THREE.Vector3(0.6*len,0.45*len,0.9*len));
    camera.lookAt(c); controls.target.copy(c);
  }

  function hashColor(str){ let h=5381; for(let i=0;i<str.length;i++) h=((h<<5)+h)+str.charCodeAt(i); const hue=((h>>>0)%360); const c=new THREE.Color(); c.setHSL(hue/360,0.56,0.56); return c; }
  function classKeyFromName(name){ if(!name) return 'Unnamed'; const seg=name.split('/')[0]; return seg.replace(/_\d+$/,''); }

  function setDefaultIfMissing(root){ root.traverse(o=>{ if(o.isMesh && !o.material) o.material=defaultMat.clone(); }); }

  function paintNode(o, baseColor, emissiveColor=null, opacity=1){
    o.traverse(n=>{
      if(!n.isMesh) return;
      if(!origMats.has(n)) origMats.set(n, n.material);
      const m=n.material.clone();
      m.transparent = opacity < 1.0; m.opacity=opacity;
      if(baseColor) m.color = baseColor.clone();
      if(m.emissive) m.emissive = (emissiveColor? emissiveColor.clone(): new THREE.Color(0x000000));
      n.material=m;
    });
  }
  function restoreNode(o){ o.traverse(n=>{ if(n.isMesh && origMats.has(n)){ n.material.dispose(); n.material=origMats.get(n); origMats.delete(n);} }); }

  function buildClassRegistry(root){
    classMap.clear();
    const seen = new Set();
    root.traverse(o=>{
      if(!o.isMesh) return;
      let p=o; while(p.parent && !p.name) p=p.parent;
      const key=classKeyFromName(p.name||o.name||'Unnamed');
      if (!classMap.has(key)) classMap.set(key,{color:hashColor(key),nodes:new Set(),label:null,count:0});
      if(!seen.has(p)){ classMap.get(key).nodes.add(p); classMap.get(key).count++; seen.add(p); }
    });
  }

  function placeLabels(){
    // remove old labels
    classMap.forEach(e => { if (e.label){ pivot.remove(e.label); e.label = null; } });
    if (!document.getElementById('toggleLabels').checked) return;

    // place one centered sprite per component class (no arrows/lines)
    classMap.forEach((entry, key) => {
      // compute this class' bounding box & center
      const box = new THREE.Box3();
      entry.nodes.forEach(n => box.union(new THREE.Box3().setFromObject(n)));
      const c = box.getCenter(new THREE.Vector3());
      const sz = box.getSize(new THREE.Vector3());
      const lift = Math.max(0.02*sz.y, 0.6);  // tiny vertical lift to avoid z-fighting

      // make a bigger text sprite and position at component center (+ small lift)
      const spr = makeTextSprite(key, { fontSize: 56, worldScale: 0.1, pad: 18 });
      spr.position.copy(c).add(new THREE.Vector3(0, lift, 0));

      pivot.add(spr);
      entry.label = spr;
    });
  }
  function colorizeByClass(){
    if(group) restoreNode(group);
    classMap.forEach(entry=> entry.nodes.forEach(node=> paintNode(node, entry.color, null, 1)));
  }

  function syncSidebar(){
    compList.innerHTML='';
    classMap.forEach((entry,key)=>{
      const li=document.createElement('li'); li.dataset.key=key; if(selectedClass===key) li.classList.add('active');
      const sw=document.createElement('span'); sw.className='swatch'; sw.style.backgroundColor='#'+entry.color.getHexString();
      const txt=document.createElement('span'); txt.textContent=key;
      const cnt=document.createElement('span'); cnt.className='count'; cnt.textContent=entry.count;
      li.appendChild(sw); li.appendChild(txt); li.appendChild(cnt);
      li.onclick=()=> selectClass(key,true);
      compList.appendChild(li);
    });
    // chips live near VLM prompt now
    chips.innerHTML='';
    classMap.forEach((_,key)=>{
      const c=document.createElement('span'); c.className='chip'; c.textContent=key; c.title='Insert into prompt';
      c.onclick=()=> insertText(` ${key} `);
      chips.appendChild(c);
    });
    adjustColumns();
  }

  function frameBox(box, {pad=1.25, duration=420} = {}){
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    // compute distance from FOV (vertical)
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const dist = (maxDim * pad) / (2 * Math.tan(fov / 2));

    // keep current azimuth/elevation: ray from center opposite current view dir
    const viewDir = new THREE.Vector3();
    camera.getWorldDirection(viewDir); // points from camera -> scene
    const targetPos = center.clone().sub(viewDir.clone().normalize().multiplyScalar(dist));

    // robust near/far
    camera.near = Math.max(0.01, maxDim/200);
    camera.far  = Math.max(camera.near+1, dist + maxDim*10);
    camera.updateProjectionMatrix();

    // animate cam+target
    const startPos = camera.position.clone();
    const startTgt = controls.target.clone();
    const endPos   = targetPos;
    const endTgt   = center.clone();
    const t0 = performance.now();

    function tick(now){
      const t = Math.min(1, (now - t0) / duration);
      const e = t<0.5 ? 2*t*t : -1+(4-2*t)*t; // easeInOutQuad
      camera.position.lerpVectors(startPos, endPos, e);
      controls.target.lerpVectors(startTgt, endTgt, e);
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }


  function selectClass(key, zoom=false){
    if(selectedClass && classMap.has(selectedClass)){
      const prev=classMap.get(selectedClass);
      prev.nodes.forEach(node=> paintNode(node, prev.color, null, 1));
    }
    selectedClass=key||null;
    if(selectedClass && classMap.has(selectedClass)){
      const entry=classMap.get(selectedClass);
      entry.nodes.forEach(node=> paintNode(node, entry.color, selectEmissive, 0.85));
      nameEl.textContent=selectedClass;
      if (zoom){
        const box=new THREE.Box3();
        entry.nodes.forEach(n=> box.union(new THREE.Box3().setFromObject(n)));
        frameBox(box, {pad:1.3, duration:450});   // center & zoom to part
      }
      [...compList.children].forEach(li=> li.classList.toggle('active', li.dataset.key===selectedClass));
    }else{
      nameEl.textContent='—';
      [...compList.children].forEach(li=> li.classList.remove('active'));
    }
  }

  const raycaster=new THREE.Raycaster(); const pointer=new THREE.Vector2();
  renderer.domElement.addEventListener('mousemove', e=>{
    const r=renderer.domElement.getBoundingClientRect();
    pointer.x=((e.clientX-r.left)/r.width)*2-1; pointer.y=-((e.clientY-r.top)/r.height)*2+1;
  });
  renderer.domElement.addEventListener('click', ()=>{
    if(!group) return;
    raycaster.setFromCamera(pointer,camera);
    const hits=raycaster.intersectObjects(group.children,true);
    if(!hits.length){ selectClass(null); return; }
    let obj=hits[0].object; while(obj.parent && !obj.name) obj=obj.parent;
    const key=classKeyFromName(obj.name); selectClass(key,true);
    fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({part_name:key})})
      .then(r=>r.json()).then(js=>{ logEl.textContent= js.ok ? `saved "${key}"` : (js.error||'save failed'); })
      .catch(e=> logEl.textContent=String(e));
  });

  function clearScene(){
    if(!group) return;
    classMap.forEach(e=>{ if(e.label){ pivot.remove(e.label); e.label=null; }});
    pivot.remove(group);
    group.traverse(o=>{ if(o.geometry) o.geometry.dispose(); if(o.material) o.material.dispose(); });
    group=null; hovered=null; selectClass(null); classMap.clear();
  }

  async function loadModel(){
    clearScene();
    const loader=new GLTFLoader(); const url='/model.glb?ts='+Date.now();
    logLine('Loading model…');
    await new Promise((res,rej)=> loader.load(url, g=>{
      group = g.scene;
      group.rotation.x = -Math.PI / 2;   // Z-up → Y-up
      setDefaultIfMissing(group);
      pivot.add(group);
      buildClassRegistry(group); colorizeByClass(); placeLabels(); syncSidebar(); fit(); saveBaselineCam(); res();
      logLine('Model loaded.');
    }, undefined, (err)=>{ logLine('GLTF load error: '+String(err),'err'); rej(err); }));
  }

  // Camera controls wiring
  fov.oninput=()=>{ camera.fov=+fov.value; camera.updateProjectionMatrix(); fovVal.textContent=`${fov.value}°`; };
  near.onchange=()=>{ camera.near=Math.max(0.001, +near.value); camera.updateProjectionMatrix(); };
  far.onchange=()=>{ camera.far=Math.max(camera.near+0.001, +far.value); camera.updateProjectionMatrix(); };
  damping.oninput=()=>{ controls.dampingFactor=(+damping.value)/100; dampVal.textContent=controls.dampingFactor.toFixed(2); };
  rotSpd.oninput=()=>{ controls.rotateSpeed=(+rotSpd.value)/100; rotVal.textContent=controls.rotateSpeed.toFixed(2); };
  zoomSpd.oninput=()=>{ controls.zoomSpeed=(+zoomSpd.value)/100; zoomVal.textContent=controls.zoomSpeed.toFixed(2); };
  panSpd.oninput=()=>{ controls.panSpeed=(+panSpd.value)/100; panVal.textContent=controls.panSpeed.toFixed(2); };

  function quickView(dirVec){
    const box=new THREE.Box3().setFromObject(pivot);
    const c=box.getCenter(new THREE.Vector3());
    const len=box.getSize(new THREE.Vector3()).length();
    const v=dirVec.clone().normalize().multiplyScalar(len*0.9).add(c);
    camera.position.copy(v);
    if(lockTarget.checked||!selectedClass){ controls.target.copy(c); }
    camera.updateProjectionMatrix();
  }
  viewIso.onclick=()=> quickView(new THREE.Vector3(1,0.7,1));
  viewTop.onclick=()=> quickView(new THREE.Vector3(0,1,0.0001));
  viewFront.onclick=()=> quickView(new THREE.Vector3(0,0,1));
  viewRight.onclick=()=> quickView(new THREE.Vector3(1,0,0));
  resetCam.onclick = ()=>{
    restoreBaselineCam();
    // optional: also re-center orbit pivot
    controls.target.copy(baselineCam?.target || new THREE.Vector3());
  };

  rotLeft.onclick=()=> { pivot.rotateY(+Math.PI/2); };
  rotRight.onclick=()=> { pivot.rotateY(-Math.PI/2); };
  fitAllBtn.onclick=()=>{ if(group) fit(); };

  gridToggle.onchange=()=> grid.visible=gridToggle.checked;
  axesToggle.onchange=()=> axes.visible=axesToggle.checked;
  toggleLabels.onchange=()=> placeLabels();

  // Prompt helpers
  function insertText(txt){ const start=promptEl.selectionStart||0, end=promptEl.selectionEnd||0, val=promptEl.value||''; promptEl.value=val.slice(0,start)+txt+val.slice(end); promptEl.focus(); promptEl.selectionStart=promptEl.selectionEnd=start+txt.length; }
  insertSelected.onclick=()=>{ if(selectedClass) insertText(` ${selectedClass} `); };

  // Image upload preview (area is visible by default)
  imgFile.onchange=()=>{
    const f=imgFile.files?.[0];
    if(!f){ imgPreview.removeAttribute('src'); return; }
    const r=new FileReader();
    r.onload=e=>{ imgPreview.src=e.target.result; };
    r.readAsDataURL(f);
  };
  clearImg.onclick=()=>{ imgFile.value=''; imgPreview.removeAttribute('src'); };

  // VLM → apply → reload → highlight
  async function applyVLMJson(jsonObj){
    let ghost=null;
    if(group){
      ghost = group.clone(true);
      ghost.traverse(o=>{
        if(o.material){
          const m=o.material.clone(); m.transparent=true; m.opacity=0.25;
          if(m.emissive) m.emissive.setHex(0x000000);
          o.material = m;
        }
      });
      pivot.add(ghost);
    }

    const r = await fetch('/apply', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({json: jsonObj})});
    if(!r.ok){ throw new Error(`apply: HTTP ${r.status}`); }
    const js = await r.json();
    if(!js.ok){ throw new Error(js.error||'apply failed'); }

    logLine(`Applied ${jsonObj.action||'modify'} ${jsonObj.target_component||''} ${JSON.stringify(jsonObj.parameters||{})}`);
    await loadModel();
    await refreshParamsHint();
    const key = js.highlight_key || jsonObj.target_component || '';
    if(key) selectClass(key, true);

    if(ghost){
      setTimeout(()=>{
        pivot.remove(ghost);
        ghost.traverse(o=>{ if(o.geometry) o.geometry.dispose(); if(o.material) o.material.dispose(); });
      }, 900);
    }
  }

  // Send to VLM
  sendVLM.onclick=async ()=>{
    const data=new FormData();
    data.append('prompt', promptEl.value || '');
    data.append('selected_class', selectedClass || '');
    data.append('classes', JSON.stringify([...classMap.keys()]));
    if(imgFile.files?.[0]) data.append('image', imgFile.files[0]);
    try{
      const r = await fetch('/vlm', {method:'POST', body:data});
      if(!r.ok){ throw new Error(`HTTP ${r.status}`); }
      const js = await r.json().catch(()=> ({}));
      const raw = js?.response?.raw || '';
      const parsed = js?.response?.json || null;
      if(parsed && typeof parsed === 'object'){
        vlmNotice.textContent = 'VLM: parsed JSON OK → applying…';
        vlmNotice.style.color = '#16a34a';
        await applyVLMJson(parsed);
      }else{
        vlmNotice.textContent = 'VLM responded, but no strict JSON found (check console).';
        vlmNotice.style.color = '#f59e0b';
        console.log('[VLM raw]', raw);
        logLine('VLM returned non-JSON. No changes applied.', 'warn');
      }
    }catch(e){
      vlmNotice.textContent = 'VLM endpoint not configured. Request prepared locally.';
      vlmNotice.style.color = '#b45309';
      logLine(String(e), 'err');
    }
  };

  // Hover highlight by class (lightweight)
  function updateHover(){
    if(!group) return;
    raycaster.setFromCamera(pointer,camera);
    const hits=raycaster.intersectObjects(group.children,true);
    let newHovered=null;
    if(hits.length){ let o=hits[0].object; while(o.parent && !o.name) o=o.parent; newHovered=classKeyFromName(o.name); }
    if(hovered!==newHovered){
      if(hovered && classMap.has(hovered) && hovered!==selectedClass){
        const e=classMap.get(hovered); e.nodes.forEach(node=> paintNode(node, e.color, null, 1));
      }
      hovered=newHovered;
      if(hovered && classMap.has(hovered) && hovered!==selectedClass){
        const e=classMap.get(hovered); e.nodes.forEach(node=> paintNode(node, e.color, hoverEmissive, 0.9));
      }
    }
  }

  // Undo/Redo
  btnUndo.onclick = async ()=>{
    const r = await fetch('/undo', {method:'POST'});
    const js = await r.json().catch(()=>({}));
    if(js.ok){ logLine('Undo applied. Reloading model...', 'warn'); await loadModel(); await refreshParamsHint(); }
    else { logLine(`Undo failed: ${js.error||'unknown'}`, 'err'); }
  };
  btnRedo.onclick = async ()=>{
    const r = await fetch('/redo', {method:'POST'});
    const js = await r.json().catch(()=>({}));
    if(js.ok){ logLine('Redo applied. Reloading model...', 'warn'); await loadModel(); await refreshParamsHint(); }
    else { logLine(`Redo failed: ${js.error||'unknown'}`, 'err'); }
  };

    // --- Whole-panel collapse/restore with persistence ---
  function applyPanelState() {
    const leftCollapsed  = localStorage.getItem('panel:left')  === '1';
    const rightCollapsed = localStorage.getItem('panel:right') === '1';
    document.body.classList.toggle('left-collapsed',  leftCollapsed);
    document.body.classList.toggle('right-collapsed', rightCollapsed);
  }
  function togglePanel(which) {
    const key = which === 'left' ? 'panel:left' : 'panel:right';
    const now = localStorage.getItem(key) === '1' ? '0' : '1';
    localStorage.setItem(key, now);
    applyPanelState();
    // ensure canvas resizes properly
    window.dispatchEvent(new Event('resize'));
  }

    // wire buttons (place after DOM is ready / in start())
  document.getElementById('toggleLeftPanel').onclick  = () => togglePanel('left');
  document.getElementById('toggleRightPanel').onclick = () => togglePanel('right');

  document.getElementById('reload').onclick=()=> loadModel().catch(e=> logLine(String(e),'err'));
  document.getElementById('clear').onclick=()=> selectClass(null);


  function animate(){ resize(); updateHover(); controls.update(); renderer.render(scene,camera); requestAnimationFrame(animate); }

  (async function start(){
  // call once during init (e.g., at top of start())
    applyPanelState();
    setupToggles();
    animate();
    await getMode();
    adjustColumns();
    await loadModel().catch(e=> logLine(String(e),'err'));
    fov.value=String(Math.round(camera.fov)); fovVal.textContent=`${Math.round(camera.fov)}°`;
    near.value=String(camera.near); far.value=String(camera.far);
    rotVal.textContent=(controls.rotateSpeed||1).toFixed(2);
    zoomVal.textContent=(controls.zoomSpeed||1).toFixed(2);
    panVal.textContent=(controls.panSpeed||1).toFixed(2);
    dampVal.textContent=controls.dampingFactor.toFixed(2);
    await refreshParamsHint();
  })();
</script></body></html>
"""

# --------------------------- BACKEND ---------------------------
app = Flask(__name__, static_folder='static')
STATE: Dict[str, Any] = {"selected_parts": []}

def _data_url_from_upload(file_storage) -> Optional[str]:
    if not file_storage: return None
    raw = file_storage.read()
    mime = file_storage.mimetype or "application/octet-stream"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"

def call_vlm(final_prompt: str, image_data_url: Optional[str]) -> Dict[str, Any]:
    """
    Prefer Ollama /api/generate if available, else optional custom LLAVA_URL.
    Returns {provider, raw} where raw is the model text (ideally JSON).
    """
    err = None

    if OLLAMA_URL:
        try:
            payload = {"model": OLLAMA_MODEL, "prompt": final_prompt, "stream": False}
            if image_data_url:
                if image_data_url.startswith("data:"):
                    payload["images"] = [image_data_url.split(",", 1)[1]]
                else:
                    payload["images"] = [image_data_url]
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
            r.raise_for_status()
            txt = r.json().get("response", "")
            return {"provider": "ollama", "raw": txt}
        except Exception as e:
            err = f"Ollama error: {e}"

    if LLAVA_URL:
        try:
            payload = {"prompt": final_prompt}
            if image_data_url: payload["image"] = image_data_url
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

@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.get("/mode")
def mode():
    mode = "GLB: assets/rover.glb" if os.path.exists(ROVER_GLB_PATH) else ("cqparts" if USE_CQPARTS else "fallback")
    return jsonify({"mode": mode})

@app.post("/label")
def label():
    data = request.get_json(force=True, silent=True) or {}
    part = (data.get("part_name") or "").strip()
    if part:
        STATE["selected_parts"].append(part)
        return jsonify({"ok": True, "part": part, "count": len(STATE["selected_parts"])})
    return jsonify({"ok": False, "error": "no part_name"})

@app.get("/labels")
def labels():
    return jsonify({"ok": True, "selected_parts": STATE["selected_parts"]})

@app.post("/recommend")
def recommend():
    try:
        # optional user free-text prompt
        prompt = (request.form.get("prompt") or "").strip()
        classes = request.form.get("classes") or "[]"
        try:
            classes = json.loads(classes)
            if not isinstance(classes, list): classes = []
        except Exception:
            classes = []

        # reference image is required for this endpoint
        ref_url = _data_url_from_upload(request.files.get("reference"))
        if not ref_url:
            return jsonify({"ok": False, "error": "no reference image"}), 400

        # optional current snapshot (canvas sent from client)
        snapshot_url = _data_url_from_upload(request.files.get("snapshot"))

        grounding = []
        grounding.append("Goal: Propose parametric changes so CAD matches the reference image better.")
        grounding.append("Known classes:")
        grounding += [f"- {c}" for c in classes]
        if prompt:
            grounding.append("\nUser prompt:\n" + prompt)
        if snapshot_url:
            grounding.append("\nA second image is the current CAD snapshot (for comparison).")

        final_prompt = f"{VLM_SYSTEM_PROMPT}\n\n---\n" + "\n".join(grounding)

        # concatenate images; Ollama supports multiple images
        provider_out = call_vlm(final_prompt, ref_url if not snapshot_url else ref_url)
        raw = provider_out.get("raw", "")

        # try strict JSON, or last {...}
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\[[\s\S]*\]\s*$|\{[\s\S]*\}\s*$", raw.strip())
            if m:
                try: parsed = json.loads(m.group(0))
                except Exception: parsed = None

        return jsonify({"ok": True, "response": {"raw": raw, "json": parsed}})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/vlm")
def vlm():
    try:
        prompt = (request.form.get("prompt") or "").strip()
        selected = (request.form.get("selected_class") or "").strip() or None
        try:
            classes = json.loads(request.form.get("classes") or "[]")
            if not isinstance(classes, list): classes = []
        except Exception:
            classes = []

        data_url = _data_url_from_upload(request.files.get("image"))

        grounding = []
        grounding.append("Known component classes:")
        for c in classes: grounding.append(f"- {c}")
        if selected: grounding.append(f"\nUser highlighted class: {selected}")
        grounding.append("\nUser prompt:\n" + prompt)

        final_prompt = f"{VLM_SYSTEM_PROMPT}\n\n---\n" + "\n".join(grounding)

        resp = call_vlm(final_prompt, data_url)
        raw = resp.get("raw", "")

        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}\s*$", raw.strip())
            if m:
                try: parsed = json.loads(m.group(0))
                except Exception: parsed = None

        return jsonify({"ok": True, "provider": resp.get("provider"), "response": {"raw": raw, "json": parsed}})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------------------- PARAM INTROSPECTION ----------------------
def _introspect_params_from_cls(cls) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    # Try cqparts param descriptors
    for name in dir(cls):
        if name.startswith("_"): continue
        try:
            val = getattr(cls, name)
            mod = getattr(getattr(val, "__class__", object), "__module__", "")
            if "cqparts.params" in mod:
                d[name] = str(val)
        except Exception:
            pass
    # Heuristic: numeric-ish class attributes
    for name in dir(cls):
        if name in d or name.startswith("_"): continue
        try:
            val = getattr(cls, name)
            if isinstance(val, (int, float)): d[name] = val
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
    return jsonify({"ok": True, "params": info})

# --------------------------- APPLY / UNDO / REDO ---------------------------
def _rebuild_and_save_glb():
    build_rover_scene_glb_cqparts()

@app.post("/apply")
def apply_change():
    try:
        spec = (request.get_json(force=True) or {}).get("json") or {}
        action = (spec.get("action") or "").strip()
        target = (spec.get("target_component") or "").strip().lower()
        params = spec.get("parameters") or {}

        # Normalize a common 'add wheel(s)' phrasing into wheels_per_side
        if action == "add" and "wheel" in target:
            cnt = params.get("count")
            wps = params.get("wheels_per_side")
            if wps is None and cnt is not None:
                # if count is total wheels, convert to per side (round up)
                try:
                    c = int(cnt)
                    params["wheels_per_side"] = max(1, (c + 1)//2)
                except Exception:
                    pass

        _push_history()
        rv = Rover(stepper=_Stepper, electronics=_Electronics, sensors=_PanTilt, wheel=_ThisWheel)
        apply_params_to_rover(rv, params)
        _rebuild_and_save_glb()

        return jsonify({"ok": True, "highlight_key": target or "wheel"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# @app.post("/apply")
# def apply_change():
#     try:
#         spec = (request.get_json(force=True) or {}).get("json") or {}
#         params = spec.get("parameters") or {}
#         target = (spec.get("target_component") or "").strip()

#         _push_history()  # snapshot before change
#         rv = Rover(stepper=_Stepper, electronics=_Electronics, sensors=_PanTilt, wheel=_ThisWheel)
#         apply_params_to_rover(rv, params)
#         _rebuild_and_save_glb()

#         return jsonify({"ok": True, "highlight_key": target})
#     except Exception as e:
#         return jsonify({"ok": False, "error": str(e)}), 500

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

# --------------------------- CQPARTS BUILD ---------------------------
def build_rover_scene_glb_cqparts() -> bytes:
    from cadquery import exporters

    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    print("Generating GLB file using cqparts...")

    rv = Rover(stepper=_Stepper, electronics=_Electronics, sensors=_PanTilt, wheel=_ThisWheel)
    print(f"Created object {rv}")

    for name, cls in (("stepper", _Stepper), ("electronics", _Electronics),
                      ("sensors", _PanTilt), ("wheel", _ThisWheel)):
        if not getattr(rv, name, None):
            setattr(rv, name, cls)
    print("Confirmed attributes exist")

    try:
        import cqparts
        from cqparts.params import PositiveFloat

        class NEMA17Stepper(cqparts.Part):
            width        = PositiveFloat(42.0)
            length       = PositiveFloat(47.0)
            shaft_diam   = PositiveFloat(5.0)
            shaft_length = PositiveFloat(22.0)
            boss_diam    = PositiveFloat(22.0)
            boss_length  = PositiveFloat(2.0)
            hole_spacing = PositiveFloat(31.0)
            hole_diam    = PositiveFloat(3.0)

            def cut_boss(self, extra: float = 0.5, depth: float | None = None, **kwargs):
                if "clearance" in kwargs and kwargs["clearance"] is not None:
                    extra = float(kwargs["clearance"])
                d = float(self.boss_diam) + 2.0 * float(extra)
                h = float(depth) if depth is not None else float(self.boss_length) + 0.5
                return cq.Workplane("XY").circle(d / 2.0).extrude(h)

            def get_shaft(self):
                z0 = float(self.length + self.boss_length)
                return (cq.Workplane("XY")
                        .circle(float(self.shaft_diam) / 2.0)
                        .extrude(float(self.shaft_length))
                        .translate((0, 0, z0)))

            def cut_shaft(self, clearance: float = 0.2, depth: float | None = None, **_):
                dia = float(self.shaft_diam) + 2.0 * float(clearance)
                h = float(depth) if depth is not None else float(self.shaft_length) + 1.0
                z0 = float(self.length + self.boss_length)
                return (cq.Workplane("XY")
                        .circle(dia / 2.0)
                        .extrude(h)
                        .translate((0, 0, z0)))

            def get_shaft_axis(self):
                z0 = float(self.length + self.boss_length)
                z1 = z0 + float(self.shaft_length)
                return (0.0, 0.0, z0), (0.0, 0.0, z1)

            @property
            def front_z(self) -> float:
                return float(self.length + self.boss_length)

            def make(self):
                body  = cq.Workplane("XY").box(self.width, self.width, self.length, centered=(True, True, False))
                boss  = cq.Workplane("XY").circle(self.boss_diam/2.0).extrude(self.boss_length).translate((0, 0, self.length))
                shaft = cq.Workplane("XY").circle(self.shaft_diam/2.0).extrude(self.shaft_length).translate((0, 0, self.length + self.boss_length))
                motor = body.union(boss).union(shaft)
                s = float(self.hole_spacing) / 2.0
                pts = [( s,  s), (-s,  s), ( s, -s), (-s, -s)]
                motor = (motor.faces(">Z").workplane(origin=(0, 0, self.length + self.boss_length))
                         .pushPoints(pts).hole(self.hole_diam, depth=self.boss_length + 5.0))
                return motor

            class _Pt:
                __slots__ = ("x","y","z","X","Y","Z")
                def __init__(self, x, y, z):
                    self.x = float(x); self.y = float(y); self.z = float(z)
                    self.X = self.x; self.Y = self.y; self.Z = self.z
                def __iter__(self):
                    yield self.x; yield self.y; yield self.z
                def toTuple(self):
                    return (self.x, self.y, self.z)
                def __repr__(self):
                    return f"_Pt({self.x}, {self.y}, {self.z})"

            class _MountPoints(list):
                def __call__(self):
                    return self

            @property
            def mount_points(self):
                s = float(self.hole_spacing) / 2.0
                z = float(self.length + self.boss_length)
                return self._MountPoints([
                    self._Pt( s,  s, z),
                    self._Pt(-s,  s, z),
                    self._Pt( s, -s, z),
                    self._Pt(-s, -s, z),
                ])

            def get_mount_points(self):
                return list(self.mount_points)

        rv.stepper = NEMA17Stepper
        print("[patch] Using NEMA17Stepper (cqparts.Part) for rover.stepper")
    except Exception as e:
        print("[patch warn] could not define/attach NEMA17Stepper component:", e)

    print("Building assembly (threaded timeout 25s)...")
    built = False
    build_err = [None]

    def _run_build():
        try:
            from electronics import OtherBatt as _OtherBatt
            _OtherBatt.make_constraints = lambda self: []
            def _otherbatt_make_components(self):
                self.local_obj = cq.Workplane("XY").box(60, 30, 15)
                return {}
            _OtherBatt.make_components = _otherbatt_make_components
            rv.build()
        except Exception as e:
            build_err[0] = e

    t = threading.Thread(target=_run_build, daemon=True)
    t.start()
    t.join(25.0)
    if t.is_alive():
        print("[timeout] rv.build() exceeded 25s; attempting assembly export anyway")
    else:
        if build_err[0] is None:
            built = True
            print("Build completed.")
        else:
            print("[warn] build error:", build_err[0])

    comps_local: Dict[str, Any] = getattr(rv, "components", {}) or {}
    if not built:
        print("Falling back to rv.make_components() only...")
        try:
            gen = rv.make_components()
            if isinstance(gen, dict):
                comps_local = gen
            else:
                comps_local = {}
                for k, v in gen: comps_local[k] = v
            try:
                object.__setattr__(rv, "components", comps_local)
            except Exception as e:
                print("[info] rv.components read-only; using local comps only:", e)
        except Exception as e:
            print("[error] make_components() failed:", e)
            comps_local = {}

    if not comps_local and getattr(rv, "components", None):
        comps_local = rv.components

    from cadquery import exporters

    def _cq_to_trimesh(obj, tol=0.6):
        try:
            stl_txt = exporters.toString(obj, "STL", tolerance=tol).encode("utf-8")
            m = trimesh.load(io.BytesIO(stl_txt), file_type="stl")
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(tuple(m.geometry.values()))
            return m
        except Exception as e:
            print("[warn] STL export failed:", e)
            return None

    def _get_shape(component):
        for attr in ("world_obj", "toCompound", "obj", "to_cadquery", "shape", "local_obj", "make"):
            if hasattr(component, attr):
                try:
                    val = getattr(component, attr)
                    shp = val() if callable(val) else val
                    if shp is not None: return shp
                except Exception as e:
                    print(f"[warn] {component.__class__.__name__}.{attr}() failed: {e}")
        return None

    def _iter_components(root):
        comps = getattr(root, "components", None)
        if isinstance(comps, dict): return comps.items()
        if comps:
            try: return list(comps)
            except Exception: pass
        return []

    scene = trimesh.Scene()
    print("Creating scene")

    def _walk_and_add(node, prefix=""):
        shp = _get_shape(node)
        if shp is not None:
            tm = _cq_to_trimesh(shp, tol=0.6)
            if tm and not getattr(tm, "is_empty", False):
                nm = prefix.rstrip("/") or node.__class__.__name__
                try: scene.add_geometry(tm, node_name=nm)
                except Exception as e: print(f"[warn] add_geometry({nm}) failed:", e)
        for child_name, child in _iter_components(node):
            child_prefix = f"{prefix}{child_name}/"
            _walk_and_add(child, child_prefix)

    whole = None
    for attr in ("world_obj", "toCompound", "obj", "to_cadquery"):
        if hasattr(rv, attr):
            try:
                cand = getattr(rv, attr)
                whole = cand() if callable(cand) else cand
                if whole is not None: break
            except Exception as e:
                print(f"[asm] rv.{attr} failed:", e)

    if whole is not None:
        mesh = _cq_to_trimesh(whole, tol=0.6)
        if mesh and not getattr(mesh, "is_empty", False):
            try: scene.add_geometry(mesh, node_name="Rover")
            except Exception as e: print("[warn] add Rover geometry failed:", e)
        else:
            print("[asm] world compound empty; descending into components")
            _walk_and_add(rv, prefix="")
    else:
        print("[asm] no world compound API; descending into components")
        _walk_and_add(rv, prefix="")

    if not scene.geometry:
        raise RuntimeError("cqparts rover: no component geometry exported")

    print("GLB file generated successfully.")
    glb_bytes = scene.export(file_type="glb")
    out_path = os.path.join(assets_dir, "rover.glb")
    with open(out_path, "wb") as f: f.write(glb_bytes)
    print(f"GLB file saved to {out_path} (bytes: {len(glb_bytes)})")
    return glb_bytes

# --------------------------- MODEL ROUTE ---------------------------
def build_rover_scene_glb(_: Optional[Dict[str, Any]] = None) -> bytes:
    if USE_CQPARTS:
        print("Using cqparts to generate the GLB file.")
        return build_rover_scene_glb_cqparts()
    if os.path.exists(ROVER_GLB_PATH):
        print(f"Using existing GLB file at {ROVER_GLB_PATH}")
        with open(ROVER_GLB_PATH, "rb") as f: return f.read()
    print("No GLB file found, and cqparts is disabled.")
    raise FileNotFoundError("cqparts disabled, and assets/rover.glb not found")

@app.get("/model.glb")
def model_glb():
    try:
        glb = build_rover_scene_glb({})
        return send_file(io.BytesIO(glb), mimetype="model/gltf-binary")
    except Exception:
        import traceback
        return Response("model.glb build failed:\n" + traceback.format_exc(),
                        status=500, mimetype="text/plain")

# --------------------------- STATIC ---------------------------
@app.route('/static/<path:filename>')
def custom_static(filename):
    root = app.static_folder
    if not os.path.exists(os.path.join(root, filename)): abort(404)
    if filename.endswith('.js'):
        return send_from_directory(root, filename, mimetype='application/javascript')
    return send_from_directory(root, filename)

# --------------------------- MAIN ---------------------------
def _warm_build():
    try:
        print("[warm] starting initial build…")
        build_rover_scene_glb({})
        print("[warm] initial GLB ready.")
    except Exception as e:
        # Don't crash server; front-end console will log any subsequent issues.
        print("[warm] initial build failed:", e)

if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "assets"), exist_ok=True)
    # Kick off a warm build so the model is ready by the time the page loads.
    threading.Thread(target=_warm_build, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
