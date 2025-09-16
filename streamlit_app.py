# streamlit_app.py — CardBlur (Streamlit, Upload + Live WebRTC with anti-flicker, fast drop-off)
# - Upload image OR use true live camera (no snapshots)
# - Modes: Text only / Face only / Text + Face / Whole card
# - Live path has temporal smoothing to prevent flicker, tuned to drop blur quickly

import os, io, pathlib, urllib.request, threading
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# -------- live video deps --------
HAS_WEBRTC = False
HAS_AV = False
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
    HAS_WEBRTC = True
except Exception:
    pass
try:
    import av  # required by streamlit-webrtc
    HAS_AV = True
except Exception:
    pass

_infer_lock = threading.Lock()  # YOLO safety across frames

# =========================
# CONFIG (env/secrets friendly)
# =========================
BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "best.pt"
WEIGHTS_PATH = pathlib.Path(os.environ.get("WEIGHTS_PATH", str(DEFAULT_WEIGHTS))).resolve()

# Inference settings
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
CONF     = float(os.environ.get("CONF", 0.28))
IOU      = float(os.environ.get("IOU", 0.5))

# Upload-only settings (tiled + flip for better recall)
UPLOAD_CONF   = float(os.environ.get("UPLOAD_CONF", 0.15))
UPLOAD_IOU    = float(os.environ.get("UPLOAD_IOU", 0.5))
TILE_SIZE     = int(os.environ.get("TILE_SIZE", 960))
TILE_OVERLAP  = float(os.environ.get("TILE_OVERLAP", 0.20))

# Live options (FAST DROP-OFF DEFAULTS)
LIVE_DETECT_EVERY   = int(os.environ.get("LIVE_DETECT_EVERY", "1"))   # run detect every frame (was 2)
LIVE_ALWAYS_BLUR_DOC= os.environ.get("LIVE_ALWAYS_BLUR_DOC", "1") == "1"     # Whole card: blur every detected doc box
LIVE_DEBUG          = os.environ.get("LIVE_DEBUG", "0") == "1"               # draw counts on the frame

# --- anti-flicker / smoothing (frames) ---
LIVE_HOLD_FRAMES    = int(os.environ.get("LIVE_HOLD_FRAMES", "6"))   # keep blur ~0.3–0.6s, was 24
LIVE_WARMUP_FRAMES  = int(os.environ.get("LIVE_WARMUP_FRAMES", "2"))  # need to see a box this many times first
LIVE_MATCH_IOU      = float(os.environ.get("LIVE_MATCH_IOU", "0.35")) # how close boxes must be to match
LIVE_SMOOTH         = float(os.environ.get("LIVE_SMOOTH", "0.35"))    # less laggy than 0.5

# Text post-process
TEXT_DILATE_FRAC    = float(os.environ.get("TEXT_DILATE_FRAC", 0.010))
TEXT_MERGE_GAP_FRAC = float(os.environ.get("TEXT_MERGE_GAP_FRAC", 0.010))
TEXT_MAX_DOC_FRAC   = float(os.environ.get("TEXT_MAX_DOC_FRAC", 0.50))
TEXT_MIN_H_FRAC     = float(os.environ.get("TEXT_MIN_H_FRAC", 0.012))
TEXT_MAX_H_FRAC     = float(os.environ.get("TEXT_MAX_H_FRAC", 0.28))
TEXT_MIN_AR         = float(os.environ.get("TEXT_MIN_AR", 2.3))
TEXT_MAX_AR         = float(os.environ.get("TEXT_MAX_AR", 40.0))
TEXT_NMS_IOU        = float(os.environ.get("TEXT_NMS_IOU", 0.35))

# OCR (disabled by default to keep build light)
USE_OCR       = os.environ.get("USE_OCR", "0") == "1"
OCR_LANG      = os.environ.get("OCR_LANG", "ar,en")
OCR_MIN_CONF  = float(os.environ.get("OCR_MIN_CONF", 0.60))
OCR_EXPAND_PX = int(os.environ.get("OCR_EXPAND_PX", 2))

# Blur strength
MIN_KERNEL   = int(os.environ.get("MIN_KERNEL", 31))
KERNEL_SCALE = float(os.environ.get("KERNEL_SCALE", 0.22))

# Labels (match your model)
DOC_LABELS  = {"id", "id_card", "idcard", "passport", "mrz", "serial", "number", "document", "card", "passport_id", "name", "dob", "expiry"}
FACE_LABELS = {"face", "person_face", "head"}
TEXT_LABELS = {"text"}

DOC_PAD_FRAC = float(os.environ.get("DOC_PAD_FRAC", 0.08))

# =========================
# UI
# =========================
st.set_page_config(page_title="CardBlur", page_icon="🪪", layout="wide")
st.title("🪪 CardBlur")
st.caption("AI-powered privacy protection for IDs & passports — by Shatha Khawaji • Renad Almutairi • Jury Alsultan • Yara Alsardi")

with st.sidebar:
    st.header("Options")
    blur_mode = st.radio("What to blur?", ["Text only", "Face only", "Text + Face", "Whole card"], index=3)
    st.caption("Tip: 'Whole card' will blur detected card boxes in live mode.")

# =========================
# Helpers
# =========================
def make_odd(n): return n if n % 2 == 1 else n + 1
def compute_kernel(w, h):
    k = int(max(w, h) * KERNEL_SCALE)
    k = max(k, MIN_KERNEL)
    return make_odd(k)

def blur_region(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2));     y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1: return
    k = compute_kernel(x2 - x1, y2 - y1)
    if k < 3: k = 3
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)

def center(x1, y1, x2, y2): return ((x1 + x2) // 2, (y1 + y2) // 2)
def contains(box, pt):
    x1, y1, x2, y2 = box; x, y = pt
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def pad_box(box, pad_frac, W, H):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_frac), int(h * pad_frac)
    nx1 = max(0, x1 - px); ny1 = max(0, y1 - py)
    nx2 = min(W, x2 + px); ny2 = min(H, y2 + py)
    return (nx1, ny1, nx2, ny2)

def expand_px(box, px=4, py=4, W=99999, H=99999):
    x1, y1, x2, y2 = box
    return (max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (ua + ub - inter + 1e-6)

def box_area(b):
    x1,y1,x2,y2=b
    return max(0,x2-x1)*max(0,y2-y1)

def merge_boxes_overlap_or_near(boxes, max_gap_px):
    if not boxes: return []
    boxes = boxes[:]
    merged = []
    used = [False]*len(boxes)

    def near_or_overlap(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        if not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1):
            return True
        h_gap = max(0, max(bx1 - ax2, ax1 - bx2))
        v_overlap = min(ay2, by2) - max(ay1, by1)
        if h_gap <= max_gap_px and v_overlap > 0: return True
        v_gap = max(0, max(by1 - ay2, ay1 - by2))
        h_overlap = min(ax2, bx2) - max(ax1, bx1)
        if v_gap <= max_gap_px and h_overlap > 0: return True
        return False

    for i in range(len(boxes)):
        if used[i]: continue
        cur = boxes[i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]: continue
                if near_or_overlap(cur, boxes[j]):
                    x1 = min(cur[0], boxes[j][0]); y1 = min(cur[1], boxes[j][1])
                    x2 = max(cur[2], boxes[j][2]); y2 = max(cur[3], boxes[j][3])
                    cur = (x1,y1,x2,y2)
                    used[j] = True
                    changed = True
        merged.append(cur)
    return merged

def nms_boxes(boxes, scores=None, iou_th=0.5):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=float)
    scores = np.ones(len(boxes), dtype=float) if scores is None else np.array(scores, dtype=float)
    order = scores.argsort()[::-1]
    keep = []
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0: return 0.0
        ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        ub = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / (ua + ub - inter + 1e-6)
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        if order.size == 1: break
        rest = order[1:]
        order = np.array([k for k in rest if _iou(boxes[i], boxes[int(k)]) <= iou_th])
    return keep

# =========================
# Weights + Model
# =========================
def ensure_weights() -> str:
    for p in [BASE_DIR/"best.pt", BASE_DIR/"models"/"best.pt", WEIGHTS_PATH]:
        if p and pathlib.Path(p).exists():
            return str(pathlib.Path(p).resolve())
    url = None
    try:
        url = st.secrets.get("WEIGHTS_URL")
    except Exception:
        url = None
    if url:
        cache_dir = pathlib.Path.home()/".cache"/"cardblur"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir/"best.pt"
        if not dst.exists():
            with st.status("Downloading model weights…"):
                urllib.request.urlretrieve(url, dst)
        return str(dst.resolve())
    raise FileNotFoundError("best.pt not found. Put it in repo root or set WEIGHTS_URL in Streamlit secrets.")

@st.cache_resource(show_spinner=True)
def load_model(weights_path: str):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    try:
        model.fuse()
    except Exception:
        pass
    names = getattr(model, "names", {})
    return model, names

# =========================
# Inference utilities
# =========================
def _predict(model, names, img_rgb, conf, iou):
    out = []
    try:
        results = model.predict(img_rgb, imgsz=IMG_SIZE, conf=conf, iou=iou, verbose=False)
    except Exception as e:
        st.warning(f"Inference error: {e}")
        return out
    if not results:
        return out
    res = results[0]

    def npint(x): return x.cpu().numpy().astype(int)
    def npfloat(x): return x.cpu().numpy().astype(float)

    try:
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            xyxy = npint(res.boxes.xyxy)
            cls  = res.boxes.cls.int().tolist() if hasattr(res.boxes, "cls") else [None]*len(xyxy)
            scr  = npfloat(res.boxes.conf).tolist() if hasattr(res.boxes, "conf") else [1.0]*len(xyxy)
            for coords, c, s in zip(xyxy.tolist(), cls, scr):
                label = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
                out.append((tuple(coords), (label or "").lower(), float(s)))
    except Exception:
        pass
    try:
        if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
            xyxy = npint(res.obb.xyxy)
            cls  = res.obb.cls.int().tolist() if hasattr(res.obb, "cls") else [None]*len(xyxy)
            scr  = npfloat(res.obb.conf).tolist() if hasattr(res.obb, "conf") else [1.0]*len(xyxy)
            for coords, c, s in zip(xyxy.tolist(), cls, scr):
                label = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
                out.append((tuple(coords), (label or "").lower(), float(s)))
    except Exception:
        pass
    return out

def _gather_boxes(preds):
    doc_boxes, face_boxes, face_scores, text_boxes, text_scores = [], [], [], [], []
    for (x1,y1,x2,y2), label, score in preds:
        if label in DOC_LABELS:    doc_boxes.append((x1,y1,x2,y2))
        elif label in FACE_LABELS: face_boxes.append((x1,y1,x2,y2)); face_scores.append(score)
        elif label in TEXT_LABELS: text_boxes.append((x1,y1,x2,y2)); text_scores.append(score)
    if face_boxes:
        keep = nms_boxes(face_boxes, face_scores, iou_th=0.45)
        face_boxes = [face_boxes[i] for i in keep]
    if text_boxes:
        keep = nms_boxes(text_boxes, None, iou_th=TEXT_NMS_IOU)
        text_boxes = [text_boxes[i] for i in keep]
    return doc_boxes, face_boxes, text_boxes

# ---------- Upload path (kept) ----------
def _predict_tiled(model, names, img_rgb, conf, iou):
    H, W = img_rgb.shape[:2]
    ts = min(TILE_SIZE, max(H, W))
    step = int(ts * (1 - TILE_OVERLAP))
    if step <= 0:
        return _predict(model, names, img_rgb, conf, iou)
    out = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            x2 = min(x + ts, W); y2 = min(y + ts, H)
            preds = _predict(model, names, img_rgb[y:y2, x:x2], conf, iou)
            for (bx1, by1, bx2, by2), label, score in preds:
                out.append(((bx1 + x, by1 + y, bx2 + x, by2 + y), label, score))
    return out

def _predict_tta_flip(model, names, img_rgb, conf, iou):
    H, W = img_rgb.shape[:2]
    flip = cv2.flip(img_rgb, 1)
    preds = _predict(model, names, flip, conf, iou)
    mapped = []
    for (x1, y1, x2, y2), label, score in preds:
        nx1 = W - x2; nx2 = W - x1
        mapped.append(((nx1, y1, nx2, y2), label, score))
    return mapped

def _filter_text_geometry(text_boxes, W, H):
    out = []
    for (x1,y1,x2,y2) in text_boxes:
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        h_frac = h / H; ar = w / h
        if h_frac < TEXT_MIN_H_FRAC: continue
        if h_frac > TEXT_MAX_H_FRAC: continue
        if ar < TEXT_MIN_AR or ar > TEXT_MAX_AR:
            if not (h_frac < (TEXT_MIN_H_FRAC*1.5) and 1.2 <= ar <= 2.5):
                continue
        out.append((x1,y1,x2,y2))
    return out

def _group_by_rows(boxes, row_tol_px):
    if not boxes: return []
    ys = [ ( (b[1]+b[3])//2, i ) for i,b in enumerate(boxes) ]
    ys.sort()
    groups, cur, last_y = [], [], None
    for yc, i in ys:
        if last_y is None or abs(yc - last_y) <= row_tol_px: cur.append(i)
        else: groups.append(cur); cur=[i]
        last_y = yc
    if cur: groups.append(cur)
    return [[boxes[i] for i in g] for g in groups]

def _conservative_text(text_boxes, doc_boxes, W, H):
    if not text_boxes: return []
    text_boxes = _filter_text_geometry(text_boxes, W, H)
    if not text_boxes: return []
    px = max(2, int(TEXT_DILATE_FRAC * W)); py = max(2, int(TEXT_DILATE_FRAC * H))
    dil = [expand_px(t, px=px, py=py, W=W, H=H) for t in text_boxes]
    row_tol = max(3, int(0.012 * H))
    row_groups = _group_by_rows(dil, row_tol_px=row_tol)
    gap = int(TEXT_MERGE_GAP_FRAC * max(W, H))
    merged_all = []
    for grp in row_groups:
        merged_all.extend(merge_boxes_overlap_or_near(grp, max_gap_px=gap))
    if doc_boxes and merged_all:
        safe = []
        for m in merged_all:
            cx, cy = center(*m)
            chosen = None
            for d in doc_boxes:
                if contains(d, (cx, cy)):
                    chosen = d; break
            if chosen is None and doc_boxes:
                ious = [iou(m, d) for d in doc_boxes]
                chosen = doc_boxes[int(np.argmax(ious))] if any(ious) else None
            if chosen and box_area(m) > TEXT_MAX_DOC_FRAC * box_area(chosen):
                inside = [b for b in dil if contains(chosen, center(*b))]
                safe.extend(merge_boxes_overlap_or_near(inside, max_gap_px=gap))
            else:
                safe.append(m)
        merged_all = safe
    if merged_all:
        keep = nms_boxes(merged_all, None, iou_th=0.4)
        merged_all = [merged_all[i] for i in keep]
    return merged_all

def run_upload(model, names, bgr_image: np.ndarray, mode: str = "both") -> np.ndarray:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    preds = []
    preds += _predict(model, names, rgb, UPLOAD_CONF, UPLOAD_IOU)
    preds += _predict_tiled(model, names, rgb, UPLOAD_CONF, UPLOAD_IOU)
    preds += _predict_tta_flip(model, names, rgb, UPLOAD_CONF, UPLOAD_IOU)
    doc_boxes, face_boxes, text_boxes = _gather_boxes(preds)
    H, W = bgr_image.shape[:2]
    if doc_boxes:
        text_boxes = [t for t in text_boxes if any(contains(d, center(*t)) for d in doc_boxes)]
        face_boxes = [f for f in face_boxes if any(contains(d, center(*f)) for d in doc_boxes)]
    text_boxes = _conservative_text(text_boxes, doc_boxes, W, H)

    m = (mode or "both").lower()
    if m in ("text only", "text"): targets = text_boxes
    elif m in ("face only", "face"): targets = face_boxes
    elif m in ("text + face", "both"):
        merged = text_boxes + face_boxes
        targets = [merged[i] for i in nms_boxes(merged, None, iou_th=0.3)] if merged else []
    elif m in ("whole card", "doc"):
        targets = [pad_box(d, DOC_PAD_FRAC, W, H) for d in doc_boxes]
    else:
        targets = []
    for (x1, y1, x2, y2) in targets:
        blur_region(bgr_image, x1, y1, x2, y2)
    return bgr_image

# ---------- Live helpers ----------
def live_get_targets(model, names, bgr_image: np.ndarray, mode: str):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    preds = _predict(model, names, rgb, CONF, IOU)
    doc_boxes, face_boxes, text_boxes = _gather_boxes(preds)
    H, W = bgr_image.shape[:2]

    if LIVE_DEBUG:
        dbg = f"doc:{len(doc_boxes)} face:{len(face_boxes)} text:{len(text_boxes)}"
        cv2.putText(bgr_image, dbg, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (32, 255, 32), 2, cv2.LINE_AA)

    m = (mode or "").lower()
    if m in ("whole card", "doc"):
        if LIVE_ALWAYS_BLUR_DOC:
            targets = [pad_box(d, DOC_PAD_FRAC, W, H) for d in doc_boxes]
        else:
            targets = []
            for d in doc_boxes:
                if any(contains(d, center(*f)) for f in face_boxes) and any(contains(d, center(*t)) for t in text_boxes):
                    targets.append(pad_box(d, DOC_PAD_FRAC, W, H))
    elif m in ("text only", "text"):
        targets = text_boxes
    elif m in ("face only", "face"):
        targets = face_boxes
    else:  # "Text + Face"
        merged = text_boxes + face_boxes
        targets = [merged[i] for i in nms_boxes(merged, None, iou_th=0.3)] if merged else []

    return targets

class BoxTracker:
    """
    Stabilize targets across frames:
    - matches by IoU
    - EMA-smooths box coords
    - holds boxes for N frames after last seen (prevents flicker)
    """
    def __init__(self, hold_frames=24, warmup=2, match_iou=0.35, smooth=0.5):
        self.hold = hold_frames
        self.warmup = warmup
        self.match_iou = match_iou
        self.smooth = smooth
        self.tracks = []  # each: {box:(x1,y1,x2,y2), ttl:int, hits:int}

    def _ema(self, a, b):
        return int(a * self.smooth + b * (1.0 - self.smooth))

    def update(self, new_boxes):
        # decay existing
        for t in self.tracks:
            t["ttl"] -= 1

        # greedy match
        for b in new_boxes:
            best = None
            best_iou = 0.0
            for t in self.tracks:
                i = iou(t["box"], b)
                if i > best_iou:
                    best_iou = i
                    best = t
            if best is not None and best_iou >= self.match_iou:
                x1 = self._ema(best["box"][0], b[0])
                y1 = self._ema(best["box"][1], b[1])
                x2 = self._ema(best["box"][2], b[2])
                y2 = self._ema(best["box"][3], b[3])
                best["box"] = (x1, y1, x2, y2)
                best["ttl"] = self.hold
                best["hits"] += 1
            else:
                self.tracks.append({"box": b, "ttl": self.hold, "hits": 1})

        # keep alive
        self.tracks = [t for t in self.tracks if t["ttl"] > 0]
        # matured tracks are shown
        return [t["box"] for t in self.tracks if t["hits"] >= self.warmup]

# =========================
# Load model once
# =========================
try:
    weights_path = ensure_weights()
    (model, names) = load_model(weights_path)
    st.success("Model loaded ✅")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# =========================
# Tabs: Upload | Live (WebRTC)
# =========================
tab_upload, tab_live = st.tabs(["📁 Upload", "🎥 Live (WebRTC)"])

with tab_upload:
    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])
    if file:
        pil = Image.open(file).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        with _infer_lock:
            out = run_upload(model, names, bgr.copy(), blur_mode)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Blurred", use_column_width=True)
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            st.download_button("⬇ Download blurred image", buf.tobytes(), file_name="cardblur_result.jpg", mime="image/jpeg")

with tab_live:
    if not HAS_WEBRTC or not HAS_AV:
        st.info("Install streamlit-webrtc and av in requirements to enable live video.")
    else:
        class LiveProcessor(VideoProcessorBase):
            def __init__(self):
                self.frame_id = 0
                self.tracker = BoxTracker(
                    hold_frames=LIVE_HOLD_FRAMES,
                    warmup=LIVE_WARMUP_FRAMES,
                    match_iou=LIVE_MATCH_IOU,
                    smooth=LIVE_SMOOTH,
                )

            def recv(self, frame):
                self.frame_id += 1
                img = frame.to_ndarray(format="bgr24")

                # every Nth frame: run detection; otherwise reuse/decay previous boxes
                if self.frame_id % LIVE_DETECT_EVERY == 0:
                    with _infer_lock:
                        new_targets = live_get_targets(model, names, img, blur_mode)
                    matured = self.tracker.update(new_targets)
                else:
                    matured = self.tracker.update([])

                for (x1, y1, x2, y2) in matured:
                    blur_region(img, x1, y1, x2, y2)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(
            key="cardblur-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_cfg,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=LiveProcessor,
        )

# Debug: show labels
try:
    label_list = list(names.values()) if isinstance(names, dict) else list(names)
    st.caption("Model labels detected: " + ", ".join(map(str, label_list)))
except Exception:
    pass
