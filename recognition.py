import io
import json
import numpy as np
import cv2

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False

try:
    from PIL import Image
except ImportError:
    Image = None


def _read_image(stream):
    stream.seek(0)
    data = stream.read()
    if not data:
        raise ValueError('Empty image stream')
    
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    if Image is not None:
        try:
            image = Image.open(io.BytesIO(data)).convert('RGB')
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"PIL decode failed: {e}")
            return None

    raise ValueError('Unable to decode image - cv2.imdecode and PIL both failed')


def _fallback_embedding(img, size=(160, 160), dim=128):
    # simple deterministic embedding: resize, convert to grayscale, normalize and project
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, size)
    vec = small.flatten().astype(np.float32)
    # normalize
    vec = (vec - vec.mean()) / (vec.std() + 1e-9)
    # reduce/expand to `dim`
    if vec.size >= dim:
        emb = vec[:dim]
    else:
        emb = np.pad(vec, (0, dim - vec.size), 'constant')
    # L2 normalize
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb.tolist()


def compute_embedding_from_stream(stream, model_name='Facenet', detector_backend='mtcnn'):
    img = _read_image(stream)
    if img is None:
        raise ValueError('Unable to decode image')
    
    if DEEPFACE_AVAILABLE:
        try:
            # DeepFace.represent accepts numpy arrays directly
            rep = DeepFace.represent(img, model_name=model_name, detector_backend=detector_backend, enforce_detection=False)
            if isinstance(rep, list) and len(rep) and isinstance(rep[0], dict):
                emb = np.array(rep[0].get('embedding', []), dtype=float)
            elif isinstance(rep, list) and len(rep):
                emb = np.array(rep[0], dtype=float)
            else:
                emb = np.array(rep, dtype=float)
            if emb.size == 0:
                raise ValueError('Empty embedding returned from DeepFace')
            return emb.tolist()
        except Exception as e:
            print(f"DeepFace error: {e}, falling back to simple embedding")
            return _fallback_embedding(img)
    else:
        return _fallback_embedding(img)


def _cosine_distance(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float('inf')
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return float('inf')
    cos_sim = num / den
    return float(1.0 - cos_sim)


def match_embedding(embedding, gallery, top_k=5):
    # gallery: list of dicts { 'id': ..., 'embedding': [...] }
    results = []
    for entry in gallery:
        vec = entry.get('embedding') or entry.get('vector')
        if not vec:
            continue
        dist = _cosine_distance(embedding, vec)
        results.append({'id': entry.get('id') or entry.get('userId') or entry.get('user'), 'distance': float(dist)})
    results.sort(key=lambda x: x['distance'])
    return results[:top_k]


def recognize_face(stream, gallery=None, threshold=0.35):
    """
    Compute embedding for uploaded image and optionally match against a provided gallery.

    - `stream`: file-like image stream
    - `gallery`: either a Python list or a JSON string representing a list of entries with `embedding` vectors
    Returns: dict with `embedding`, optional `matches`, `best`, and `matched` boolean.
    """
    try:
        embedding = compute_embedding_from_stream(stream)
    except Exception as e:
        return {'error': str(e)}

    out = {'embedding': embedding}

    if not gallery:
        return out

    if isinstance(gallery, str):
        try:
            gallery = json.loads(gallery)
        except Exception:
            gallery = []

    if not isinstance(gallery, list):
        gallery = []

    matches = match_embedding(embedding, gallery, top_k=5)
    best = matches[0] if matches else None
    matched = bool(best and best['distance'] <= threshold)
    out.update({'matches': matches, 'best': best, 'matched': matched})
    return out
