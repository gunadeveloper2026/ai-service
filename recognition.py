import io
import json
import numpy as np
import cv2
from PIL import Image


def _read_image(stream):
    stream.seek(0)
    data = stream.read()
    if not data:
        raise ValueError('Empty image stream')
    
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    try:
        image = Image.open(io.BytesIO(data)).convert('RGB')
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f'Unable to decode image: {e}')


def _extract_face_embedding(img, size=(160, 160), dim=128):
    """Simple face embedding: resize to grayscale, normalize, flatten."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize to standard size
        resized = cv2.resize(gray, size)
        # Flatten
        vec = resized.flatten().astype(np.float32)
        # Normalize
        vec = (vec - vec.mean()) / (vec.std() + 1e-9)
        # Project to embedding dimension
        if vec.size >= dim:
            emb = vec[:dim]
        else:
            emb = np.pad(vec, (0, dim - vec.size), 'constant')
        # L2 normalize
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb.tolist()
    except Exception as e:
        raise ValueError(f'Failed to extract embedding: {e}')


def compute_embedding_from_stream(stream, **kwargs):
    """Extract face embedding from image stream."""
    img = _read_image(stream)
    return _extract_face_embedding(img)


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
    """Find top-k matches from gallery."""
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
    Extract embedding for uploaded image and optionally match against gallery.
    
    - `stream`: file-like image stream
    - `gallery`: JSON string or list of entries with `embedding` vectors
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
