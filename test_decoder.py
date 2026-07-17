import io
import sys
import numpy as np
import cv2
from recognition import _read_image, compute_embedding_from_stream

# create a simple PNG image in memory
img = np.zeros((100,100,3), dtype=np.uint8)
img[:] = (50,150,200)
ret, buf = cv2.imencode('.png', img)
if not ret:
    print('encode failed', file=sys.stderr)
    sys.exit(1)

stream = io.BytesIO(buf.tobytes())
img_out = _read_image(stream)
print('read_image', type(img_out), None if img_out is None else img_out.shape)

stream.seek(0)
emb = compute_embedding_from_stream(stream)
if isinstance(emb, dict) and emb.get('error'):
    print('embedding error:', emb.get('error'))
else:
    print('embedding length', len(emb))
