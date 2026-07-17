import io
import numpy as np
import cv2
from recognition import recognize_face

# create a synthetic face-like image (random but reproducible)
arr = (np.linspace(0,255,160*160).reshape((160,160))).astype('uint8')
img = cv2.merge([arr, arr, arr])
# encode to jpeg bytes
_, buf = cv2.imencode('.jpg', img)
stream = io.BytesIO(buf.tobytes())

res = recognize_face(stream, gallery=[{'id':'u1','embedding': [0.0]*128}, {'id':'u2','embedding':[1.0]*128}])
print('Result:', res)
