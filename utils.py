import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Load model (GPU preferred)
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(img):
    faces = app.get(img)
    if faces:
        return faces[0].embedding  # 512D
    return None