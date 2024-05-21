# Video Generator 
import itertools
import numpy as np
import cv2


IMG_WIDTH = 224 
IMG_HEIGHT = 224
NO_OF_CATEGORIES = 25

def load_videos(paths_list, max_frames, resize=(IMG_WIDTH, IMG_HEIGHT), start=0):
    paths = itertools.cycle(paths_list)
    while True:
        try:
            path, class_ = next(paths)
        except:
            break

        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Error")
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame/255)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()

        frames = np.expand_dims(frames, axis=0)
        cate = [1 if c == class_ else 0 for c in range(NO_OF_CATEGORIES)]
        cate = np.expand_dims(cate, axis=0)

        yield frames, cate