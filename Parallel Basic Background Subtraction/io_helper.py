from scipy import misc
import os
import cv2
import numpy as np


def read_frames(path, gray=False):
    frames = []
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        frame = misc.imread(img_path, gray)
        frames.append(frame)
    return frames


def read_video_frames(path, num_of_frames):
    frames = []
    video = cv2.VideoCapture(path)
    while True:
        ret, frame = video.read()
        if ret and num_of_frames > 0:
            frames.append(frame)
            num_of_frames -= 1
        else:
            break
    video.release()
    return frames


def subtract_background_from_frames(path, background, threshold, gray=False):
    if not os.path.exists('output'):
        os.mkdir('output')
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        frame = misc.imread(img_path, gray)
        frame = np.array((abs(background.astype(int) - frame.astype(int)) > threshold) * 255).astype(np.uint8)
        out_path = os.path.join('output', img.replace('in', 'out'))
        misc.imsave(out_path, frame)


def subtract_background_from_video(path, background, threshold, gray=False):
    if gray:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    video = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720), not gray)
    while True:
        ret, frame = video.read()
        if ret:
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.array((abs(background.astype(int) - frame.astype(int)) > threshold) * 255).astype(np.uint8)
            out.write(frame)
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    video.release()
    out.release()
