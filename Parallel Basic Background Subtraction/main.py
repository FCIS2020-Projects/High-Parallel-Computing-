from mpi4py import MPI
import numpy as np
from scipy import misc
import io_helper
import cv2

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

frames = []
num_of_frames = 0
if rank == 0:
    # frames = io_helper.read_video_frames('highway.mp4', 40)
    frames = io_helper.read_frames('BackGround', gray=True)
    num_of_frames = len(frames)
    frames = np.array_split(frames, size)

scat = comm.scatter(frames, root=0)

# padding
max_scattered_count = comm.reduce(len(scat), op=MPI.MAX, root=0)
max_scattered_count = comm.bcast(max_scattered_count, root=0)
num_of_pads = max_scattered_count - len(scat)
# print(rank, " ", scat.shape)
for i in range(num_of_pads):
    scat = np.append(scat, [np.zeros(scat[0].shape)], axis=0)
# print(rank, " ", scat.shape)

local_sum = np.sum(scat, axis=0)
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
if rank == 0:
    mean = global_sum / num_of_frames
    mean = mean.astype(np.uint8)
    # cv2.imwrite('background.jpg', mean)
    # io_helper.subtract_background_from_video('highway.mp4', mean, 64, gray=False)
    misc.imsave('background.jpg', mean)
    io_helper.subtract_background_from_frames('BackGround', mean, 64, gray=True)
