from mpi4py import MPI
from scipy import misc
import numpy as np
import math

# -----------------------------------------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -----------------------------------------------------------------------------------------------------

def colorFreq(img, start, end, height):
    values = [0] * 256
    for i in range(int(start), int(end) + 1):
        for j in range(height):
             values[int(img[i, j])] += 1
    return values

# -----------------------------------------------------------------------------------------------------

def count_all(resultt):
    for i in range(1, len(resultt)):
        for j in range(256):
            resultt[0][j] += resultt[i][j]
    return resultt[0]

# ----------------------------------------------------------------------------------------------------

def prob(values):
    prob = [0] * 256
    for j in range(len(values)):
        prob[j] = values[j] / (img.shape[0] * img.shape[1])
    return prob

# ----------------------------------------------------------------------------------------------------

def cum_prob(prob):
    cum_prob = [0]*256
    for i in range(len(prob)):
        cum_prob[i] = cum_prob[i-1] + prob[i]

    for i in range(1, len(cum_prob)):
        cum_prob[i] *= 20
        cum_prob[i] = math.floor(cum_prob[i])

    maxi = max(cum_prob)
    mini = min(cum_prob)

    for i in range(len(cum_prob)):
            cum_prob[i] = (cum_prob[i] - mini) * (255 / (maxi - mini))

    return cum_prob

# -----------------------------------------------------------------------------------------------------

def equalize_image(img, start, end, height):
    my_cdf = colorFreq(img, start, end, height)
    return my_cdf

# -----------------------------------------------------------------------------------------------------


if rank == 0:
    img = misc.imread("image.png",True)

    print('read gray image')
    width = img.shape[0]
    print('width:',width)
    height = img.shape[1]
    print('height:',height)

    # --------------------------------------------------

    for i in range(size - 1):
        comm.send(img, dest=i + 1)
        print('send image for rank ',i+1)
        comm.send(width, dest=i + 1)
        print('send width for rank ',i+1)
        comm.send(height, dest=i + 1)
        print('send height for rank ',i+1)
    # --------------------------------------------------

    result = []
    prob_list = []
    cum_list = []

    for k in range(1, size):
        prob_list = comm.irecv(source=k)
        print('recv colorfreq for rank ', k)
        result.append(prob_list.wait())

    # --------------------------------------------------

    result = count_all(result)
    print('sum colorfreq for image')

    cum_list = cum_prob(result)

    print('calculate new pixels intensities ')

    # --------------------------------------------------

    for i in range(size - 1):
        print('send new  pixels intensities for rank ', i+1)
        comm.isend(cum_list, dest=i + 1)

    # --------------------------------------------------

    for k in range(1, size):
        imgs = comm.recv(source=k)

        rec_req = comm.irecv(source=k)
        rank = rec_req.wait()
        print('change  pixels intensities for rank ', rank)
        print('recv new part of image from rank ', rank)

        start = (width / (size - 1) * (rank - 1))
        end = (width / (size - 1) * rank) -1
        for i in range(int(start), int(end) + 1):
            for j in range(height):
                img[i, j] = imgs[i, j]

    # --------------------------------------------------

    print('save new equalize image ')
    misc.imsave("image.1.png", img)

    # --------------------------------------------------

else:
    img = comm.recv(source=0)
    width = comm.recv(source=0)
    height = comm.recv(source=0)
    print(width)
    print(height)
    # --------------------------------------------------

    start = (width / (size - 1) * (rank - 1))
    end = (width / (size - 1) * rank) - 1
    colorFreq = colorFreq(img, start, end, height)

    #---------------------------------------------------

    prob_l = prob(colorFreq)
    comm.isend(prob_l, dest=0)

    # --------------------------------------------------

    rec_req = comm.irecv(source=0)
    new_colore_list = rec_req.wait()

    # --------------------------------------------------


    img.setflags(write=1)
    for i in range(int(start), int(end) + 1):
        for j in range(height):
            img[i, j] = np.array(new_colore_list)[int(img[i, j])]

    comm.send(img, dest=0)
    comm.isend(rank, dest=0)

    # --------------------------------------------------
