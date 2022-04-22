import numpy as np
import itertools
from typing import Tuple, Optional
from PIL import Image, ImageOps
from matplotlib import cm

# construct basic 2-pixels permutations.
pix_instruct = np.array(list(itertools.combinations(range(4),2)))
ATOMIC_PXLS = np.zeros((len(pix_instruct), 4)).astype(np.int8)
for i, inst in enumerate(pix_instruct):
    ATOMIC_PXLS[i][inst] = 1
ATOMIC_PXLS = ATOMIC_PXLS.reshape(-1,2,2)



def load_img(img_path= "blkdrg.jpg"):
    rgb_img = Image.open(img_path)
    gray_img = ImageOps.grayscale(rgb_img)

    return np.array(gray_img)

def gray2binary(np_img:np.ndarray, threshold: Optional[int]= None):
    if not threshold:
        threshold = 150

    return (np_img >= threshold).astype(np.int8)

def gen_key(shape: Tuple[int,int]):
    rows, cols = shape
    rand_pattern = np.random.randint(0, len(ATOMIC_PXLS), shape)
    key = np.zeros((2*rows, 2*cols)).astype(np.int8)
    for row, col in itertools.product(range(rows), range(cols)):
        key[2*row:2*row+2, 2*col:2*col+2] = ATOMIC_PXLS[rand_pattern[row,col]]
    return key

def enc_bin_image(bin_image):
    rows, cols = bin_image.shape
    key = gen_key(bin_image.shape)
    enc = np.zeros(key.shape).astype(np.int8)
    for row, col in itertools.product(range(rows), range(cols)):
        key_patt = key[2*row:2*row+2, 2*col:2*col+2] 
        origin_pxl = bin_image[row, col]
        enc[2*row:2*row+2, 2*col:2*col+2] = origin_pxl*key_patt + (1-origin_pxl)*(1-key_patt)
    return key, enc


def decrypte(key, enc):
    return(np.bitwise_or(key,enc))



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    img = load_img()
    bin_img = gray2binary(img, 70)

    key, enc = enc_bin_image(bin_img)

    Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))

    fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
    axs[0].imshow(key, cmap='gray_r')
    axs[1].imshow(enc, cmap='gray_r')
    axs[2].imshow(decrypte(key, enc), cmap='gray_r')

    # plt.show()


    fig, axs = plt.subplots(5,5)
    axs = [element for row in axs for element in row]
    for i, ax in enumerate(axs):
        ax.imshow(gray2binary(img, i*10))
        ax.set_title(str(i*10))

    plt.show()


    # from IPython import embed; embed()