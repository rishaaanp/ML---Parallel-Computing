import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Function: Apply convolution to a row
def process_row(args):

    gray, kernel, row = args
    width = gray.shape[1]

    result_row = []

    for j in range(1, width-1):

        region = gray[row-1:row+2, j-1:j+2]
        value = np.sum(region * kernel)

        result_row.append(value)

    return result_row

# Sequential Filter
def sequential_filter(img, kernel):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    output = np.zeros_like(gray)

    for i in range(1, gray.shape[0]-1):

        for j in range(1, gray.shape[1]-1):

            region = gray[i-1:i+2, j-1:j+2]

            output[i, j] = np.sum(region * kernel)

    return output

# Parallel Filter
def parallel_filter(img, kernel):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    rows = gray.shape[0]

    args = [(gray, kernel, i) for i in range(1, rows-1)]

    with Pool() as p:

        result = p.map(process_row, args)

    output = np.zeros_like(gray)

    for i, row in enumerate(result):

        output[i+1, 1:-1] = row

    return output

# Main Program
if __name__ == "__main__":

    # Load Image
    image = cv2.imread("image.jpg")

    if image is None:
        print("Error loading image")
        exit()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Image loaded successfully")
    print("Image shape:", image.shape)

    # Kernels
    blur_kernel = np.ones((3,3)) / 9

    sharpen_kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    sobel_kernel = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])

    # Sequential Blur
    start = time.time()

    blur_seq = sequential_filter(image, blur_kernel)

    seq_time = time.time() - start

    print("Sequential Blur Time:", seq_time)

    # Parallel Blur
    start = time.time()

    blur_par = parallel_filter(image, blur_kernel)

    par_time = time.time() - start

    print("Parallel Blur Time:", par_time)

    # Speedup
    speedup = seq_time / par_time

    print("Speedup:", speedup)

    # Other Filters (Parallel)
    sharpen_img = parallel_filter(image, sharpen_kernel)

    edge_img = parallel_filter(image, sobel_kernel)

    # Save Output Images
    cv2.imwrite("blur.jpg", blur_par)
    cv2.imwrite("sharpen.jpg", sharpen_img)
    cv2.imwrite("edges.jpg", edge_img)

    print("Processed images saved.")

    # Display Images
    plt.figure(figsize=(12,6))

    plt.subplot(1,3,1)
    plt.title("Blur")
    plt.imshow(blur_par, cmap='gray')
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Sharpen")
    plt.imshow(sharpen_img, cmap='gray')
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Edge Detection")
    plt.imshow(edge_img, cmap='gray')
    plt.axis("off")

    plt.show()
