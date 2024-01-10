"""
Sobel edge detection makes use of convolution to determine where edges exist in a grayscale image.
In particular, Sobel edge detection makes use of two kernels Hx and Hy, given by the matrices
Hx = [ [−1, 0, 1] [− 2, 0, 2] [-1, 0, 1] ] and Hy = [−1, −2, −1] [0, 0, 0] [1, 2, 1] ].

For a given image X, the Sobel edge detection image Y is given by
Y[n, m] = sqrt((Hx * X)[n, m]^2 + (Hy * X)[n, m]^2)

The goal of this exercise is to implement this edge detection algorithm. We will ignore edge effects.
That is, for any given input X (say, 100 by 200 pixels),
the output Y is 2 pixels shorter and narrower than the input (that is, 98 by 198 pixels).

Your task is to implement the function performSobelEdgeDetection that produces a grayscale image that is
exactly 2 pixels narrower and 2 pixels shorter than the input,
and that represents the result of performing Sobel edge detection on the input image.
"""
import numpy as np
from PIL import Image


class GrayImage:
    """
    Class to store gray-scale images
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = np.zeros((height, width))


def read_pgm(filename):
    """
    Function to read PGM images
    """
    image = Image.open(filename)
    image = image.convert('L')
    width, height = image.size
    img = GrayImage(width, height)
    img.data = np.array(image)
    return img


def write_pgm(filename, image):
    """
    Function to write PGM images
    """
    im = Image.fromarray(image.data)
    im.save(filename)


def convolve_image(kernel, image):
    """
    This function applied a 2D convolution to an image using the given kernel.
    :param kernel: The image to be convolved.
    :param image: The kernel to be used.
    :return: The convolved image squared.
    """
    # Here we calculate the output image dimensions.
    output_height = image.height - kernel.height + 1
    output_width = image.width - kernel.width + 1

    # Here we create the output image object.
    output = GrayImage(output_width, output_height)
    # The first two loops are going to loop over all the values we need to calculate for out output image.
    for n in range(output_height):
        for m in range(output_width):
            # The next two loops are going to loop over all the values we
            # need to calculate for each pixel in the output image.
            for i in range(kernel.height):
                for j in range(kernel.width):
                    output.data[n][m] += int(np.multiply(kernel.data[i][j], image.data[i + n][j + m]))

    # Here we square the output image as this is part of the formula for Sobel edge detection. This is to help with
    # the computing efficiency of the program.
    output.data = np.square(output.data)
    return output


def perform_sobel_edge_detection(image):
    """
    This function performs Sobel edge detection on a given image.
    Hx = [ [−1, 0, 1] [− 2, 0, 2] [-1, 0, 1] ] and Hy = [−1, −2, −1] [0, 0, 0] [1, 2, 1] ].
    :param image: The image to apply the Sobel edge detection to.
    :return: The image after Sobel edge detection has been applied.
    """
    # Here we create the two kernels we need to apply Sobel edge detection.
    Hx = GrayImage(3, 3)
    Hx.data = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Hy = GrayImage(3, 3)
    Hy.data = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Here we convolve the image with the two kernels.
    convolved_Image_Hx = convolve_image(Hx, image)
    convolved_Image_Hy = convolve_image(Hy, image)

    # Here we create the output image object.
    output = GrayImage(convolved_Image_Hx.width, convolved_Image_Hx.height)

    # Here we calculate the output image. Hx and Hy are already squared, so we just need to add them together and then
    # square root the result.
    output.data = np.sqrt(convolved_Image_Hx.data + convolved_Image_Hy.data)

    return output


def main():
    filename = input().strip()

    # read input image
    image = read_pgm(filename)

    # process image
    output = perform_sobel_edge_detection(image)
    output.data = output.data.astype(np.int32)

    # generate output
    total = int(np.sum(output.data))
    print(total)
    write_pgm("output.pgm", output)


if __name__ == "__main__":
    main()
