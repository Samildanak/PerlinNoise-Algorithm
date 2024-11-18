import numpy as np
import matplotlib.pyplot as plot
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def perlin(x, y, seed=0):
    # Create a permutation tabler based on number of pixels
    # seed is the initial value we want to start with
    # we also use seed function to get same set of numbers
    # this helps to keep our perlin graph smooth
    np.random.seed(seed)
    ptable = np.arange(256, dtype=int)

    # Shuffle our numbers in the table
    np.random.shuffle(ptable)

    # Create a 2D array and then turn it one dimensional
    # so tgat we can apply our dot product interpolations easily
    ptable = np.stack([ptable, ptable]).flatten()

    # Grid coordinates
    xi, yi = x.astype(int), y.astype(int)

    # Distance vector coordiantes
    xg, yg = x - xi, y - yi

    # Apply fade function to distance coordinates
    xf, yf = fade(xg), fade(yg)
    
    # the gradient vector coordinates in the top left, top right, bottom left, bottom right
    n00 = gradient(ptable[ptable[xi] + yi], xg, yg)
    n01 = gradient(ptable[ptable[xi] + yi + 1], xg, yg - 1)
    n11 = gradient(ptable[ptable[xi + 1] + yi + 1], xg - 1, yg - 1)
    n10 = gradient(ptable[ptable[xi + 1] + yi], xg - 1, yg)

    # Apply linear interpolation i.e dot product to calculate average
    x1 = lerp(n00, n10, xf)
    x2 = lerp(n01, n11, xf)

    return lerp(x1, x2, yf)

def lerp(a, b, x):
    "Liner interpolation i.e dot product"
    return a + x * (b - a)

# smoothing function,
# the first derivative and second both are zero for this function

def fade(f):
    return 6 * f**5 - 15 * f**4 + 10 * f**3

#calculate the gradient vectors and dot product
def gradient(c, x, y):
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    gradient_co = vectors[c % 4]
    return gradient_co[:, :, 0] * x + gradient_co[:, :, 1] * y

def main():
    fig, ax = plot.subplots(subplot_kw={"projection" : "3d"})

    # create evenly spaced out numbers in a specified interval
    lin_array = np.linspace(1, 10, 500, endpoint=False)

    #create grid using linear 1d arrays
    x, y = np.meshgrid(lin_array, lin_array)

    # Plot the surface in 3D
    surf = ax.plot_surface(x, y, perlin(x, y, seed = 500), cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    plot.show()


if __name__ == "__main__":
    main()