import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def getImagesFrom(folder_addr, extension='.png'):
    """
    Returns a list of images from a path, with the desired extension.
    
    Parameters
    ----------
    folder_addr : str
        Path to a target directory
    extension : str
        Extension of target image files, such as '.png', '.jpg', etc.

    Returns
    -------
    list
        The list of addresses of the images
    """
    
    files = sorted(os.listdir(folder_addr))
    if not folder_addr[-1]==os.sep: folder_addr += os.sep
    list_probeFiles = [i for i in files if i.endswith(extension)]
    for i in range(list_probeFiles.__len__()):
        list_probeFiles[i] = folder_addr + list_probeFiles[i]
    return list_probeFiles


'''
The following lines of code are mainly from:
    [https://matplotlib.org/examples/mplot3d/surface3d_demo.html]
'''
def mesh(Z):
    """
    Similar to MATLAB 'mesh' function, can show peak of cross-correlation plane.
    
    Parameters
    ----------
    Z : numpy.ndarray('float32')
        2D matrix of cross-correlation

    Returns
    -------
    <nothing>
    
    """
    Z = np.squeeze(Z)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    M, N = Z.shape
    X = np.arange(0, N, 1)
    Y = np.arange(0, M, 1)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(Z.min()*.99, Z.max()*1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
