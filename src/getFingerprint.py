"""
Please read the copyright notice located on the readme file (README.md).    
"""
import src.Functions as Fu 
import cv2 as cv
import numpy as np

def getFingerprint(Images, sigma=3., fromHowMany=-1):
    """
     Extracts and averages noise from all images and outputs a camera 
     fingerprint

    Parameters
    ----------
    Images : list
        List of color images to process. They have to be from the same camera
        and the same size and orientation.
    sigma : float32
        Standard deviation of the expected noise (PRNU)

    Returns
    -------
    numpy.ndarray('float32')
        3D matrix of reference pattern - estimate of PRNU (in the output file)
    dict
         Dictionary of Linear Pattern data
         
    -------------------------------------------------------------------------
    [1] M. Goljan, T. Filler, and J. Fridrich. Large Scale Test of Sensor
    Fingerprint Camera Identification. In N.D. Memon and E.J. Delp and P.W. 
    Wong and J. Dittmann, editors, Proc. of SPIE, Electronic Imaging, Media 
    Forensics and Security XI, volume 7254, pages % 0I010I12, January 2009.
    -------------------------------------------------------------------------
    """

    database_size = Images.__len__() if fromHowMany==-1 else fromHowMany; del fromHowMany    # Number of the images
    if database_size==0: raise ValueError('No images of specified type in the directory.')
    ###  Parameters used in denoising filter
    L = 4 #  number of decomposition levels
    qmf = [ 	.230377813309,	.714846570553, .630880767930, -.027983769417,
           -.187034811719,	.030841381836, .032883011667, -.010597401785]
    qmf /= np.linalg.norm(qmf)
    
    t = 0
    ImagesinRP = []
    for i in range(database_size):
        Fu.SeeProgress(i),
        im = Images[i]
        X = cv.imread(im);
        if np.ndim(X)==3: X = X[:,:,::-1] # BGR to RGB
        X = _double255(X)
        if t == 0:
            M,N,three=X.shape
            if three == 1:
                continue # only color images will be processed
            ###  Initialize sums
            RPsum = np.zeros([M,N,3],dtype='single')
            NN = np.zeros([M,N,3],dtype='single') # number of additions to each pixel for RPsum
        else:
            s = X.shape
            if X.ndim != 3:
                print('Not a color image - skipped.\n')
                continue # only color images will be used
            if set([M,N,three]) != set(X.shape):
                print('\n Skipping image %(im)s of size %(s1)d x %(s2)d x %(s3)d \n' %{'im':im,'s1':s(1-1),'s2':s(2-1),'s3':s(3-1)})
                continue # only same size images will be used

        # The image will be the t-th image used for the reference pattern RP
        t=t+1  # counter of used images
        ImagesinRP.append(im)

        for j in range(3):
            ImNoise = np.single(Fu.NoiseExtract(X[:,:,j],qmf,sigma,L))
            Inten = np.multiply(Fu.IntenScale(X[:,:,j]),\
                                Fu.Saturation(X[:,:,j]))    # zeros for saturated pixels
            RPsum[:,:,j] = RPsum[:,:,j] + np.multiply(ImNoise,Inten)   	# weighted average of ImNoise (weighted by Inten)
            NN[:,:,j] = NN[:,:,j] + np.power(Inten,2)


    del ImNoise, Inten, X
    if t==0: raise ValueError('None of the images was color image in landscape orientation.')
    RP = np.divide(RPsum, NN + 1)
    # Remove linear pattern and keep its parameters
    RP, LP = Fu.ZeroMeanTotal(RP)
    return RP, LP, ImagesinRP

### FUNCTIONS ##
def _double255(X):
    # In MATLAB: convert to 'double' ranging from 0 to 255
    # Here in this Python implementation we convert it to 'single' (np.float32)
    X = X.astype(np.single)
    return X
