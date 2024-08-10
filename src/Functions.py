"""
Please read the copyright notice located on the readme file (README.md).    
"""
import numpy as np
from scipy import special
import src.Filter as Ft


def crosscorr(array1, array2):
    """
    Computes 2D cross-correlation of two 2D arrays.
    
    Parameters
    ----------
    array1 : numpy.ndarray
        first 2D matrix
    array2: numpy.ndarray
        second 2D matrix
    
    Returns
    ------
    numpy.ndarray('float64')
        2D cross-correlation matrix

    """ 

    array1 = array1.astype(np.double)
    array2 = array2.astype(np.double)
    array1 = array1 - array1.mean()
    array2 = array2 - array2.mean()

    ############### End of filtering
    normalizator = np.sqrt(np.sum(np.power(array1,2))*np.sum(np.power(array2,2)))
    tilted_array2 = np.fliplr(array2);     del array2
    tilted_array2 = np.flipud(tilted_array2)
    TA = np.fft.fft2(tilted_array2);           del tilted_array2
    FA = np.fft.fft2(array1);                  del array1
    AC = np.multiply(FA, TA);                  del FA, TA

    if normalizator==0:
        ret = None
    else:
        ret = np.real(np.fft.ifft2(AC))/normalizator
    return ret

'''
def crosscorr(array1, array2):
    # function ret = crosscor2(array1, array2)
    # Computes 2D crosscorrelation of 2D arrays
    # Function returns DOUBLE type 2D array
    # No normalization applied

    array1 = array1.astype(np.double)
    array2 = array2.astype(np.double)
    array1 = array1 - array1.mean()
    array2 = array2 - array2.mean()

    ############### End of filtering
    tilted_array2 = np.fliplr(array2);     del array2
    tilted_array2 = np.flipud(tilted_array2)
    TA = np.fft.fft2(tilted_array2);           del tilted_array2
    FA = np.fft.fft2(array1);                  del array1
    FF = np.multiply(FA, TA);                  del FA, TA

    ret = np.real(np.fft.ifft2(FF))
    return ret
'''

def imcropmiddle(X, sizeout, preference='SE'):
    """
    Crops the middle portion of a given size.
    
    Parameters
    ----------
    x : numpy.ndarray
        2D or 3D image matrix
    sizeout: list
        size of the output image
    
    Returns
    ------
    numpy.ndarray
        cropped image

    """ 

    if sizeout.__len__() >2:
        sizeout = sizeout[:2]
    if np.ndim(X)==2: X = X[...,np.newaxis]
    M, N, three = X.shape
    sizeout = [min(M,sizeout[0]), min(N,sizeout[1])]
    # the cropped region is off center by 1/2 pixel
    if preference == 'NW':
        M0 = np.floor((M-sizeout[0])/2)
        M1 = M0+sizeout[0]
        N0 = np.floor((N-sizeout[1])/2)
        N1 = N0+sizeout[1]
    elif preference == 'SW':
        M0 = np.ceil((M-sizeout[0])/2)
        M1 = M0+sizeout[0]
        N0 = np.floor((N-sizeout[1])/2)
        N1 = N0+sizeout[1]
    elif preference == 'NE':
        M0 = np.floor((M-sizeout[0])/2)
        M1 = M0+sizeout[0]
        N0 = np.ceil((N-sizeout[1])/2)
        N1 = N0+sizeout[1]
    elif preference == 'SE':
        M0 = np.ceil((M-sizeout[0])/2)
        M1 = M0+sizeout[0]
        N0 = np.ceil((N-sizeout[1])/2)
        N1 = N0+sizeout[1]
    X = X[M0:M1+1,N0:N1+1,:]
    return X


def IntenScale(inp):
    """
    Scales input pixels to be used as a multiplicative model for PRNU detector.
    
    Parameters
    ----------
    x : numpy.ndarray('uint8')
        2D or 3D image matrix
    
    Returns
    ------
    numpy.ndarray('float32')
        Matrix of pixel intensities in to be used in a multiplicative model
        for PRNU.

    """

    T = 252.
    v = 6.
    out = np.exp(-1*np.power(inp-T,2)/v)

    out[inp < T] = inp[inp < T]/T
    
    return out


def LinearPattern(X):
    """
    Output column and row means from all 4 subsignals, subsampling by 2.
    
    Parameters
    ----------
    x : numpy.ndarray('float32')
        2D noise matrix
    
    Returns
    -------
    dict
        A dictionary with the following items:
            row means as LP.r11, LP.r12, LP.r21, LP.r22 (column vectors) 
            column means as LP.c11, LP.c12, LP.c21, LP.c22 (row vectors)
            
    numpy.ndarray('float32')
        The difference between input X and ZeroMean(X); i.e. X-output would be
        the zero-meaned version of X

    """

    M, N = X.shape
    me = X.mean()
    X = X-me

    #LP = {"r11":[],"c11":[],"r12":[],"c12":[],"r21":[],"c21":[],"r22":[],"c22":[],"me":[],"cm":[]}
    LP = dict(r11=[], c11=[], r12=[], c12=[], r21=[], c21=[], r22=[], c22=[], me=[], cm=[])
    LP['r11'] = np.mean(X[::2,::2],axis=1)
    LP['c11'] = np.mean(X[::2,::2],axis=0)
    cm11 = np.mean(X[::2,::2])
    LP['r12'] = np.mean(X[::2,1::2],axis=1)
    LP['c12'] = np.mean(X[::2,1::2],axis=0)
    cm12 = np.mean(X[::2,1::2]) # = -cm  Assuming mean2(X)==0
    LP['r21'] = np.mean(X[1::2,::2],axis=1)
    LP['c21'] = np.mean(X[1::2,::2],axis=0)
    cm21 = np.mean(X[1::2,::2]) # = -cm  Assuming mean2(X)==0
    LP['r22'] = np.mean(X[1::2,1::2],axis=1)
    LP['c22'] = np.mean(X[1::2,1::2],axis=0)
    cm22 = np.mean(X[1::2,1::2]) # = cm   Assuming mean2(X)==0
    LP['me'] = me
    LP['cm'] = [cm11,cm12,cm21,cm22]

    del X
    D = np.zeros([M,N],dtype=np.double)
    [aa,bb] = np.meshgrid(LP["c11"],LP["r11"],indexing='ij')
    D[::2,::2] = aa+bb+me-cm11
    [aa,bb] = np.meshgrid(LP["c12"],LP["r12"],indexing='ij')
    D[::2,1::2] = aa+bb+me-cm12
    [aa,bb] = np.meshgrid(LP["c21"],LP["r21"],indexing='ij')
    D[1::2,::2] = aa+bb+me-cm21
    [aa,bb] = np.meshgrid(LP["c22"],LP["r22"],indexing='ij')
    D[1::2,1::2] = aa+bb+me-cm22

    return LP, D


def NoiseExtract(Im,qmf,sigma,L):
    """
    Extracts noise signal that is locally Gaussian N(0,sigma^2)

    Parameters
    ----------
    Im : numpy.ndarray
        2D noisy image matrix
    qmf : list
        Scaling coefficients of an orthogonal wavelet filter
    sigma : float32
        std of noise to be used for identicication
        (recomended value between 2 and 3)
    L : int
        The number of wavelet decomposition levels. 
        Must match the number of levels of WavePRNU. 
        (Generally, L = 3 or 4 will give pretty good results because the
        majority of the noise is present only in the first two detail levels.)

    Returns
    -------
    numpy.ndarray('float32')
        extracted noise converted back to spatial domain
        
    Example
    -------
    Im = np.double(cv.imread('Lena_g.bmp')[...,::-1])        % read gray scale test image
    qmf = MakeONFilter('Daubechies',8)
    Image_noise = NoiseExtract(Im, qmf, 3., 4)
    
    Reference
    ---------
    [1] M. Goljan, T. Filler, and J. Fridrich. Large Scale Test of Sensor
    Fingerprint Camera Identification. In N.D. Memon and E.J. Delp and P.W. Wong and
    J. Dittmann, editors, Proc. of SPIE, Electronic Imaging, Media Forensics and
    Security XI, volume 7254, pages # 0I010I12, January 2009.

    """

    Im = Im.astype(np.float32)

    M, N = Im.shape
    m = 2**L
    # use padding with mirrored image content
    minpad=2    # minimum number of padded rows and columns as well
    nr = (np.ceil((M+minpad)/m)*m).astype(int);  nc = (np.ceil((N+minpad)/m)*m).astype(int)  # dimensions of the padded image (always pad 8 pixels or more)
    pr = np.ceil((nr-M)/2).astype(int)      # number of padded rows on the top
    prd= np.floor((nr-M)/2).astype(int)    # number of padded rows at the bottom
    pc = np.ceil((nc-N)/2).astype(int)      # number of padded columns on the left
    pcr= np.floor((nc-N)/2).astype(int)     # number of padded columns on the right
    Im = np.block([
        [ Im[pr-1::-1,pc-1::-1],       Im[pr-1::-1,:],       Im[pr-1::-1,N-1:N-pcr-1:-1]],
        [ Im[:,pc-1::-1],              Im,                   Im[:,N-1:N-pcr-1:-1] ],
        [ Im[M-1:M-prd-1:-1,pc-1::-1], Im[M-1:M-prd-1:-1,:], Im[M-1:M-prd-1:-1,N-1:N-pcr-1:-1] ]
                   ])

    # Precompute noise variance and initialize the output
    NoiseVar = sigma**2
    # Wavelet decomposition, without redudance
    wave_trans = Ft.mdwt(Im,qmf,L)
    # Extract the noise from the wavelet coefficients

    for i in range(1,L+1):

        # Horizontal noise extraction
        wave_trans[0:nr//2, nc//2:nc], _ = \
            Ft.WaveNoise(wave_trans[0:nr//2, nc//2:nc], NoiseVar)

        # Vertical noise extraction
        wave_trans[nr//2:nr, 0:nc//2], _ = \
            Ft.WaveNoise(wave_trans[nr//2:nr, 0:nc//2],NoiseVar)

        # Diagonal noise extraction
        wave_trans[nr//2:nr, nc//2:nc], _ = \
            Ft.WaveNoise(wave_trans[nr//2:nr, nc//2:nc], NoiseVar)

        nc = nc//2
        nr = nr//2
        
    # Last, coarest level noise extraction
    wave_trans[0:nr,0:nc] = 0

    # Inverse wavelet transform
    image_noise = Ft.midwt(wave_trans,qmf,L)

    # Crop to the original size
    image_noise = image_noise[pr:pr+M,pc:pc+N]
    return image_noise


def Qfunction(x):
    """
    Calculates probability that Gaussian variable N(0,1) takes value larger
    than x
    
    Parameters
    ----------
    x : float
        value to evalueate Q-function for

    Returns
    -------
    float
        probability that a variable from N(0,1) is larger than x
    float
        logQ

    """

    if x<37.5:
        Q = 1/2*special.erfc(x/np.sqrt(2))
        logQ = np.log(Q)
    else:
        Q = (1/(np.sqrt(2*np.pi)*x))*np.exp(-np.power(x,2)/2)
        logQ = -np.power(x,2)/2 - np.log(x)-1/2*np.log(2*np.pi)

    return Q, logQ


def rgb2gray1(X):
    """
    Converts RGB-like real data to gray-like output.
    
    Parameters
    ----------
    X : numpy.ndarray('float32')
        3D noise matrix from RGB image(s)

    Returns
    -------
    numpy.ndarray('float32')
        2D noise matrix in grayscale

    """
    
    datatype = X.dtype
    
    if X.shape[2]==1: G=X; return G
    M,N,three = X.shape
    X = X.reshape([M * N, three])

    # Calculate transformation matrix
    T = np.linalg.inv(np.array([[1.0, 0.956, 0.621],
                                [1.0, -0.272, -0.647],
                                [1.0, -1.106, 1.703]]))
    coef = T[0,:]
    G = np.reshape(np.matmul(X.astype(datatype), coef), [M, N])
    
    return G


def Saturation(X, gray=False):
    """
    Determines saturated pixels as those having a peak value (must be over 250)
    and a neighboring pixel of equal value
    
    Parameters
    ----------
    X : numpy.ndarray('float32')
        2D or 3D matrix of image with pixels in [0, 255]
    gray : bool
        Only for RGB input. If gray=true, then saturated pixels in output 
        (denoted as zeros) result from at least 2 saturated color channels 

    Returns
    -------
    numpy.ndarray('bool')
        binary matrix, 0 - saturated pixels
    
    """

    M = X.shape[0];  N = X.shape[1]
    if X.max()<=250:
        if not gray:
            SaturMap = np.ones(X.shape,dtype=np.bool)
        else:
            SaturMap = np.ones([M,N],dtype=np.bool)
        return SaturMap
    
    SaturMap = np.ones([M,N],dtype=int8)
    
    Xh = X - np.roll(X, 1, axis=1)
    Xv = X - np.roll(X, 1, axis=0)
    Satur = np.logical_and(np.logical_and(Xh, Xv), 
                np.logical_and(np.roll(Xh, -1, axis=1),np.roll(Xv, -1, axis=0)))

    if np.ndim(np.squeeze(X))==3:
        maxX = []
        for j in range(3):
            maxX.append(X[:,:,j].max())
            if maxX[j]>250:
                SaturMap[:,:,j] = np.logical_not(np.logical_and(X[:,:,j]==maxX[j],
                                                                np.logical_not(Satur[:,:,j])))
    elif np.ndim(np.squeeze(X))==2:
        maxX = X.max()
        SaturMap = np.logical_not(np.logical_and(X==maxX, np.logical_not(SaturMap)))
    else: raise ValueError('Invalid matrix dimensions')

    if gray and np.ndim(np.squeeze(X))==3:
        SaturMap = SaturMap[:,:,1]+SaturMap[:,:,3]+SaturMap[:,:,3]
        SaturMap[SaturMap>1] = 1

    return SaturMap


def SeeProgress(i):
    """
    SeeProgress(i) outputs i without performing carriage return
    This function is designed to be used in slow for-loops to show how the 
    calculations progress. If the first call in the loop is not with i=1, it's
    convenient to call SeeProgress(1) before the loop.
    """
    if i==1 | i==0 : print('\n               ')
    print('*   %(i)d   *' % {"i": i}, end="\r")


def WienerInDFT(ImNoise,sigma):
    """
    Removes periodical patterns (like the blockness) from input noise in 
    frequency domain
    
    Parameters
    ----------
    ImNoise : numpy.ndarray('float32')
        2D noise matrix extracted from one images or a camera reference pattern
    sigma : float32
        Standard deviation of the noise that we want not to exceed even locally
        in DFT domain
        
    Returns
    -------
    numpy.ndarray('float32')
        filtered image noise (or camera reference pattern) ... estimate of PRNU

    """
    M,N = ImNoise.shape

    F = np.fft.fft2(ImNoise);   del ImNoise
    Fmag = np.abs(np.real(F / np.sqrt(M*N)))       #  normalized magnitude

    NoiseVar = np.power(sigma, 2)
    Fmag1, _ = Ft.WaveNoise(Fmag, NoiseVar)

    fzero = np.where(Fmag==0); Fmag[fzero]=1; Fmag1[fzero]=0;    del fzero
    F = np.divide(np.multiply(F, Fmag1), Fmag)

    # inverse FFT transform
    NoiseClean = np.real(np.fft.ifft2(F))

    return NoiseClean


def ZeroMean(X, zType='CFA'):
    """
    Subtracts mean from all subsignals of the given type
    
    Parameters
    ----------
    X : numpy.ndarray('float32')
        2-D or 3-D noise matrix
    zType : str
        Zero-meaning type. One of the following 4 options: {'col', 'row', 'both', 'CFA'}

    Returns
    -------
    numpy.ndarray('float32')
        noise matrix after applying zero-mean
    dict
        dictionary including mean vectors in rows, columns, total mean, and 
        checkerboard mean
    
    Example
    -------
    Y,_ = ZeroMean(X,'col') ... Y will have all columns with mean 0.
    Y,_ = ZeroMean(X,'CFA') ... Y will have all columns, rows, and 4 types of
    odd/even pixels zero mean.
    
    """
    
    M, N, K = X.shape
    # initialize the output matrix and vectors
    Y = np.zeros(X.shape, dtype=X.dtype)
    row = np.zeros([M,K], dtype=X.dtype)
    col = np.zeros([K,N], dtype=X.dtype)
    cm=0
    
    # subtract mean from each color channel
    mu = []
    for j in range(K):
        mu.append(np.mean(X[:,:,j], axis=(0,1)))
        X[:,:,j] -= mu[j]
    
    for j in range(K): 
        row[:,j] = np.mean(X[:,:,j],axis=1)
        col[j,:] = np.mean(X[:,:,j],axis=0)

    if zType=='col':
        for j in range(K): Y[:,:,j] = X[:,:,j] - np.tile(col[j,:],(M,1))
    elif zType=='row':
        for j in range(K): Y[:,:,j] = X[:,:,j] - np.tile(row[:,j],(N,1)).transpose()
    elif zType=='both':
        for j in range(K): Y[:,:,j] = X[:,:,j] - np.tile(col[j,:],(M,1))
        for j in range(K): Y[:,:,j] = X[:,:,j] - np.tile(row[:,j],(N,1)).transpose()# equal to Y = ZeroMean(X,'row'); Y = ZeroMean(Y,'col');
    elif zType=='CFA':
        for j in range(K): Y[:,:,j] = X[:,:,j] - np.tile(col[j,:],(M,1))
        for j in range(K): Y[:,:,j] = X[:,:,j] - np.tile(row[:,j],(N,1)).transpose()    # equal to Y = ZeroMean(X,'both');
        for j in range(K):
            cm = np.mean(Y[::2, ::2, j], axis=(1,2))
            Y[::2, ::2, j]   -= cm
            Y[1::2, 1::2, j] -= cm
            Y[::2, 1::2, j]  += cm
            Y[1::2, ::2, j]  += cm
    else:
        raise(ValueError('Unknown type for zero-meaning.'))
        
    # Linear pattern data:
    LP = {}# dict(row=[], col=[], mu=[], checkerboard_mean=[])
    LP['row'] = row
    LP['col'] = col
    LP['mu'] = mu
    LP['checkerboard_mean'] = cm
    return Y, LP


def ZeroMeanTotal(X):
    """
    Subtracts mean from all black and all white subsets of columns and rows
    in a checkerboard pattern
    
    Parameters
    ----------
    X : numpy.ndarray('float32')
        2-D or 3-D noise matrix

    Returns
    -------
    numpy.ndarray('float32')
        noise matrix after applying ZeroMeanTotal
    dict
        dictionary of four dictionaries for the four subplanes, each includes
        mean vectors in rows, columns, total mean, and checkerboard mean.
    
    """
    dimExpanded = False
    if np.ndim(X)==2: X = X[...,np.newaxis];  dimExpanded = True      
    Y = np.zeros(X.shape, dtype=X.dtype)
    
    Z, LP11 = ZeroMean(X[::2, ::2, :],'both')
    Y[::2, ::2, :] = Z
    Z, LP12 = ZeroMean(X[::2, 1::2, :],'both')
    Y[::2, 1::2,:] = Z
    Z, LP21 = ZeroMean(X[1::2, ::2, :],'both')
    Y[1::2, ::2,:] = Z
    Z, LP22 = ZeroMean(X[1::2, 1::2, :],'both')
    Y[1::2, 1::2,:] = Z
    
    if dimExpanded: Y = np.squeeze(Y)
    
    LP = {}# dict(d11=[], d12=[], d21=[], d22=[])
    LP['d11'] = LP11
    LP['d12'] = LP12
    LP['d21'] = LP21
    LP['d22'] = LP22 
    
    return Y, LP

