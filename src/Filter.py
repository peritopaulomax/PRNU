"""
Please read the copyright notice located on the readme file (README.md).  
"""
import cv2 as cv
import numpy as np
from scipy import signal
import src.Functions as Fu


def Threshold(y, t):
    """
    Applies max(0,y-t).

    Parameters
    ----------
    y : numpy.ndarray('float32')
        Array of Wavelet coefficients 
    t : float32
        Variance of PRNU

    Returns
    -------
    numpy.ndarray('float32')
        The thresholded Wavelet coefficients for a later filtering

    """
    res = y - t
    x = np.maximum(res, 0.)
    return x


def WaveNoise(coef, NoiseVar):
    """
    Applies Wiener-like filter in Wavelet Domain (residual filtering).
    
    Models each detail wavelet coefficient as conditional Gaussian random 
    variable and use four square NxN moving windows, N in [3,5,7,9], to 
    estimate the variance of noise-free image for each wavelet coefficient. 
    Then it applies a Wiener-type denoising filter to the coefficients.

    Parameters
    ----------
    coef : numpy.ndarray('float32')
        Wavelet detailed coefficient at certain level 
    NoiseVar : float32
        Variance of the additive noise (PRNU)

    Returns
    -------
    numpy.ndarray('float32')
        Attenuated (filtered) Wavelet coefficient
    numpy.ndarray('float32')
        Finall estimated variances for each Wavelet coefficient
    """

    tc = np.power(coef, 2)
    coefVar = Threshold(
        signal.fftconvolve(tc, np.ones([3, 3], dtype=float32) / (3 * 3), mode='same'),
        NoiseVar)

    for w in range(5, 9 + 1, 2):
        EstVar = Threshold(
            signal.fftconvolve(tc, np.ones([w, w], dtype=float32) / (w * w), mode='same'),
            NoiseVar)
        coefVar = np.minimum(coefVar, EstVar)

    # Wiener filter like attenuation
    tc = np.multiply(coef, np.divide(NoiseVar, coefVar + NoiseVar))

    return tc, coefVar

'''
def WaveFilter(coef, NoiseVar):
    """
    Applies Wiener-like filter in Wavelet Domain (image filtering).
    
    Models each detail wavelet coefficient as conditional Gaussian random 
    variable and use four square NxN moving windows, N in [3,5,7,9], to 
    estimate the variance of noise-free image for each wavelet coefficient. 
    Then it applies a Wiener-type denoising filter to the coefficients.

    Parameters
    ----------
    coef : numpy.ndarray('float32')
        Wavelet detailed coefficient at certain level 
    NoiseVar : float32
        Variance of the additive noise

    Returns
    -------
    numpy.ndarray('float32')
        Attenuated (filtered) Wavelet coefficient
    numpy.ndarray('float32')
        Finall estimated variances for each Wavelet coefficient
    """

    tc = np.power(coef, 2)
    coefVar = Threshold(
        signal.fftconvolve(np.ones([3, 3]) / (3. * 3.), tc, mode='valid'),
        NoiseVar);

    for w in range(5, 9 + 1, 2):
        EstVar = Threshold(
            signal.fftconvolve(np.ones([w, w]) / (w * w), tc, mode='valid'),
            NoiseVar)
        coefVar = min(coefVar, EstVar)

    # Wiener filter like attenuation
    tc = np.multiply(coef, np.divide(coefVar, coefVar + NoiseVar))

    return tc, coefVar
'''

def NoiseExtractFromImage(image, sigma=3.0, color=False, noZM=False):
    """
    Estimates PRNU from one image

    Parameters
    ----------
    image : str or numpy.ndarray('uint8')
        either test image filename or numpy matrix of image
    sigma : float32
        std of noise to be used for identicication
        (recomended value between 2 and 3)
    color : bool
        for an RGB image, whether to extract noise for the three channels 
        separately (default: False)
    noZM
        whether to apply zero-mean to the extracted (filtered) noise

    Returns
    -------
    numpy.ndarray('float32')
        extracted noise from the input image, a rough estimate of PRNU fingerprint
        
    Example
    -------
    noise = NoiseExtractFromImage('DSC00123.JPG',2);
    
    Reference
    ---------
    [1] M. Goljan, T. Filler, and J. Fridrich. Large Scale Test of Sensor
    Fingerprint Camera Identification. In N.D. Memon and E.J. Delp and P.W. Wong and
    J. Dittmann, editors, Proc. of SPIE, Electronic Imaging, Media Forensics and
    Security XI, volume 7254, pages # 0I010I12, January 2009.

    """
    
    # ----- Parameters ----- #
    L = 4  # number of wavelet decomposition levels (between 2-5 as well)
    if isinstance(image, str):
        X = cv.imread(image)
        if np.ndim(X)==3: X = X[:,:,::-1] # BGR2RGB
    else:
        X = image
        del image

    M0, N0, three = X.shape
    if X.dtype == 'uint8':
        # convert to [0,255]
        X = X.astype(float)
    elif X.dtype == 'uint16':
        X = X.astype(float) / 65535 * 255

    qmf = [ 	.230377813309,	.714846570553, .630880767930, -.027983769417,
           -.187034811719,	.030841381836, .032883011667, -.010597401785]
    qmf /= np.linalg.norm(qmf)
    
    if three != 3:
        Noise = Fu.NoiseExtract(X, qmf, sigma, L)
    else:
        Noise = np.zeros(X.shape)
        for j in range(3):
            Noise[:, :, j] = Fu.NoiseExtract(X[:, :, j], qmf, sigma, L)
        if not color:
            Noise = Fu.rgb2gray1(Noise)
    if noZM:
        print('not removing the linear pattern')
    else:
        Noise, _ = Fu.ZeroMeanTotal(Noise)

    #Noise = Noise.astype(float)

    return Noise

#%% ----- 'mdwt' mex code ported to python -----#
def mdwt(x, h, L):
    """
    multi-level Discrete Wavelet Transform, implemented similar to 
    Rice Wavelet Toolbox (https://www.ece.rice.edu/dsp/software/rwt.shtml)
    
    Parameters
    ----------
    X : numpy.ndarray('float32')
        2D input image
    h : list
        db4 (D8) decomposition lowpass filter
    L : Int
        Number of levels for DWT decomposition
    
    Returns
    -------
    numpy.ndarray('float32')
        input image in DWT domain      
    
    """
    
    isint = lambda x: x % 1 == 0

    m, n = x.shape[0], x.shape[1]  
    if m > 1:
        mtest = m / (2.**L)
        if not isint(mtest):
            raise(ValueError("Number of rows in input image must be of size m*2^(L)"))
    if n > 1:
        ntest = n / (2.**L)
        if not isint(ntest):
            raise(ValueError("Number of columns in input image must be of size n*2^(L)"))
    
    
    # -- internal --
    
    def _fpsconv(x_in, lx, h0, h1, lhm1, x_outl, x_outh):
        # circular-like padding
        x_in[lx:lx+lhm1] = x_in[:lhm1]
        #
        tmp = np.convolve(x_in[:lx+lhm1],h0)
        x_outl[:lx//2]= tmp[lhm1:-lhm1-1:2]
        tmp = np.convolve(x_in[:lx+lhm1],h1)
        x_outh[:lx//2]= tmp[lhm1:-lhm1-1:2]
        '''
        # or (as in the C++ implementation):
        ind = 0
        for i in range(0,lx,2):
            x_outl[ind] = np.dot( x_in[i:i+lhm1+1], np.flip(h0) )
            x_outh[ind] = np.dot( x_in[i:i+lhm1+1], np.flip(h1) )
            ind += 1
        '''
        return x_in, x_outl, x_outh
    
    def _MDWT(x, h, L):
        lh = len(h)
        _m, _n = x.shape[0], x.shape[1]  
        y = np.zeros([_m,_n], dtype=float32)
        
        xdummy  = np.zeros([max(_m,_n) + lh-1], dtype=float32)
        ydummyl = np.zeros([max(_m,_n)], dtype=float32)
        ydummyh = np.zeros([max(_m,_n)], dtype=float32)
        
        # analysis lowpass and highpass
        if _n == 1:
            _n = _m
            _m = 1
        
        h0 = np.flip(h)
        h1 = [h[i]*(-1)**(i+1) for i in range(lh)] 
        lhm1 = lh - 1
        actual_m = 2 * _m
        actual_n = 2 * _n
        
        # main loop
        for actual_L in range(1, L+1):
            if _m == 1:
                actual_m = 1
            else:
                actual_m = actual_m // 2
                r_o_a = actual_m // 2
            actual_n = actual_n // 2
            c_o_a = actual_n // 2
            
            # go by rows
            for ir in range(actual_m):# loop over rows
                # store in dummy variable
                if actual_L == 1:
                    xdummy[:actual_n] = x[ir, :actual_n]# from input
                else:
                    xdummy[:actual_n] = y[ir, :actual_n]# from LL of previous level
                # perform filtering lowpass and highpass
                xdummy, ydummyl, ydummyh = _fpsconv(xdummy, actual_n, h0, h1, lhm1, ydummyl, ydummyh)
                # restore dummy variables in matrices
                y[ir, :c_o_a     ] = ydummyl[:c_o_a]
                y[ir, c_o_a:2*c_o_a] = ydummyh[:c_o_a]
            
            
            if _m > 1: # in case of a 2D signal
                # go by columns
                for ic in range(actual_n):# loop over column
                    # store in dummy variables 
                    xdummy[:actual_m] = y[:actual_m, ic]
                    # perform filtering lowpass and highpass
                    xdummy, ydummyl, ydummyh = _fpsconv(xdummy, actual_m, h0, h1, lhm1, ydummyl, ydummyh)
                    # restore dummy variables in matrix
                    y[:r_o_a,      ic] = ydummyl[:r_o_a]
                    y[r_o_a:2*r_o_a, ic] = ydummyh[:r_o_a]
                    
        return y

    # --------------
    
    y = _MDWT(x, h, L)
    
    return y

#%% ----- 'midwt' mex code ported to python -----#
def midwt(y, h, L):
    """
    multi-level inverse Discrete Wavelet Transform, implemented similar to 
    Rice Wavelet Toolbox (https://www.ece.rice.edu/dsp/software/rwt.shtml)
    
    Parameters
    ----------
    y : numpy.ndarray('float32')
        2D matrix of image in multi-level DWT domain
    h : list
        db4 (D8) decomposition lowpass filter
    L : Int
        Number of levels for DWT decomposition
    
    Returns
    -------
    numpy.ndarray('float32')
        input image in DWT domain      
    
    """
    
    isint = lambda x: x % 1 == 0
    
    m, n = y.shape[0], y.shape[1]  
    if m > 1:
        mtest = m / (2.**L)
        if not isint(mtest):
            raise(ValueError("Number of rows in input image must be of size m*2^(L)"))
    if n > 1:
        ntest = n / (2.**L)
        if not isint(ntest):
            raise(ValueError("Number of columns in input image must be of size n*2^(L)"))
    
    # -- internal --
    def _bpsconv(x_out, lx, g0, g1, lhhm1, x_inl, x_inh):
        x_inl[:lhhm1] = x_inl[lx:lx+lhhm1]
        x_inh[:lhhm1] = x_inh[lx:lx+lhhm1]
        
        tmp = np.convolve(x_inl[:lx+lhhm1+1], g0[::2]) + \
                np.convolve(x_inh[:lx+lhhm1+1], g1[::2]);
        x_out[:2*lx:2] = tmp[lhhm1:-lhhm1-1]
        
        tmp = np.convolve(x_inl[:lx+lhhm1+1], g0[1::2]) + \
                np.convolve(x_inh[:lx+lhhm1+1], g1[1::2])
        x_out[1:2*lx:2] = tmp[lhhm1:-lhhm1-1]
        '''
        # or (as in the C++ implementation):
        ind = 0
        for i in range(lx):
            x_out[ind]   = np.dot(x_inl[i:i+lhhm1+1], np.flip(g0[::2])) + \
                            np.dot(x_inh[i:i+lhhm1+1], np.flip(g1[::2]))
            x_out[ind+1] = np.dot(x_inl[i:i+lhhm1+1], np.flip(g0[1::2])) + \
                                  np.dot(x_inh[i:i+lhhm1+1], np.flip(g1[1::2]))
            ind += 2
        '''
        return x_out
    
    def _MIDWT(y, h, L):
        lh = len(h)
        _m, _n = y.shape[0], y.shape[1]
        xdummy  = np.zeros([max(_m, _n)], dtype=float32)
        ydummyl = np.zeros([max(_m, _n)+lh//2-1], dtype=float32)
        ydummyh = np.zeros([max(_m, _n)+lh//2-1], dtype=float32)
        
        # synthesis lowpass and highpass
        if _n == 1:
            _n = _m
            _m = 1
        
        g0 = h
        g1 = [h[lh-i-1]*((-1)**i) for i in range(lh)] 
        #lhm1 = lh - 1
        lhhm1 = lh // 2 - 1
        
        # 2^L
        sample_f = 2**(L-1)
        
        actual_m = _m // sample_f if _m > 1 else 1
        actual_n = _n // sample_f
        
        x = y
        
        # main loop
        for actual_L in range(L,0,-1):
            r_o_a = actual_m // 2
            c_o_a = actual_n // 2
            
            # in case of a 2D signal
            if _m > 1:
                # go by columns
                for ic in range(actual_n):# loop over column
                    # store in dummy variables
                    ydummyl[lhhm1:lhhm1+r_o_a] = x[:r_o_a, ic]
                    ydummyh[lhhm1:lhhm1+r_o_a] = x[r_o_a:2*r_o_a, ic]
                    # perform filtering lowpass and highpass 
                    xdummy = _bpsconv(xdummy, r_o_a, g0, g1, lhhm1, ydummyl, ydummyh)
                    # restore dummy variables in matrix
                    x[:actual_m, ic] = xdummy[:actual_m]
            # go by rows
            for ir in range(actual_m):# loop over rows
                # store in dummy variable
                ydummyl[lhhm1:lhhm1+c_o_a] = x[ir, :c_o_a]
                ydummyh[lhhm1:lhhm1+c_o_a] = x[ir, c_o_a:2*c_o_a]
                # perform filtering lowpass and highpass
                xdummy = _bpsconv(xdummy, c_o_a, g0, g1, lhhm1, ydummyl, ydummyh);
                # restore dummy variables in matrices
                x[ir, :actual_n] = xdummy[:actual_n]
            
            actual_m = 1 if _m == 1 else actual_m * 2
            actual_n = actual_n * 2
        
        return x
    # --------------
    
    x = _MIDWT(y, h, L)
    
    return x

