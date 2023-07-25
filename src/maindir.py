"""
Please read the copyright notice located on the readme file (README.md).    
"""
import src.Functions as Fu
import numpy as np
from scipy import special


def PCE(C, shift_range=[0,0], squaresize=11):
    """
    Computes Peak-to-Correlation Energy (PCE) obtained from correlation surface 
    restricted to possible shifts due to cropping. In this implementation of 
    PCE, it carries the sign of the peak (i.e. PCE can be negative)

    Parameters
    ----------
    C : numpy.ndarray('float32')
        cross-correlation surface calculated by function 'crosscorr'
    shift_range : list
        maximum shift is from [0,0] to [shift_range]
    squaresize : int
        removes the peak neighborhood of size (squaresize x squaresize)
        
    Returns
    -------
    dict
        A dictionary with the following items:
            PCE : peak-to-correlation energy
            PeakLocation : location of the primary peak, [0 0] when correlated
                    signals are not shifted to each other
            pvalue : probability of obtaining peakheight or higher (under
                    Gaussian assumption)
            P_FA : probability of false alarm (increases with increasing range
                    of admissible shifts (shift_range)
    dict
        A dictionary similar to the first output but for test under assumption
        of no cropping (i.e. equal to 'Out0,_ = PCE(C)')
        
    Example
    -------
    Out, Out0 = PCE(crosscorr(Noise1, Noise2), size(Noise)-1);
    C = crosscorr(Noise1,Noise2); Out0,_ = PCE(C)
    
    Note: 'Out0.PCE == Out.PCE' and 'Out.P_FA == Out.pvalue' when no shifts are considered
    
    """

    if any(np.greater_equal(shift_range,C.shape)):
        shift_range = min(shift_range,C.shape-1)   # all possible shift in at least one dimension

    shift_range = np.array(shift_range)
    Out = dict(PCE=[], pvalue=[], PeakLocation=[], peakheight=[], P_FA=[], log10P_FA=[])

    if not C.any():            # the case when cross-correlation C has zero energy (see crosscor2)
        Out['PCE'] = 0
        Out['pvalue'] = 1
        Out['PeakLocation'] = [0,0]
        return

    Cinrange = C[-1-shift_range[0]:,-1-shift_range[1]:]  	# C[-1,-1] location corresponds to no shift of the first matrix argument of 'crosscor2'
    [max_cc, imax] = np.max(Cinrange.flatten()), np.argmax(Cinrange.flatten())
    [ypeak, xpeak] = np.unravel_index(imax,Cinrange.shape)[0], np.unravel_index(imax,Cinrange.shape)[1]
    Out['peakheight'] = Cinrange[ypeak,xpeak]
    del Cinrange
    Out['PeakLocation'] = [shift_range[0]-ypeak, shift_range[1]-xpeak]

    C_without_peak = _RemoveNeighborhood(C,
                                         np.array(C.shape)-Out['PeakLocation'],
                                         squaresize)
    correl = C[-1,-1];        del C

    # signed PCE, peak-to-correlation energy
    PCE_energy = np.mean(C_without_peak*C_without_peak)
    Out['PCE'] = (Out['peakheight']**2)/PCE_energy * np.sign(Out['peakheight'])

    # p-value
    Out['pvalue'] = 1/2*special.erfc(Out['peakheight']/np.sqrt(PCE_energy)/np.sqrt(2))     # under simplifying assumption that C are samples from Gaussian pdf
    [Out['P_FA'], Out['log10P_FA']] = _FAfromPCE(Out['PCE'], np.prod(shift_range+1))

    Out0 = dict(PCE=[], P_FA=[], log10P_FA=[])
    Out0['PCE'] = (correl**2)/PCE_energy
    Out0['P_FA'], Out0['log10P_FA'] = _FAfromPCE(Out0['PCE'],1)
    return Out, Out0
# ----------------------------------------

def _RemoveNeighborhood(X,x,ssize):
    # Remove a 2-D neighborhood around x=[x1,x2] from matrix X and output a 1-D vector Y
    # ssize     square neighborhood has size (ssize x ssize) square
    [M,N] = X.shape
    radius = (ssize-1)/2
    X = np.roll(X,[np.int(radius-x[0]),np.int(radius-x[1])], axis=[0,1]) 
    Y = X[ssize:,:ssize];   Y = Y.flatten()
    Y = np.concatenate([Y, X.flatten()[int(M*ssize):]], axis=0)
    return Y

def _FAfromPCE(pce,search_space):
    # Calculates false alarm probability from having peak-to-cross-correlation (PCE) measure of the peak
    # pce           PCE measure obtained from PCE.m
    # seach_space   number of correlation samples from which the maximum is taken
    #  USAGE:   FA = FAfromPCE(31.5,32*32);

    [p,logp] = Fu.Qfunction(np.sign(pce)*np.sqrt(np.abs(pce)))
    if pce<50:
        FA = np.power(1-(1-p),search_space)
    else:
        FA = search_space*p                # an approximation

    if FA==0:
        FA = search_space*p
        log10FA = np.log10(search_space)+logp*np.log10(np.exp(1))
    else:
        log10FA = np.log10(FA)

    return FA, log10FA
