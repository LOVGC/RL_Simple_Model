from scipy.fft import fft, fftfreq, fftshift

def fourier_spectrum(signal, fs):
    """[summary] assume signal is a row vector

    Args:
        signal ([type] 1D numpy array): [description]
        fs ([type]): [description] sampling frequency of the signal

    Returns:
        freqs: numpy array of freqs with 0 freq centered 
        spectrum: complex-valued spectrum of the signal
    """
    
    y = fft(signal)
    spectrum = fftshift(y)

    n = signal.size # number of elements in x
    freqs = fftfreq(n, 1/fs)
    
    freqs = fftshift(freqs) # shift zero freq to center
    freqs = freqs.reshape((1, -1))

    return (freqs, spectrum)