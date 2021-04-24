import numpy as np

def awgn(x, snr):
    N = x.size
    Ps = np.sum(x ** 2 / N)

    Psdb = 10 * np.log10(Ps)
    Pn = Psdb - snr
    noise = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, x.shape)
    return x + noise


if __name__=='__main__':
    x = np.ones(shape=(100))
    print(x)
    print(awgn(x, 0))