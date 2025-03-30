import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import scipy as sp
from scipy import stats
import pywt
from sklearn import preprocessing
import math
from scipy import signal

class Preprocessing():

    def minmaxnorm(self, data):

        scaler = preprocessing.MinMaxScaler()
        scaler_minmax = scaler.fit(data)

        return scaler_minmax, scaler_minmax.transform(data)

    def znorm(self, data):
        scaler = preprocessing.StandardScaler()
        scaler_z = scaler.fit(data)

        return scaler_z, scaler_z.transform(data)

    def MEAN(self, data):
        M = np.mean(data, axis = 0)
        return data - M

    '''return type: numpy array'''
    def window_test(self, x, window):
        data = []
        for i in range(len(x) - window + 1):
            data.append(x[i:i + window])

        y = np.arange(len(x)) / (len(x) - 1)
        y = y.reshape(-1,1)
        y = y[window-1:]


        return np.array(data), y

    def window_traing(self, x, cutoff, window):
        data = []
        for i in range(len(x) - window + 1):
            data.append(x[i:i + window])

        y = np.arange(cutoff) / (cutoff - 1)
        y = y.reshape(-1,1)
        y = y[window-1:]

        if not len(x) == cutoff:
            y_cutoff = -1 * np.ones(shape = (len(x) - cutoff,1))
            y = np.concatenate((y, y_cutoff), axis = 0)

        return np.array(data), y

class Stat_process():

    def MSE(self, pre, tar):

        len = pre.shape[0]
        sum = 0

        for i in range(len):
            abs = pre[i] - tar[i]
            abs = (abs * abs) / len
            sum = abs + sum

        return sum


    def RMS(self, data):

        data = np.array(data)
        rms = np.sqrt(np.mean(data**2, axis = 0))

        return rms

    def Crest(self, data):

        data = np.array(data)
        rms = self.RMS(data)
        crest = np.max(np.abs(data), axis = 0)

        return crest/rms

    def Shape(self, data):
        data = np.array(data)
        rms = self.RMS(data)
        mean = np.mean(data, axis = 0)

        return rms / mean

    def p2p(self, data):
        data = np.array(data)
        p2p = np.abs(np.max(data, axis = 0) - np.min(data, axis = 0))

        return p2p

    def Impulse(self, data):
        data = np.array(data)
        p2p = self.p2p(data)
        mean = np.mean(data, axis = 0)

        return p2p / mean

    def Margin(self, data):
        data = np.array(np.abs(data))
        p2p = self.p2p(data)
        cif = np.square(np.mean(np.sqrt(data), axis = 0))

        return p2p / cif

    def Entropy(self, data):
        data = np.array(data)
        entropy = np.array([])
        for i in range(data.shape[1]):
            entropy = np.append(entropy, stats.entropy(data[:,i]))

        return entropy

    def waveform_Entropy(self, data):

        crest = self.Crest(data)
        return (crest * np.log(crest))

    def get_median_absolute_deviation(self, data):
        return np.median(np.absolute(data - np.median(data, axis=0)), axis=0)

    def skew(self, data):
        return stats.skew(data, axis = 0)

    def kurt(self, data):
        return stats.kurtosis(data, axis = 0)


    def des_statiscal(self, data):

        #return count, mean, std, min, 25%, 50%, 75%, max
        des = data.describe()

        #median = data.median(axis = 0)
        #median = pd.DataFrame(np.array(median).reshape(1, -1), index=['median'])


        mad = data.mad(axis = 0)
        mad = pd.DataFrame(np.array(mad).reshape(1, -1), index=['mad'])

        #var = data.var(axis = 0)
        #var = pd.DataFrame(np.array(var).reshape(1, -1), index=['var'])

        skew = data.skew(axis = 0)
        skew= pd.DataFrame(np.array(skew).reshape(1, -1), index=['skew'])

        kurt = data.kurt(axis = 0)
        kurt = pd.DataFrame(np.array(kurt).reshape(1, -1), index=['kurt'])

        #diff = pd.DataFrame(np.array(diff).reshape(1, -1), index=['diff'])

        pct_change = data.pct_change(axis = 0)
        #pct_change = pd.DataFrame(np.array(pct_change).reshape(1, -1), index=['pct_change'])

        #return mean, std, 25%, 50%, 75%, mad, skew, kurt
        return pd.concat([des, mad, skew, kurt]).drop(['mean', 'count', 'max', 'min'])



    def Concat(self, data):
        data = pd.DataFrame(data)
        des = self.des_statiscal(data)

        rms = self.RMS(data)
        rms = pd.DataFrame(rms.reshape(1,-1), index = ['rms'])

        shape = self.Crest(data)
        shape = pd.DataFrame(shape.reshape(1, -1), index=['shape'])

        p2p = self.p2p(data)
        p2p = pd.DataFrame(p2p.reshape(1, -1), index=['p2p'])

        #impulse = self.Impulse(data)
        #impulse = pd.DataFrame(impulse.reshape(1, -1), index=['impulse'])

        mar = self.Margin(data)
        mar = pd.DataFrame(mar.reshape(1,-1), index = ['mar'])

        #entropy = self.Entropy(data)
        #entropy = pd.DataFrame(entropy.reshape(1,-1), index = ['entropy'])

        wave_entropy = self.waveform_Entropy(data)
        wave_entropy = pd.DataFrame(wave_entropy.reshape(1,-1), index = ['wave_entropy'])

        # return std, 25%, 50%, 75%, mad, skew, kurt, rms, shape, p2p, mar, wave_entropy
        return pd.concat([des, rms, shape, p2p, mar, wave_entropy])

class Signal_process():

    '''
    input shape data = [raw, acc]
    output shape =
    '''
    #wavelet parameter를 추가하고 기존 db6 mother function을 사용하는 것에서 바꿈
    def WaveletPacketDecomposition(self, signal, wavelet, level_parm):
        WPD = []
        for i in range(signal.shape[1]):
            wp = pywt.WaveletPacket(data = signal[:,i], wavelet = wavelet, mode = 'symmetric')
            iwp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric')
            level = [node.path for node in wp.get_level(level = level_parm, order = 'freq')]

            for j in range(len(level)):
                iwp[level[j]] = wp[level[j]].data
                WPD.append(iwp[level[j]].reconstruct(update=False))

        return np.transpose(np.array(WPD))

    def FFT_process(self, data, fs):

        n = len(data)
        NFFT = n
        k = np.arange(NFFT)
        f0 = k * fs/NFFT
        f0 = f0[range(math.trunc(NFFT/2))]

        Y = np.fft.fft(data, axis = 0) / NFFT
        Y = Y[range(math.trunc(NFFT/2))]
        ampli = 2 * abs(Y)

        #ampli_norm = ((ampli - np.min(ampli, axis = 0)) / (np.max(ampli, axis = 0) - np.min(ampli, axis = 0) + 1e-6))

        return f0, ampli

    def DownSampling(self, data, length):
        return signal.resample(data, length, axis = 0)

class HealthIndicator_Metric():

    def Corr(self, data):

        '''pearson coefficient of each features'''
        time = np.arange(len(data)).reshape(-1,1)
        corr_numerator = np.abs(np.sum((data - data.mean(axis = 0)) * (time - time.mean(axis = 0)), axis = 0))
        corr_denominator = np.sqrt(np.sum((data - data.mean(axis = 0))**2, axis = 0) * np.sum((time - time.mean(axis = 0))**2, axis = 0))

        return corr_numerator / corr_denominator

    def Monotonicity(self, data):

        '''Monotonicity of each features'''
        time = np.array(len(data) - 1)
        data_front = data[:-1].copy()
        data_back = data[1:].copy()

        df = data_back - data_front

        positive_mask = df > 0
        negative_mask = df < 0

        return np.abs(np.sum(positive_mask, axis = 0) / time - np.sum(negative_mask, axis = 0) / time)

    def Criteria(self, data):

        return (self.Monotonicity(data) +  self.Corr(data)) / 2
















