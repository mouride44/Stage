# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 19:09:32 2016

@author: khalil
"""
import math
import numpy as np
from matplotlib.pyplot import *
import scipy.io.wavfile as wave
from numpy.fft import fft
import matplotlib.pyplot as plt

rate,data = wave.read('Sample_313.wav')
n = data.size
frate = 11025.0
print rate
dure=1.0*n/rate
print data.size
spectre = np.fft.fft(data)
freq = np.fft.fftfreq(data.size)
#mask=freq>0   
#plt.plot(freq[mask],np.abs(spectre[mask]))
#print freq[0:3[0]

#print np.abs(spectre),np.angle(spectre)
positive_frequencies = freq[np.where(freq > 0)]
magnitudes = abs(spectre[np.where(freq>0)])
#print len(magnitudes)
print len(positive_frequencies )
peak_frequency = np.argmax(magnitudes)
cpt=0
#print peak_frequency 
print  positive_frequencies
  
        

#print np.absolute(fft(data[2.5:0.0]))
#pfreqrint len(np.abs(spectre[mask]))
"""freqs = np.fft.fftfreq(len(w))
print(freqs.min()*frate, freqs.max()*frate)
# (-0.5, 0.499975)
# Find the peak in the coefficients
idx = np.argmax(np.abs(w))
freq = freqs[idx]
freq_in_hertz = abs(freq * frate)
print idx
print n
print(freq_in_hertz)
frqhz=[]
print  np.abs(w)"""
	