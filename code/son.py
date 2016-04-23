# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 19:29:23 2016

@author: khalil
"""

import scipy.io.wavfile as wav
import numpy as np
import scipy as sp
filename =[ 'Sample_316.wav','Sample_322.wav']
filename1 = 'Sample_322.wav'
import matplotlib.pyplot as plt
"""# wav.read returns the sample_rate and a numpy array containing each audio sample from the .wav file
sample_rate, recording = wav.read(filename)
#print sample_rate,recording
def calculate_normalized_power_spectrum(recording, sample_rate):
    # np.fft.fft returns the discrete fourier transform of the recording
    fft = np.fft.fft(recording) 
    number_of_samples = len(recording)        
    # sample_length is the length of each sample in seconds
    sample_length = 1./sample_rate 
    # fftfreq is a convenience function which returns the list of frequencies measured by the fft
    frequencies = np.fft.fftfreq(number_of_samples, sample_length)  
    positive_frequency_indices = np.where(frequencies>0) 
    # positive frequences returned by the fft
    frequencies = frequencies[positive_frequency_indices]
    # magnitudes of each positive frequency in the recording
    magnitudes = abs(fft[positive_frequency_indices]) 
    # some segments are louder than others, so normalize each segment
    magnitudes = magnitudes / np.linalg.norm(magnitudes)
    return frequencies, magnitudes
    
def split_recording(recording, segment_length, sample_rate):
    segments = []
    index = 0
    while index < len(recording):
        segment = recording[index:index + segment_length*sample_rate]
        segments.append(segment)
        index += segment_length*sample_rate
    return segments

segment_length = .5 # length in seconds
segments = split_recording(recording, segment_length, sample_rate)
   
def create_power_spectra_array(segment_length, sample_rate):
    number_of_samples_per_segment = int(segment_length * sample_rate)
    time_per_sample = 1./sample_rate
    frequencies = np.fft.fftfreq(number_of_samples_per_segment, time_per_sample)
    positive_frequencies = frequencies[frequencies>0]
    power_spectra_array = np.empty((0, len(positive_frequencies)))
    return power_spectra_array

def fill_power_spectra_array(splits, power_spectra_array, fs):
    filled_array = power_spectra_array
    for segment in splits:
        freqs, mags = calculate_normalized_power_spectrum(segment, fs)
        filled_array = np.vstack((filled_array, mags))
    return filled_array
freq,mag= calculate_normalized_power_spectrum(recording, sample_rate)
power_spectra_array = create_power_spectra_array(segment_length,sample_rate)
power_spectra_array = fill_power_spectra_array(segments, power_spectra_array, sample_rate)
print power_spectra_array"""
from scikits.talkbox.features import mfcc
for row in filename :
    sample_rate, X = sp.io.wavfile.read(row)
    ceps, mspec, spec = mfcc(X)
    print(ceps.shape)
    num_ceps=len(ceps)
    print len(ceps[0])
    x = np.mean(ceps[int(num_ceps*0.1):int(num_ceps*0.9)], axis=0)
    print x
#    fig, ax = plt.subplots(1, 1, sharey=True)
#    cax = ax.imshow(np.transpose(ceps), origin="lower", aspect="auto",  interpolation="nearest")
#    plt.show() 
