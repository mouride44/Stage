# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:13:16 2016

@author: khalil
"""

#text to txt 
infile = open('2070_MarkedPeaks.txt','r')
outfile= open('2070test.txt','w')
outfile.write( "bienvenue" + '\n')
copy = False
for line in infile:

    if line.strip() == "Frequency [Hz]           Peak amplitude           Phase [degrees]":
        bucket = []
        copy = True

    elif line.strip() == "Peak interpolation: Numeric":
        for strings in bucket:
            outfile.write( strings + '\n')
        copy = False

    elif copy:
        bucket.append(line.strip())
