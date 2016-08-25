#!/usr/bin/env/python

'''
FRBstack.py

Stacks the repeating FRB absorption spectra. Be sure to edit params in the order [filename,full-width-half-max (ms)] as new burst data are added. Can optionally plot the stacked on- and off-pulse spectra as well.

Usage on the command line:
>> python FRBstack.py -infile INFILE(.npz files) [OPTIONS]

Stella Koch Ocker (socker@oberlin.edu) - Aug 22, 2016

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rc
import argparse
from ConfigParser import ConfigParser
import sys

#defining command line arguments
parser = argparse.ArgumentParser('FRBstack.py')
parser.add_argument('-infile',help='List of .npz file(s) to analyze',nargs='*',action='store',dest='infile',required=True)
parser.add_argument('-absorption',help='If flag is set, plots stacked absorption spectra.',action='store_true',dest='absorption')
parser.add_argument('-onpulse',help='If flag is set, plots stacked on-pulse spectra. Default is False.',action='store_true',dest='onpulse')
parser.add_argument('-offpulse',help='If flag is set, plots stacked off-pulse spectra on top of on-pulse spectra. Default is false.',action='store_true',dest='offpulse')
parser.set_defaults(absorption=False,onpulse=False,offpulse=False,save=False)
args = parser.parse_args()
listdata = args.infile
absorption = args.absorption
onpulse = args.onpulse
offpulse = args.offpulse

#specifying full-width-at-half-maximum of each burst
params = ['p2030.20121102.G175.04-00.26.C.b4.00000_wfall_bpass_full_128.npz', 3.3, 		'p2030.20150517.FRBGRID2.b6.00000_wfall_bpass_full_164.npz', 3.8, 
	'p2030.20150517.FRBGRID2.b6.00000_wfall_bpass_full_736.npz', 3.3, 
	'p2030.20150602.FRBGRID2.b6.00000_wfall_bpass_full_381.npz', 4.6,		
	'p2030.20150602.FRBGRID2.b6.00000_wfall_bpass_full_950.npz', 8.7,
	'p2030.20150602.FRBGRID6.b0.00000_wfall_bpass_full_447.npz', 2.8,
	'p2030.20150602.FRBGRID6.b0.00000_wfall_bpass_full_469.npz', 6.1,
	'p2030.20150602.FRBGRID6.b0.00000_wfall_bpass_full_527.npz', 6.6,
	'p2030.20150602.FRBGRID6.b0.00000_wfall_bpass_full_714.npz', 6.0,
	'p2030.20150602.FRBGRID6.b0.00000_wfall_bpass_full_883.npz', 8.0,
	'p2030.20150602.FRBGRID6.b0.00000_wfall_bpass_full_940.npz', 3.06]


def main(listdata,params):
	#initializing figure
	fig = plt.figure()
	ax = fig.add_subplot(111)

	#initializing stacking arrays
	onstack = np.zeros(960)
	diffstack = np.zeros(960)
	offstack = np.zeros(960)

	freqs = np.linspace(1214.28955078125, 1536.68811035156, 960)

	for i in listdata:
		
		#loading data
		npz_fn = str(i)
		spectrum = np.load(npz_fn)
		dat = spectrum['data']

		#pulse profile array
		pulse_profile = np.sum(dat,axis=0)

		#channel width (ms), pulse peak index, full-width-half-max
		tchan = 0.065454751146
		indpeak = np.argmax(pulse_profile)
		index = params.index(npz_fn)
		fwhm = params[index+1]
		
		#calculating minimum and maximum indices of burst (everything w/in 3 sigma of pulse peak)
		sigma = fwhm/(2*np.sqrt(2*np.log(2)))
		indmin = indpeak - ((3*sigma)/tchan)
		indmax = indpeak + ((3*sigma)/tchan)

		#subtracting the off-pulse mean from the total data array and dividing by the off-pulse standard dev.
		sm1 = np.mean(dat[:,0:indmin])
		sm2 = np.mean(dat[:,indmax:3055])
		sm = (sm1+sm2)/2
		strd1 = np.std(dat[:,0:indmin])
		strd2 = np.std(dat[:,indmax:3055])
		strd = (strd1+strd2)/2
		dat = (dat-sm)/strd

		#off- and on-pulse spectra
		offpulspec = np.sum(dat[:,0:indmin],axis=1)/(indmin)
		onpulspec = np.sum(dat[:,indmin:indmax],axis=1)/(indmax-indmin)

		#removing polynomial trend from on-pulse spectrum
		poly2 = np.poly1d(np.polyfit(freqs,onpulspec,3))
		onpulspec = onpulspec - poly2(freqs)

		#correcting indices to avoid bad small values
		onpulspec[(onpulspec+poly2(freqs))<= sm] = 0
		offpulspec[(onpulspec+poly2(freqs))<= sm] = 0
		offpulspec[0:15]=0

		#absorption spectrum
		diff = onpulspec - offpulspec

		#stacking arrays
		diffstack = np.vstack((diffstack,diff))
		onstack = np.vstack((onstack,onpulspec))
		offstack = np.vstack((offstack,offpulspec))

		npz_fn = None
		spectrum =  None
		dat = None
		sm1 = None
		sm2 = None
		sm = None
		strd1 = None
		strd2 = None
		strd = None
		tpeak = None
		indpeak = None
		fwhm = None
		sigma = None
		indmin = None
		indmax = None
		onpulspec = None
		offpulspec = None
		diff = None

	#stacked spectra
	onstack1 = (np.sum(onstack,axis=0))/float(len(listdata)) 
	offstack1 = (np.sum(offstack,axis=0))/float(len(listdata))
	diffstack1 = (np.sum(diffstack,axis=0))/float(len(listdata))

	plt.rc('text',usetex=True)
	plt.rc('font',family='serif')
	
	if absorption==True:
		ax.plot(freqs,diffstack1,color='black')
	elif onpulse==True:
		ax.plot(freqs,onstack1,color='black')
	elif offpulse==True:
		ax.plot(freqs,onstack1,'black')
		ax.plot(freqs,offstack1,'grey')

	ax.set_xlabel(r'Frequency (MHz)')
	ax.set_ylabel(r'Intensity')

	plt.show()

main(listdata,params)














		
