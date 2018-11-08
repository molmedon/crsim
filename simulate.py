import sys
import glob
from cr_param import *
import numpy as np
from scipy import signal as sig
from scipy import stats as stat
from matplotlib import pyplot as plt
from copy import *
#from random import seed
#from random import random


###update sys.path
sys.path.append('/Users/aztlan/work/ANITA/crsim/python')

############################################
####load data and create cosmic ray list####
############################################
def load_data_names(directory: str) -> np.ndarray:
  
  pattern = (f'{directory}/**.txt')
 
  filename = glob.glob(pattern) 

  if filename:
  
    return filename 
  
  else:
  
    print(f'no file matched \'{pattern}\'')
    exit(1)


def crList(cr_list):

  cr_wavefrom_tmp = []

  filenames = load_data_names('../a3cosmicRays')

  count=0

  for filename in filenames:
    
    cr_waveform_tmp = np.genfromtxt(filename) 

    cr_waveform_tmp_pad = np.hstack((cr_waveform_tmp[:,1], np.zeros(1560 - len(cr_waveform_tmp)))) 

    cr_tmp = A3CosmicRay('cr{}'.format(count),cr_waveform_tmp_pad)

    #plt.plot(cr_tmp.waveform)
    #plt.show();
    #print(len(cr_tmp.waveform))
    cr_list.append(cr_tmp)
    count = 1 + count

############################################
###########useful functions#################

###########peak to peak#####################
def pk2pk(wave):
  
  max_val = np.amax(wave)
  #print('max_val = {}'.format(max_val))
  min_val = np.amin(wave)
  #print('min_val = {}'.format(min_val))

  max_index = np.argmax(wave)
  min_index = np.argmin(wave)

  sec_peak_val=0
  main_peak_val=0

  if abs(min_val) > max_val:
    #print(min_index)
    main_peak_val = abs(min_val)    
    
    for i in range(min_index,len(wave)):
      #print(i,len(wave))
      if i < len(wave) and i==len(wave)-1: 
        sec_peak_val = wave[i]

      elif wave[i] > wave[i+1]: 
        sec_peak_val = wave[i]
        break

  if max_val > abs(min_val): 
    #print(max_index)
 
    main_peak_val = max_val
 
    for i in range(max_index,len(wave)):

      if i < len(wave) and i==len(wave)-1: 
        sec_peak_val = wave[i]

      elif wave[i+1] > wave[i]: 
        sec_peak_val = wave[i]
        break
      
  #print('sec_peak_val = {}'.format(sec_peak_val))
  return main_peak_val + abs(sec_peak_val) 

###########root mean square#################
def mbrms(mbwave):
  
  np_wave = np.array(mbwave)
  
  rms = np.sqrt(np.mean(np_wave**2))
  return rms
############################################
#######empty list for original a3cr#########
a3_list = []

############################################
#############filling it up##################
crList(a3_list)
#print(a3_list[0].name)

############################################
##########empty list for sim crs############
simcr_list = []
hilbert_list = []
mrms_list = []


minbias_names = load_data_names('../minbias_waveform')
########################################
##########start simulation##############
########################################

########loading minbias datasets########
rms_avg = 3.4868307654938864
count=0
overall_count=0
###########Looping through original cr list############
for crs in a3_list:

  crhpeak = np.amax(abs(sig.hilbert(crs.waveform)))

  #############scale to desired snr#########################
  #scale_factor = pk2pk(crs.waveform)/(0.5*2*rms_avg)
  scale_factor = crhpeak/(50*rms_avg)

  for minbias in minbias_names:

     ###create new cosmic rays from copies of original ones########
     simcr_tmp = deepcopy(crs)    
     
     ###rescale the wave form###################################### 
     simcr_tmp.rescale(scale_factor)
 
     ###calc snr of waveform########################################
     minbias_waveform = np.genfromtxt(minbias) 
     minbias_rms = mbrms(minbias_waveform)

     sigpo = pk2pk(simcr_tmp.waveform)/2.0

     wsnr = sigpo/rms_avg 
     #print('snr = {}'.format(wsnr))     
     #plt.plot(simcr_tmp.waveform) 
     #plt.show()

     if wsnr < 10000:  
       mrms_list.append(minbias_rms)
     
       ###add minbias wavefom to cosmic ray object##################
       #simcr_tmp.setMinbias(minbias[n:m,1])
       #simcr_tmp_pad = np.block(simcr_tmp.waveform]) 
       simcr_tmp_fft = np.fft.fft(simcr_tmp.waveform) 
   
       #minbias_waveform_pad = np.block([a,minbias_waveform,b]) 
 
       #print('plotting fft of waveform real')
       #plt.plot(simcr_tmp_fft)
       #print(len(simcr_tmp_fft.real)) 
       #plt.show()
       #print('plotting fft of waveform img')
       #plt.plot(simcr_tmp_fft.imag)
       #plt.show()
        
       #print('plotting fft real part of minbias') 
       minbias_waveform_fft = np.fft.fft(minbias_waveform) 
       #plt.plot(minbias_waveform_fft.real) 
       #print(len(minbias_waveform_fft.real))
       #plt.show()
       #print('plotting fft img part of minbias') 

       #simcr_tmp_fft_pad = np.hstack((simcr_tmp_fft.real[0:780], np.zeros(1560 - len(simcr_tmp_fft.real[0:780])))) 

       #waveform_fft_amp = np.absolute(minbias_waveform_fft) + np.absolute(simcr_tmp_fft) 
       waveform_fft_amp = np.absolute(simcr_tmp_fft) 
       minbias_fft_amp = np.absolute(minbias_waveform_fft) 
       #print(waveform_fft_real)
       #plt.plot(np.absolute(minbias_waveform_fft))
       #plt.plot(np.absolute(simcr_tmp_fft))
       #plt.plot(np.absolute(waveform_fft_amp))
       #plt.show()

       #waveform_fft_phase = np.angle(minbias_waveform_fft) - np.angle(simcr_tmp_fft) 
       waveform_fft_phase = np.angle(simcr_tmp_fft)  
       minbias_fft_phase = np.angle(minbias_waveform_fft) #+ np.angle(simcr_tmp_fft) 
 
       #print(waveform_fft_img)

       waveform_tmp_fft = waveform_fft_amp*(np.cos(waveform_fft_phase) +1j*np.sin(waveform_fft_phase)) + minbias_fft_amp*(np.cos(minbias_fft_phase) + 1j*np.sin(minbias_fft_phase))  
       #waveform_tmp_fft = minbias_fft_amp*(np.cos(minbias_fft_phase) + 1j*np.sin(minbias_fft_phase))  
    
       
 
       #waveform_tmp_fft = minbias_fft_amp*(np.cos(minbias_fft_phase) +1j*np.sin(minbias_fft_phase))   
 
       #waveform_tmp_fft =  waveform_fft_real + waveform_fft_img 
       #print(waveform_tmp_fft)

       #print('plotting img part of combined waveform fft ')
       #plt.plot(waveform_fft_img)
       #plt.show()
       #print('plotting real part of combined waveform fft')
       #plt.plot(waveform_tmp_fft.real)      
       #plt.plot(waveform_tmp_fft.imag)      
       #plt.show()
       #print(simcr_tmp.snr)
       waveform_tmp_inv = np.fft.ifft(waveform_tmp_fft)
       #print(waveform_tmp_inv)

       simcr_tmp.waveform = waveform_tmp_inv.real

       sigp = pk2pk(simcr_tmp.waveform)/2.0
       hpeak = np.amax(abs(sig.hilbert(simcr_tmp.waveform)))
       simcr_tmp.setSNR(hpeak/rms_avg)
       #print('a/sigma = {}'.format(simcr_tmp.snr))
       #print(len(waveform_tmp_inv))
       #print(simcr_tmp.waveform)

       simcr_list.append(simcr_tmp.snr) 
#      hilbert_list.append(np.amax(abs(sig.hilbert(simcr_tmp.waveform))))
       #hilbert_list.append(sigp/3.897233496966727)
       #if (sigp/3.897233496966727) > 4: #and np.argmax(simcr_tmp.waveform)!=main_index:
       #  print('true')
       #print(np.argmax(simcr_tmp.waveform))
       #  print(sigp/3.897233496966727)
       #  #sigp = rep2p(384,simcr_tmp.waveform)/2.0 
       #  print('done')
       #plt.plot(simcr_tmp.waveform)
       #plt.show() 
       #  #continue
       #  #hilbert_list.append(1)
       #else:
       sigp = pk2pk(simcr_tmp.waveform)/2.0
       hilbert_list.append(hpeak/rms_avg)
 
     ###move up minbias list####################################### 
       n = 1560*(i+1)
       #print('n = {}'.format(n))
       m = n+1560
       #print('m = {}'.format(m))
       plt.plot(simcr_tmp.waveform)
       plt.show()
       #print(simcr_tmp.hpeak)
       np.savetxt('sim_pulses/snr_13/sim_cr_{}.txt'.format(overall_count),np.c_[minbias[0:1560,0],simcr_tmp.waveform])
       overall_count = overall_count + 1
  count = count + 1
  #break
##################################
####end of simulation#############
##################################
b,loc,scale = stat.rice.fit(hilbert_list)
x = np.linspace(stat.rice.ppf(0.0001,b,loc,scale), stat.rice.ppf(0.9999,b,loc,scale), 1000)

locg,scaleg = stat.norm.fit(hilbert_list)
xg = np.linspace(stat.norm.ppf(0.0001,locg,scaleg), stat.norm.ppf(0.9999,locg,scaleg), 1000)

np.savetxt('snr_13.txt',hilbert_list)
plt.hist(simcr_list,100)
plt.axis([0,50,0,60])
plt.xlabel('snr')
plt.ylabel('a.u.')
plt.savefig('snr.png')
plt.show()

rms_mean = np.mean(mrms_list)

print(rms_mean)

plt.hist(hilbert_list,density=rms_mean)
plt.plot(xg,stat.norm.pdf(xg,locg,scaleg))
plt.plot(x,stat.rice.pdf(x,b,loc,scale))
plt.savefig('rice_13.png')
plt.show() 

plt.hist(cr_snr_list)
plt.show()
plt.savefig('a3_cr_snr.png')

plt.hist(np.log(simcr_list),100)
plt.savefig('lnsnr.png')
