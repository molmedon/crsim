from matplotlib import pyplot as plt

class A3CosmicRay:

  def __init__(self,name_,waveform_):
    self.name = name_ 
    self.waveform = waveform_ 
    self.minbias = []
    self.snr = 0 
 
  #def setMinbias(self,minbias_):
  #  self.minbias = minbias_

  def combine(self):
    return self.waveform + self.minbias

  def rescale(self,denom):
    self.waveform[:] = [x / denom for x in self.waveform] 
    
  def setSNR(self,snr_):
    self.snr = snr_
