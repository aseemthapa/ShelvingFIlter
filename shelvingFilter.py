"""
    Name : Aseem Thapa
    ID : 1001543178
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def ftplotter (xft,yft):
    Xmax =  (np.max(xft))
    Ymax =  (np.max(yft))
    max = calc_max(Xmax,Ymax)
    #print(max)
    output = plt.figure(figsize=(8, 5))
    """INPUT"""
    output.add_subplot(1,2,1)
    Nx = np.arange(0,len(xft))
    plt.plot(Nx,xft)
    plt.title('original signal')
    axesin = plt.gca()
    axesin.set_xlim([0,len(xft)/4])
    axesin.set_ylim([0,max+100])
    #plt.show()
    
    """OUTPUT"""
    output.add_subplot(1,2,2)
    Ny = np.arange(0,len(yft))
    plt.plot(Ny,yft)
    plt.title('filtered signal')
    axesout = plt.gca()
    axesout.set_xlim([0,len(yft)/4])
    axesout.set_ylim([0,max+100])
    #plt.show()
    pass

def calc_max(a,b):
    if(a>=b):
        return a
    else:
        return b
    
def applyShelvingFilter(inName, outName, g, fc) :
    #Read the data:
    x, fs = sf.read(inName)
    mu = 10**(g/20.0) ##mu = 10^(g*20)
    ThetaC = 2*np.pi*(fc/fs) #Normalized cutoff frequency
    beta = (4/(1+mu)) * np.tan(ThetaC/2)
    gamma = (1-beta)/(1+beta)
    #gamma = (1-((4/(1+mu)) * np.tan(ThetaC/2)))/(1+((4/(1+mu)) * np.tan(ThetaC/2)))
    alpha = (1-gamma)/2
    
    """ Debug:
    print(float(mu))
    print(float(ThetaC))
    print(float(beta))
    print(float(gamma))
    print(float(alpha))
    """
    
    u = np.zeros(len(x))
    y = np.zeros(len(x))
    #y[n] = x[n] + (mu-1)*(alpha*(x[n]+x[n-1]) + gamma*y[n-1])

    for n in range(0,len(x)):
        if (n==0):
            u[n] = alpha*x[n]
            #print(y[n])
        else:
            u[n] = (alpha*(x[n]+x[n-1]) + (gamma * u[n-1]))
            #print(y[n])
        
    for n in range(0,len(x)):
        y[n] = x[n] + (mu-1) * u[n]
        
    """PLOT THE FTs"""
    xft = abs(np.fft.fft(x))

    yft = abs(np.fft.fft(y))
  
    ftplotter(xft,yft)
    
    """WRITE OUT"""
    sf.write(outName,y,fs)
    pass 



##########################  main  ##########################
if __name__ == "__main__" :
    inName = "P_9_1.wav" #Input File Name
    
    gain = 20 # can be positive or negative
                # WARNING: small positive values can greatly amplify the sounds
    cutoff = 30000
    
    #output filename->
    outName = "shelvingOutput.wav"

    applyShelvingFilter(inName, outName, gain, cutoff)
