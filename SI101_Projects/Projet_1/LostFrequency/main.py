import IPython.display as ipd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

set_1 = {i+1: chr(ord('a') + i) for i in range(26)}
set_1[0] = ' '
x, Fe = sf.read('/Users/rayanedakhlaoui/Desktop/frequence_perdue-main/sounds/mess_difficile.wav')
y, Fe = sf.read('/Users/rayanedakhlaoui/Desktop/frequence_perdue-main/sounds/mess.wav')
z,Fe = sf.read('/Users/rayanedakhlaoui/Desktop/frequence_perdue-main/sounds/mess_ssespace.wav')
w, Fe = sf.read('/Users/rayanedakhlaoui/Desktop/frequence_perdue-main/sounds/symboleU2.wav')


def decode_padding(x,Fe):
    seuil = 25
    Nfft = 24000
    x = x*np.hanning(len(x))
    u = np.fft.fft(x,Nfft)
    show_TF(u)
    index = np.argmax(abs(u[(501 * Nfft)//Fe :(527 * Nfft)//Fe])) + (501 * Nfft) // Fe
    frequence_estimee = round((index * Fe) / Nfft)
    if abs(u[index]) > seuil:
        return frequence_estimee
    return 500
    

def show_TF(u):
    Nfft = 24000
    locs = np.linspace(0,26,num=26*Nfft//Fe,endpoint=True)
    plt.stem(locs, abs(u[(501* Nfft)//Fe:(527* Nfft)//Fe]))
    plt.show()

def show(x,Fe):
    t = np.linspace(0, x.shape[0]/Fe, x.shape[0])
    plt.plot(t, x, label="Signal échantillonné")
    plt.grid()
    plt.xlabel(r"$t$ (s)")
    plt.ylabel(r"Amplitude")
    plt.title(r"Signal sonore")
    plt.show()




def decode_letter(x,Fe):
    key = int(decode_padding(x,Fe) - 500)
    print(key)
    keys = set_1.keys()
    if key not in keys:
        print('space')
        return ' '
    else:
        print(set_1[key]) 
        return set_1[key]



def decode(x,Fe):
    set = []
    n = len(x)
    nb_car = int(n//2500)
    for k in range(nb_car):
        set.append(x[k*2500:k*2500+2000])
    m = len(set)
    str = ''
    for i in range(m):
        str = str + decode_letter(set[i],Fe)
    return str

print(decode(x,Fe))
print(decode(y,Fe))
print(decode(z,Fe))
print(decode_letter(w,Fe))
# filter(x,Fe)
