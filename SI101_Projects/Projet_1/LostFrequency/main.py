import IPython.display as ipd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

set_1 = {i+1: chr(ord('a') + i) for i in range(26)}
set_1[0] = ' '

a, Fe = sf.read('../sounds/symboleA.wav')
b, Fe = sf.read('../sounds/symboleA2.wav')
c, Fe = sf.read('../sounds/symboleU.wav')
d, Fe = sf.read('../sounds/symboleU2.wav')
e, Fe = sf.read('../sounds/mess_ssespace.wav')
f, Fe = sf.read('../sounds/mess.wav')
g, Fe = sf.read('../sounds/mess_difficile.wav')

Nfft = 72000


def decode_padding(x,Fe, hanning):
    seuil = 25
    global Nfft
    if hanning:
        seuil = 22
        x = x*np.hanning(len(x))
    u = np.fft.fft(x,Nfft)
    #show_TF(u)
    index = np.argmax(abs(u[(501 * Nfft)//Fe :(527 * Nfft)//Fe])) + (501 * Nfft) // Fe
    frequence_estimee = round((index * Fe) / Nfft)
    if abs(u[index]) > seuil:
        return frequence_estimee
    return 500
    

def show_TF(u):
    global Nfft
    locs = np.linspace(1,27,num=26*Nfft//Fe,endpoint=True)
    plt.plot(locs, abs(u[(501* Nfft)//Fe:(527* Nfft)//Fe]))
    plt.show()


def decode_letter(x,Fe, hanning = False):
    key = int(decode_padding(x,Fe, hanning) - 500)
    keys = set_1.keys()
    if key not in keys:
        return ' '
    else:
        return set_1[key]


def decode(x,Fe):
    set = []
    n = len(x)
    if n == 2000:
        return decode_letter(x,Fe,hanning = True)
    nb_car = int(n//2500)
    for k in range(nb_car):
        set.append(x[k*2500:k*2500+2000])
    m = len(set)
    str = ''
    for i in range(m):
        str = str + decode_letter(set[i],Fe)
    return str


print(decode(a,Fe))
print(decode(b,Fe))
print(decode(c,Fe))
print(decode(d,Fe))
print(decode(e,Fe))
print(decode(f,Fe))
print(decode(g,Fe))
