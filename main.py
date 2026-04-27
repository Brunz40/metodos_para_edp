import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def derivada(x,h,func):
    return (func(x+h)-func(x-h))/(2*h)

def derivada_segunda(x,h,func):
    return (func(x+h)-2*func(x)+func(x-h))/(h**2)

def solução_analitica(x,y):
    return np.exp(-y)*np.sin(2*np.pi*x)

constante=4*np.pi**2-3
h1=0.1
h2=0.05
h3=0.025
h4=0.0125
h5=0.00625
H=[h1,h2,h3,h4,h5]

malha_aprox1=np.zeros((10,20))
malha_aprox2=np.zeros((20,40))
malha_aprox3=np.zeros((40,80))
malha_aprox4=np.zeros((80,160))
malha_aprox5=np.zeros((160,320))

malhas_exatas = {}
erros={}
for h in H:
    #criando a malha
    x=np.linspace(0,1,int(1/h))
    y = np.linspace(-1, 1, int(2 / h))
    X,Y=np.meshgrid(x,y)


    malhas_exatas[h]=solução_analitica(X,Y)



    fig, (ax,ax2) = plt.subplots(ncols=2,subplot_kw={"projection": "3d"})

    ax.plot_surface(X, Y, malhas_exatas[h], cmap=cm.jet, linewidth=0, antialiased=False)

    plt.show()