import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import diags

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

    Npontos=(len(x))*(len(y))
    #PONTOS INTERNOS
    diagonal_principal=-8*np.ones(Npontos)
    diag_sup=diag_inf=np.ones(Npontos-1)
    diag_longe_sup=(3 - constante*h/2) * np.ones(Npontos-len(x) +1 )
    diag_longe_inf=(3 + constante*h/2) * np.ones(Npontos-len(x) +1 )
    #pontos fronteiras y
    diagonal_principal[-len(x) +1:]-=2*h*(3 - constante*h/2)
    diag_sup[:len(x) -1]=diag_inf[-len(x) +1:]
    #ajuste de buraco pulando linha
    for i in range(1, int(Npontos/len(x))):
        diag_sup[i*len(x) - 1] = 0
        diag_inf[i*len(x) - 1] = 0
    #fronteiras em x
    nx=len(x)
    for i in range(Npontos):
        if i % len(x) == 0 or i % len(x) == len(x) - 1:
            diagonal_principal[i] = 1
            if i < Npontos - 1: diag_sup[i] = 0
            if i > 0: diag_inf[i-1] = 0
            if i < Npontos - nx: diag_longe_sup[i] = 0
            if i >= nx: diag_longe_inf[i-nx] = 0
    #montando matriz esparsa
    A = diags(
    [diagonal_principal, diag_sup, diag_inf, diag_longe_sup, diag_longe_inf],
    [0, 1, -1, len(x), -len(x)],shape=(Npontos, Npontos)).tocsr()

    fig, (ax,ax2) = plt.subplots(ncols=2,subplot_kw={"projection": "3d"})

    ax.plot_surface(X, Y, malhas_exatas[h], cmap=cm.jet, linewidth=0, antialiased=False)

    plt.show()