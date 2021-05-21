
import matplotlib.pyplot as plt
import numpy as np


def make_note(i, f, dy=5):
    X = np.linspace(i, i+1, 100)
    m1 = 100 + np.random.rand() * 100
    m2 = np.random.rand() * 2 * np.pi
    m3 = np.random.normal(dy, 2)
    Y = f + (np.sin(X * m1 /(2*np.pi) + m2) + np.random.rand(X.size))*m3
    return X, Y


def plot_single(p, X, F, I, txt=[]):
    fig, ax = plt.subplots(1,2, figsize=(10,4.25))

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_xlim(-0.5, 8.5)
    ax[0].set_ylim(180, 420)
    ax[1].set_ylim(-50, 1250)

    for x, f in zip(X, F):
        ax[0].plot(*make_note(x, f, 2), '-k')
    ax[0].set_ylabel("Frequency / Hz")
    ax[0].set_xticks([])

    if len(I):
        for x, i in zip(X, I):
            ax[1].plot(*make_note(x, i), '-k')
        ax[1].set_ylabel("Frequency ratio (interval) / cents")
    else:
#       fig.delaxes(ax[1])
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].set_yticks([])
    ax[1].set_xticks([])

    yextra = [10, 30]

    if len(txt):
        for i, a in enumerate(ax):
            for x, y, t in zip(X, [F, I][i], txt):
                a.annotate(t, (x+0.2, y + yextra[i]), fontsize=14)
    
    fig.savefig(f"../Figures/scales_intro_{p}.pdf", bbox_inches='tight')
    fig.savefig(f"../Figures/scales_intro_{p}.png", bbox_inches='tight')


def plot_series():
    imaj = np.array([0, 200, 200, 100, 200, 200, 200, 100])
    smaj = np.cumsum(imaj)
    fmaj = 200 * 2**(smaj/1200)

    X = np.arange(fmaj.size)
    plot_single(0, [X[0]], [fmaj[0]], [])
    plot_single(1, X, fmaj, [])
    plot_single(2, X, fmaj, smaj)
    
    plot_single(3, list(X) + [X[-1]], list(fmaj) + [fmaj[0]], list(smaj) + [smaj[0]])
    
    txt = "CDEFGABCC"
    plot_single(4, list(X) + [X[-1]], list(fmaj) + [fmaj[0]], list(smaj) + [smaj[0]], txt=txt)




