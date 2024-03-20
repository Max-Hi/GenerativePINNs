

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
#mpl.use('pgf')


# further TODO: network structure : ResNet, RNN, etc


def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size



def plot_with_ground_truth(output_vec, X_star, X , T,  ground_truth, ground_truth_ref=False, ground_truth_refpts=[], filename = "ground_truth_comparison.png"):
    """
    input: output_vec and ground_truth as 1 dimensional np.array.
    X_star(grid) correspond to the input positions of both output and ground_truth.
    x, t constitute X_input
    if ground_truth_ref==true,show difference of the two plots alongside.
    """
    # pgf_with_latex = {                      # setup matplotlib to use latex for output
    # "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    # "text.usetex": True,                # use LaTeX to write all text
    # "font.family": "serif",
    # "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    # "font.sans-serif": [],
    # "font.monospace": [],
    # "axes.labelsize": 10,               # LaTeX default is 10pt font.
    # "font.size": 10,
    # "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    # "xtick.labelsize": 8,
    # "ytick.labelsize": 8,
    # "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth
    # "pgf.preamble": [
    #     r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
    #     r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
    #     ]
    # }
    # mpl.rcParams.update(pgf_with_latex)
    fig = plt.figure(figsize=(10, 8))
    gs0 = gridspec.GridSpec(2, 1)
    gs0.update(top=1-0.06, bottom=0.06, left=0.15, right=0.85, wspace=5)
    
    # error_u = np.linalg.norm(output_vec-ground_truth)
    U_pred = griddata(X_star, output_vec.flatten(), (X, T), method = "cubic")
    U_actual = griddata(X_star, ground_truth.flatten(), (X, T), method = "cubic")
    
    ax = plt.subplot(gs0[0, :])
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[T.min(), T.max(), X.min(), X.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_title('Predicted')

    ax2 = plt.subplot(gs0[1, :])
    h2 = ax2.imshow(U_actual.T,interpolation='nearest', cmap='rainbow', 
                  extent=[T.min(), T.max(), X.min(), X.max()], 
                  origin='lower', aspect='auto')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h2, cax=cax2)
    ax2.set_title('Ground Truth')
    plt.savefig(filename)
    plt.show()

def plot_loss(loss_history, filename):
    import seaborn as sns
    """loss_history: dictionary"""
    epoch = list(range(1, len(loss_history["epoch"])+1))
    # if epoch in loss_history.keys():
    #     epoch = loss_history["epoch"]
    # else:
    #     epoch = list(range(1, len(loss_history["Generator"])))

    plt.figure(figsize=(8, 6))
    for label in loss_history.keys():
        if label != "epoch" and loss_history[label] != []:
            y = np.array(loss_history[label]).flatten()
            sns.lineplot(x=epoch, y=y, label = label)

    plt.title('Loss Descent Over Training')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    




