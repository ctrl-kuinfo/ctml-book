import matplotlib.pyplot as plt

def global_config(fontsize=25, type=0):
    plt.rc('text', usetex=True) # use latex
    # Set global font size
    plt.rcParams['font.size'] = fontsize  # Set the font size for all text
    plt.rcParams['axes.labelsize'] = fontsize  # Font size for axis labels
    plt.rcParams['xtick.labelsize'] = fontsize-7  # Font size for x-axis tick labels
    plt.rcParams['ytick.labelsize'] = fontsize-7  # Font size for y-axis tick labels
    plt.rcParams['legend.fontsize'] = fontsize  # Font size for the legend
    plt.rcParams['figure.titlesize'] = fontsize  # Font size for the figure title
    if type == 0:
        return (8,6)
    elif type==1: 
        return (8,7)
    elif type==2:
        return (12,6)
    else:
        return (4,3)