import matplotlib.pyplot as plt


def plot_wrapper(plot_object, title, subtitle, xlabel, ylabel, xtick_rotation=None):
    """
    Simple function to set title, labels etc. of a seaborn plot. Helps with readability.
    The function prints the plot.

    Arguments
    ---------
    :param plot_object:    a seaborn plot object
    :param title:          str, sets a bold title
    :param subtitle:       str, sets a subtitle
    :param xlabel:         str, replaces xlabel
    :param ylabel:         str, replaces ylabel
    :param xtich_rotation: int, rotates xticks

    Returns
    -------
    :return None:          the function prints a plot
    """
    g = plot_object

    # Title, Subtitle and Axis
    g.text(x=0.5,
           y=1.06,
           s=title,
           fontsize=10, weight='bold', ha='center', va='bottom', transform=g.transAxes)
    g.text(x=0.5,
           y=1.01,
           s=subtitle,
           fontsize=10, alpha=0.75, ha='center', va='bottom', transform=g.transAxes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xtick_rotation is not None:
        plt.xticks(rotation=xtick_rotation)
    plt.show()


