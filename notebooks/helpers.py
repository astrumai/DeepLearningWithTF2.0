import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import tensorflow as tf

def vector_plot(vecs, xlim, ylim, cols=["#1190FF", "#FF9A13"], alpha=1):
    plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'red'})
    plt.axvline(x=0, color='k', zorder=0)
    plt.axhline(y=0, color='k', zorder=0)

    for i in range(len(vecs)):
        if (isinstance(alpha, list)):
            alpha_i = alpha[i]
        else:
            alpha_i = alpha
        x = np.concatenate([[0,0],vecs[i]])
        plt.quiver([x[0]],
                   [x[1]],
                   [x[2]],
                   [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=cols[i],
                  alpha=alpha_i)
    plt.ylim(-xlim, xlim)
    plt.xlim(-ylim, ylim)
    plt.grid()


def plot_vector2d(vector2d, origin=[0, 0], **options):
    return plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1],
              head_width=0.2, head_length=0.3, length_includes_head=True,
              **options)


def plot_transform(P_before, P_after, text_before, text_after, name, color=['#FF9A13', '#1190FF'], axis = [0, 5, 0, 4], arrows=False):
    if arrows:
        for vector_before, vector_after in zip(tf.transpose(P_before), tf.transpose(P_after)):
            plot_vector2d(vector_before, color="#FF9A13", linestyle="--")
            plot_vector2d(vector_after, color="#1190FF", linestyle="-")
    plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'red'})
    plt.gca().add_artist(Polygon(tf.transpose(P_before), alpha=0.2))
    plt.gca().add_artist(Polygon(tf.transpose(P_after), alpha=0.3, color="#FF9A13"))
    plt.text(-.25, 1, text_before, size=18, color=color[1])
    plt.text(1.5, 0, text_after, size=18, color=color[0])
    plt.title(name, color='w')
    plt.axis(axis)
    plt.grid()
