import matplotlib.pyplot as pyplot
import numpy as np


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    #print(f'stroke before penup: {stroke}')
    f, ax = pyplot.subplots()

    #stroke[-1,0] = 1   #penup so we can see the last part of the stroke
    #print(f'stroke after penup: {stroke}')

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    #print('entire stroke', x, y)
    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    #print(f'np.where(stroke[:, 0] == 1): {np.where(stroke[:, 0] == 1)}')
    #print(f'plot_stroke: cuts={cuts}')
    start = 0
    for cut_value in cuts:
        #print('black', start, cut_value)
        #print(x[start:cut_value], y[start:cut_value])
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        # show pen up part in red
        if cut_value + 2 < len(y):
            #print('red', cut_value-1, cut_value+2)
            ax.plot(x[cut_value-1:cut_value+2], y[cut_value-1:cut_value+2],
                    'r-', linewidth=2)
        start = cut_value + 1

    # final stroke. Especially important if there were no penups specified
    last_cut = cuts[-1] if len(cuts) > 0 else 0
    ax.plot(x[last_cut:len(x)], y[last_cut:len(y)],
            'r-', linewidth=2)

    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
      try:
        pyplot.savefig(
            save_name,
            bbox_inches='tight',
            pad_inches=0.5)
      except Exception:
        print("Error building image!: " + save_name)

    return f
