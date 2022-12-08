import matplotlib.pyplot as plt
import os

def plot_and_fit(x, label_x, y, label_y, title, name, dir):
    plt.title(title)

    print(len(x))
    print(len(y))

    # add y = x line
    plt.plot([min(x), max(x)], [min(x), max(x)], color='lightgray', linestyle='-')

    plt.scatter(x, y, s=1)

    plt.xlabel(label_x)
    plt.ylabel(label_y)

    # save the plot
    plt.savefig(os.path.join(dir, name + ".png"))

    # clear the plot
    plt.clf()