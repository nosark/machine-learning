
__author__ = 'kyle nosar'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from adaline import AdalineGD


def main():
    try:
        df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)   #load data into dataframes
        print('Visual Check: \n %s' % df.tail())   #make sure visually that the data loaded correctly by checking the last 5 elements
    except Exception as e:
        print('Failed to load the data into dataframes properly!: \n %s' % e)
        raise


    y = df.iloc[0:100, 4].values
    y = np.where(y =='Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    #above loads in iris dataset and should set our values to use for gradient
    #descent
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (8,4))
    ada1 = AdalineGD(eta=0.01, n_iters=10).fit(X,y)
    ax[0].plot(range(1,len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(n_iters=10, eta=0.0001).fit(X,y)
    ax[1].plot(range(1,len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.00001')
    plt.show()
    plt.pause(10)


if __name__ == '__main__':
    main()
