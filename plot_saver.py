import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_n_save(files, labels, filename, show = False):
    """
    Function to save plots based on the directories given in files and labels given in labels.
    The plots are saved in the Plots directory with the name filename.

    Parameters
    ----------
    files : list
        List of directories where the plots are to be saved
    labels : list
        List of information for the plots
    filename : str
        Name of the file to be saved
    show : bool, Default False
        Whether to show the plot or not
    """
    plt.figure(figsize=(10, 10))
    for i in range(len(files)):
        directory = files[i]
        legend_label = labels[i]

        df = pd.read_csv(directory)
        
        # Get the first column of df
        x = df.iloc[:, 0]
        # Get the second column of df
        y = df.iloc[:, 1]

        plt.plot(x, y, label = legend_label)
        
    plt.xlabel('Models')
    plt.ylabel('Average Perplexity')
    plt.title('Models vs Average Perplexity')
    plt.legend()

    # Check if a directory named "Plots" exists and create it if it doesn't
    if not os.path.exists('Plots'):
        os.makedirs('Plots')

    plt.savefig('Plots\\' + filename + '.png')
    if show:
        plt.show()
    plt.close()

def caller():
    """
    Function to call the plot_n_save function
    """
    # Create a list of the directories where to fetch the data fram and the labels for the plots

    # Plot and save Turing Smoothing 
    files = ['average_perplexity\\avg_perplexity_turing_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_turing_smoothing_train.csv']
    labels = ['Turing Smoothing Test',
              'Turning Smoothing Train']
    filename = 'Turing'
    plot_n_save(files, labels, filename, show = False)

    # Plot and save Laplace Smoothing
    files = ['average_perplexity\\avg_perplexity_laplace_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_laplace_smoothing_train.csv']
    labels = ['Laplace Smoothing Test',
              'Laplace Smoothing Train']
    filename = 'Laplace'
    plot_n_save(files, labels, filename, show = False)

    # Plot and save None Smoothing
    files = ['average_perplexity\\avg_perplexity_None_smoothing_train.csv']
    labels = ['No Smoothing Train']
    filename = 'No_Smoothing'
    plot_n_save(files, labels, filename, show = False)

    # Plot and save Additive Smoothing for all values of K
    files = ['average_perplexity\\avg_perplexity_additive_0.01_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_0.1_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_10_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_100_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_0.01_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_0.1_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_10_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_100_smoothing_train.csv']
    labels = ['Additive Smoothing with K = 0.01 Test',
              'Additive Smoothing with K = 0.1 Test',
              'Additive Smoothing with K = 10 Test',
              'Additive Smoothing with K = 100 Test',
              'Additive Smoothing with K = 0.01 Train',
              'Additive Smoothing with K = 0.1 Train',
              'Additive Smoothing with K = 10 Train',
              'Additive Smoothing with K = 100 Train']
    filename = 'Additive_Smoothing'
    plot_n_save(files, labels, filename, show = False)

    # Plot and save Additive Smoothing and Laplace Smoothing for all values of K
    files = ['average_perplexity\\avg_perplexity_additive_0.01_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_0.1_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_10_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_100_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_0.01_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_0.1_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_10_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_100_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_laplace_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_laplace_smoothing_train.csv']
    labels = ['Additive Smoothing with K = 0.01 Test',
              'Additive Smoothing with K = 0.1 Test',
              'Additive Smoothing with K = 10 Test',
              'Additive Smoothing with K = 100 Test',
              'Additive Smoothing with K = 0.01 Train',
              'Additive Smoothing with K = 0.1 Train',
              'Additive Smoothing with K = 10 Train',
              'Additive Smoothing with K = 100 Train',
              'Laplace Smoothing Test',
              'Laplace Smoothing Train']
    filename = 'Additive_Laplace_Smoothing'
    plot_n_save(files, labels, filename, show = False)

    # Plot and save None Smoothing and Turing Smoothing
    files = ['average_perplexity\\avg_perplexity_None_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_turing_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_turing_smoothing_train.csv']
    labels = ['No Smoothing Train',
              'Turing Smoothing Test',
              'Turing Smoothing Train']
    filename = 'Turing_No_Smoothing'
    plot_n_save(files, labels, filename, show = False)

    # Plot and save None Smoothing and Laplace Smoothing
    files = ['average_perplexity\\avg_perplexity_None_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_laplace_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_laplace_smoothing_train.csv']
    labels = ['No Smoothing Train',
              'Laplace Smoothing Test',
              'Laplace Smoothing Train']
    filename = 'Laplace_No_Smoothing'
    plot_n_save(files, labels, filename, show = False)

    # Plot and save None Smoothing and Additive Smoothing for all values of K
    files = ['average_perplexity\\avg_perplexity_additive_0.01_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_0.1_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_10_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_100_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_0.01_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_0.1_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_10_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_100_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_None_smoothing_train.csv']
    labels = ['Additive Smoothing with K = 0.01 Test',
              'Additive Smoothing with K = 0.1 Test',
              'Additive Smoothing with K = 10 Test',
              'Additive Smoothing with K = 100 Test',
              'Additive Smoothing with K = 0.01 Train',
              'Additive Smoothing with K = 0.1 Train',
              'Additive Smoothing with K = 10 Train',
              'Additive Smoothing with K = 100 Train',
              'No Smoothing Train']
    filename = 'Additive_No_Smoothing'
    plot_n_save(files, labels, filename, show = False)

    # Plot and save Laplace Smoothing and Turing Smoothing 
    files = ['average_perplexity\\avg_perplexity_laplace_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_laplace_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_turing_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_turing_smoothing_train.csv']
    labels = ['Laplace Smoothing Test',
              'Laplace Smoothing Train',
              'Turing Smoothing Test',
              'Turing Smoothing Train']
    filename = 'Laplace_Turing_Smoothing'
    plot_n_save(files, labels, filename, show = False)

    # Plot and save Turing and Additive Smoothing for all values of K
    files = ['average_perplexity\\avg_perplexity_additive_0.01_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_0.1_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_10_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_100_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_additive_0.01_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_0.1_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_10_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_additive_100_smoothing_train.csv',
             'average_perplexity\\avg_perplexity_turing_smoothing_test.csv',
             'average_perplexity\\avg_perplexity_turing_smoothing_train.csv']
    labels = ['Additive Smoothing with K = 0.01 Test',
              'Additive Smoothing with K = 0.1 Test',
              'Additive Smoothing with K = 10 Test',
              'Additive Smoothing with K = 100 Test',
              'Additive Smoothing with K = 0.01 Train',
              'Additive Smoothing with K = 0.1 Train',
              'Additive Smoothing with K = 10 Train',
              'Additive Smoothing with K = 100 Train',
              'Turing Smoothing Test',
              'Turing Smoothing Train']
    
    filename = 'Additive_Turing_Smoothing'
    plot_n_save(files, labels, filename, show = False)

caller()