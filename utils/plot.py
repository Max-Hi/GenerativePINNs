
import numpy as np
import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save

def compare_nn_output(nn_output, preexisting_data, labels=['NN Output', 'Preexisting Data'], filename='plot.tex'):
    """
    Compares the output of a neural network with preexisting data by plotting both on the same graph.
    
    :param nn_output: A matrix (2D NumPy array) representing the neural network's output.
    :param preexisting_data: A matrix (2D NumPy array) representing the preexisting data to compare against.preexisting data to compare against.
    :param labels: A list of strings representing the labels for the neural network output and preexisting data.
    :param filename: The filename to which the LaTeX (TikZ/PGFplots) representation of the plot will be saved.
    """
    
    # Assuming both matrices have the same dimensions for simplicity
    assert nn_output.shape == preexisting_data.shape, "nn_output and preexisting_data must have the same shape"
    
    # Generate x values based on the matrix dimensions
    x_values = np.arange(nn_output.shape[1])
    
    # Plotting

    plt.figure()
    for data, label in zip([nn_output, preexisting_data], labels):
        # Assuming each row in the matrices represents a separate data series
        for row in data:
            plt.plot(x_values, row, label=label)
    
    # Adding legend, labels, etc.
    plt.legend()
    plt.title('Comparison of Neural Network Output with Preexisting Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    
    # Export to TikZ/PGFplots
    tikz_save(filename)
    
    # Optionally display the plot
    plt.savefig('1.png')
    plt.show()

# Example usage
nn_output = np.random.rand(2, 10)  # Simulated neural network output
preexisting_data = np.random.rand(2, 10)  # Simulated preexisting data

compare_nn_output(nn_output, preexisting_data)