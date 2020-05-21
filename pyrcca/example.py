# Imports
import numpy as np
from pyrcca import rcca


def generate_sample_data():
    # Initialize number of samples
    nSamples = 1000

    # Define two latent variables (number of samples x 1)
    latvar1 = np.random.randn(nSamples, )
    latvar2 = np.random.randn(nSamples, )

    # Define independent components for each dataset (number of observations x dataset dimensions)
    indep1 = np.random.randn(nSamples, 4)
    indep2 = np.random.randn(nSamples, 5)

    # Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
    data1 = 0.25 * indep1 + 0.75 * np.vstack((latvar1, latvar2, latvar1, latvar2)).T
    data2 = 0.25 * indep2 + 0.75 * np.vstack((latvar1, latvar2, latvar1, latvar2, latvar1)).T
    print(data1.shape, data2.shape)
    # Split each dataset into two halves: training set and test set
    train1 = data1[:int(nSamples / 2)]
    train2 = data2[:int(nSamples / 2)]

    test1 = data1[int(nSamples / 2):]
    test2 = data2[int(nSamples / 2):]

    print(train1.shape, train2.shape, test1.shape, test2.shape)
    return train1, train2, test1, test2


def get_cca_result(train_list, test_list):
    # Create a cca object as an instantiation of the CCA object class.
    cca = rcca.CCA(kernelcca=False, reg=0., numCC=2)

    # Use the train() method to find a CCA mapping between the two training sets.
    cca.train(train_list)

    # Use the validate() method to test how well the CCA mapping generalizes to the test data.
    # For each dimension in the test data, correlations between predicted and actual data are computed.
    testcorrs = cca.validate(test_list)
    print(testcorrs)
    return testcorrs


train1, train2, test1, test2 = generate_sample_data()
result = get_cca_result([train1, train2], [test1, test2])
print(result)
