# %% md

# Usage example of pyrcca


# %% md

## Pyrcca with predefined hyperparameters

# %% md

### Initialization

# %%

# Imports
import numpy as np
import rcca

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

# Split each dataset into two halves: training set and test set
train1 = data1[:nSamples // 2]
train2 = data2[:nSamples // 2]
test1 = data1[nSamples // 2:]
test2 = data2[nSamples // 2:]



# Create a cca object as an instantiation of the CCA object class.
nComponents = 4
cca = rcca.CCA(kernelcca=False, reg=0., numCC=nComponents)

# Use the train() method to find a CCA mapping between the two training sets.
cca.train([train1, train2])

# Use the validate() method to test how well the CCA mapping generalizes to the test data.
# For each dimension in the test data, correlations between predicted and actual data are computed.
testcorrs = cca.validate([test1, test2])

# %% md

### Results

# %%

# Imports
import matplotlib.pyplot as plt
# from brewer2mpl import qualitative
from palettable.colorbrewer import qualitative

# %% md

#### Canonical correlations

# %%

# Plot canonical correlations (cca.cancorrs)
plt.plot(np.arange(nComponents) + 1, cca.cancorrs, 'ko')
plt.xlim(0.5, 0.5 + nComponents)
plt.xticks(np.arange(nComponents) + 1)
plt.xlabel('Canonical component')
plt.ylabel('Canonical correlation')
plt.title('Canonical correlations')
print('''The canonical correlations are:\n
Component 1: %.02f\n
Component 2: %.02f\n
Component 3: %.02f\n
Component 4: %.02f\n
''' % tuple(cca.cancorrs))

# %% md


# %% md

#### Cross-dataset predictions

# %%

# Plot correlations between actual test data and predictions
# obtained by projecting the other test dataset via the CCA mapping for each dimension.
nTicks = max(testcorrs[0].shape[0], testcorrs[1].shape[0])
# bmap1 = qualitative.Dark2[3]
bmap1 = qualitative.Dark2_3
plt.plot(np.arange(testcorrs[0].shape[0]) + 1, testcorrs[0], 'o', color=bmap1.mpl_colors[0])
plt.plot(np.arange(testcorrs[1].shape[0]) + 1, testcorrs[1], 'o', color=bmap1.mpl_colors[1])
plt.xlim(0.5, 0.5 + nTicks + 3)
plt.ylim(0.0, 1.0)
plt.xticks(np.arange(nTicks) + 1)
plt.xlabel('Dataset dimension')
plt.ylabel('Prediction correlation')
plt.title('Prediction accuracy')
plt.legend(['Dataset 1', 'Dataset 2'])
print('''The prediction accuracy for the first dataset is:\n
Dimension 1: %.02f\n
Dimension 2: %.02f\n
Dimension 3: %.02f\n
Dimension 4: %.02f\n
''' % tuple(testcorrs[0]))
print('''The prediction accuracy for the second dataset is:\n
Dimension 1: %.02f\n
Dimension 2: %.02f\n
Dimension 3: %.02f\n
Dimension 4: %.02f\n
Dimension 5: %.02f\n
''' % tuple(testcorrs[1]))



# %% md

## Pyrcca with hyperparameter grid search using cross-validation


# %% md

### Analysis

# %%

# Initialize a cca object as an instantiation of the CCACrossValidate class.
ccaCV = rcca.CCACrossValidate(kernelcca=False, numCCs=[1, 2, 3, 4],
                              regs=[0., 1e2, 1e4, 1e6])

# Use the train() and validate() methods to run the analysis and perform cross-dataset prediction.
ccaCV.train([train1, train2])
testcorrsCV = ccaCV.validate([test1, test2])

# %%

print('Optimal number of components: %d\nOptimal regularization coefficient: %d' % (ccaCV.best_numCC, ccaCV.best_reg))

# %% md

# %% md

### Results

# %% md

#### Canonical correlations

# %%

# Plot canonical correlations (cca.cancorrs)
plt.plot(np.arange(ccaCV.best_numCC) + 1, ccaCV.cancorrs, 'ko')
plt.xlim(0.5, 0.5 + ccaCV.best_numCC)
plt.xticks(np.arange(ccaCV.best_numCC) + 1)
plt.ylim(0, 1)
plt.xlabel('Canonical component')
plt.ylabel('Canonical correlation')
plt.title('Canonical correlations')
print('''The canonical correlations are:\n
Component 1: %.02f\n
Component 2: %.02f\n
''' % tuple(ccaCV.cancorrs))

# %% md

# %% md

#### Cross-dataset predictions

# %%

# Plot correlations between actual test data and predictions
# obtained by projecting the other test dataset via the CCA mapping for each dimension.
nTicks = max(testcorrsCV[0].shape[0], testcorrsCV[1].shape[0])
bmap1 = qualitative.Dark2_3
plt.plot(np.arange(testcorrsCV[0].shape[0]) + 1, testcorrsCV[0], 'o', color=bmap1.mpl_colors[0])
plt.plot(np.arange(testcorrsCV[1].shape[0]) + 1, testcorrsCV[1], 'o', color=bmap1.mpl_colors[1])
plt.xlim(0.5, 0.5 + nTicks + 3)
plt.ylim(0.0, 1.0)
plt.xticks(np.arange(nTicks) + 1)
plt.xlabel('Dataset dimension')
plt.ylabel('Prediction correlation')
plt.title('Prediction accuracy')
plt.legend(['Dataset 1', 'Dataset 2'])
print('''The prediction accuracy for the first dataset is:\n
Dimension 1: %.02f\n
Dimension 2: %.02f\n
Dimension 3: %.02f\n
Dimension 4: %.02f\n
''' % tuple(testcorrsCV[0]))
print('''The prediction accuracy for the second dataset is:\n
Dimension 1: %.02f\n
Dimension 2: %.02f\n
Dimension 3: %.02f\n
Dimension 4: %.02f\n
Dimension 5: %.02f\n
''' % tuple(testcorrsCV[1]))

# %% md


# %% md

# %%

nIterations = 1000
all_numCC = np.zeros((nIterations,))
all_reg = np.zeros((nIterations,))

for ii in range(nIterations):
    # Initialize number of samples
    nSamples = 1000

    # Define two latent variables (number of samples x 1)
    latvar1 = np.random.randn(nSamples, )
    latvar2 = np.random.randn(nSamples, )

    # Define independent components for each dataset (number of samples x dataset dimensions)
    indep1 = np.random.randn(nSamples, 4)
    indep2 = np.random.randn(nSamples, 5)

    # Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
    data1 = 0.25 * indep1 + 0.75 * np.vstack((latvar1, latvar2, latvar1, latvar2)).T
    data2 = 0.25 * indep2 + 0.75 * np.vstack((latvar1, latvar2, latvar1, latvar2, latvar1)).T

    # Split each dataset into two halves: training set and test set
    train1 = data1[:nSamples // 2]
    train2 = data2[:nSamples // 2]
    test1 = data1[nSamples // 2:]
    test2 = data2[nSamples // 2:]

    # Analysis
    # Initialize a cca object as an instantiation of the CCACrossValidate class.
    ccaCV = rcca.CCACrossValidate(kernelcca=False, numCCs=[1, 2, 3, 4], regs=[0., 1e2, 1e4, 1e6], verbose=False)

    # Use the train() and validate() methods to run the analysis and perform cross-dataset prediction.
    ccaCV.train([train1, train2])

    all_numCC[ii] = ccaCV.best_numCC
    all_reg[ii] = ccaCV.best_reg

# %%

print('''Number of times each number of components was chosen:\n
1 component: %d\n
2 components: %d\n
3 components: %d\n
4 components: %d\n
''' % tuple([(all_numCC == i).sum() for i in (1, 2, 3, 4)]))


print('''Number of times each regularization coefficient was chosen:\n
Regularization coefficient = 0: %d\n
Regularization coefficient = 1e2: %d\n
Regularization coefficient = 1e4: %d\n
Regularization coefficient = 1e6: %d\n
''' % tuple([(all_reg == i).sum() for i in (0, 1e2, 1e4, 1e6)]))

# %% md

