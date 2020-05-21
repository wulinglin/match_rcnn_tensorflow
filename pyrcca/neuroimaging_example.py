#%% md

# Pyrcca: regularized kernel canonical correlation analysis in Python and its applications to neuroimaging.

#%% md

#%% md

## Imports for neuroimaging data analysis

#%%

import h5py
import rcca
import sys
import numpy as np
import cortex
zscore = lambda d: (d-d.mean(0))/d.std(0)

#%% md

## Load data from CRCNS

#%% md

#%%

data = []
vdata = []
numSubjects = 3
# subjects is a list of subject names in Pycortex corresponding to the three subjects in the analysis.
subjects = ['S1', 'S2', 'S3']
# xfms is a list of transform names in Pycortex aligning the functional and anatomical data for each subject.
xfms = ['S1_xfm', 'S2_xfm', 'S3_xfm']
dataPath ="./data/VoxelResponses_subject%d.mat"
for subj in range(numSubjects):
    # Open data file
    f = h5py.File(dataPath % (subj+1),'r')
    # Get size of the data
    datasize = (int(f["ei"]["datasize"].value[2]),int(f["ei"]["datasize"].value[1]),int(f["ei"]["datasize"].value[0]))
    # Get the cortical mask from Pycortex
    mask = cortex.db.get_mask(subjects[subj], xfms[subj], type = 'thick')
    # Get the training data for the subject
    data_subj = np.nan_to_num(zscore(np.nan_to_num(f["rt"].value.T)))
    data.append(data_subj.reshape((data_subj.shape[0],)+datasize)[:, mask])
    # Get the validation data for the subject
    vdata_subj = np.nan_to_num(zscore(np.nan_to_num(f["rv"].value.T)))
    vdata.append(vdata_subj.reshape((vdata_subj.shape[0],)+datasize)[:, mask])

#%% md

## Define CCA parameters

#%%

# We will consider a range of regularization values betewen 1e-4 and 1e2
regs = np.array(np.logspace(-4, 2, 10))


# We will consider numbers of components between 3 and 10
numCCs = np.arange(3, 11)

# Initialize the cca object
cca = rcca.CCACrossValidate(numCCs=numCCs, regs=regs)

#%% md

## Run and save the analysis

#%%

# NOTE: this analysis is computationally intensive due to the size of data. Running it in the notebook
# would take a considerable amount of time, so we recommend parallelizing it and/or running
# it on a computer cluster, and then loading in the results for visualization.

# Train the CCA mapping on training data
cca.train(data)

# Validate the CCA mapping on validation data
cca.validate(vdata)

# Compute variance explained for validation responses in each voxel
cca.compute_ev(vdata)

# Save analysis results
cca.save("./data/CCA_results.hdf5")

#%% md

## Visualize results for one of the subjects

#%% md

### Plot correlation histogram

#%% md

#%%

# Visualization imports

import matplotlib.pyplot as plt

# Import Brewer colormaps for visualization
from brewer2mpl import qualitative

nSubj = len(cca.corrs)
nBins = 30
bmap = qualitative.Set1[nSubj]
f = plt.figure(figsize = (8, 6))
ax = f.add_subplot(111)
for s in range(nSubj):
    # Plot histogram of correlations across all voxels for all three subjects
    ax.hist(cca.corrs[s], bins = nBins, color = bmap.mpl_colors[s], histtype="stepfilled", alpha = 0.6)
plt.legend(['Subject 1', 'Subject 2', 'Subject 3'], fontsize = 16)
ax.set_xlabel('Prediction correlation', fontsize = 20)
ax.set_ylabel('Number of voxels', fontsize = 20)
ax.set_title("Prediction performance across voxels", fontsize = 20)
# Significance threshold at p<0.05 (corrected for multiple comparisons
# Significance is calculated using an asymptotic method (see paper text for detail)
thresh = 0.0893
ax.axvline(x = thresh, ymin = 0, ymax = 7000, linewidth = 2, color = 'k')
ax.text(thresh+0.05, 5000, 'p<0.05', fontsize = 16)
ax.set_xticklabels(0.1*np.arange(-8, 11, 2), fontsize = 16)
ax.set_yticklabels(np.arange(0, 10000, 1000), fontsize = 16)

#%% md

### Cortical map plots

#%% md


import cortex
from matplotlib import cm
from copy import deepcopy
subj = 0
subjName = "S1"
subjTransform = "S1_xfm"
corrs = deepcopy(cca.corrs[subj])
# Set all voxels below the signficance threshold to 0
corrs[corrs<thresh] = 0
_ = cortex.quickflat.make_figure(cortex.Volume(corrs, subjName, subjTransform, cmap = cm.PuBuGn_r, vmin = 0., vmax = 1.), with_curvature = True)

#%% md

##### Canonical component RGB map


rescale = lambda d: 1/(d.max() - d.min())*(d - d.min())

_ = cortex.quickflat.make_figure(cortex.VolumeRGB(rescale(np.abs(cca.ws[0].T[0])), rescale(np.abs(cca.ws[0].T[1])), rescale(np.abs(cca.ws[0].T[2])), 'SNfs', 'SNfs4Tnb'), with_curvature = True, with_colorbar = False)

#%% md



maxmins = [15, 50, 40]
for i in range(cca.best_numCC):
    cortex.quickflat.make_figure(cortex.Volume2D(np.nan_to_num(cca.ws[subj].T[i]), np.nan_to_num(cca.ev[subj][i]), subjName, subjTransform, cmap = "GreenWhiteBlue_2D", vmin = -maxmins[i], vmax = maxmins[i], vmin2 = 0, vmax2 = 0.75), with_curvature = True)

#%%


