
# coding: utf-8

# # Intorduction

# ## What is in this notebook?

# ## Inputs

# The following are the inputs which the model needs to run, please select one of the below for each input:

# In[7]:

# inputs go here


# ## Magics & Versions

# The below table shows the version of libraries and packages used for running the model.

# In[8]:

# Inline matplotlib
get_ipython().magic('matplotlib inline')

# Interactive matplotlib plot()
get_ipython().magic('matplotlib notebook')

# Autoreload packages before runs
# https://ipython.org/ipython-doc/dev/config/extensions/autoreload.html
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# %install_ext http://raw.github.com/jrjohansson/version_information/master/version_information.py
# ~/anaconda/bin/pip install version_information
get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information numpy, scipy, matplotlib, pandas')


# ## Standard imports

# In[9]:

# Standard library
import os
import sys
sys.path.append("../src/")

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Date and time
import datetime
import time

# Ipython imports
from IPython.display import FileLink


# ## Other imports

# In[10]:

# Other imports

# Stats models
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess

from patsy import dmatrices

from sklearn import datasets, svm


# ## Customization

# In[11]:

# Customizations
sns.set() # matplotlib defaults

# Any tweaks that normally go in .matplotlibrc, etc., should explicitly go here
plt.rcParams['figure.figsize'] = (12, 12)


# In[12]:

# Find the notebook the saved figures came from
fig_prefix = "../figures/2019-04-25-mh-titanic"


# In[ ]:



