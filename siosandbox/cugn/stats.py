""" Perform stats on this and that """


import numpy as np
from numpy.testing import suppress_warnings
from operator import index

import builtins
from collections import namedtuple

from IPython import embed

#def gaussian_cdf():

BinnedStatisticResult = namedtuple('BinnedStatisticResult',
                                   ('statistic', 'bin_edges', 'binnumber'))

