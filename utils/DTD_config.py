####################################################
#
# File for storing DTD Parameters
#
####################################################

class DTDConfig:

    def __init__(self, train_n_mixtures=1000, train_n_cells=100, test_n_mixtures=1000, test_n_cells=100, fRunOnAppl=False, nIter=1000):

        self.train_n_mixtures = train_n_mixtures
        self.train_n_cells = train_n_cells

        self.test_n_mixtures = test_n_mixtures
        self.test_n_cells = test_n_cells

        self.fRunOnAppl = fRunOnAppl

        self.nIter = nIter