####################################################
#
# File for storing ADTD Parameters
#
####################################################

class ADTDConfig:

    def __init__(self, Cstatic=False, lambda1=1e-6, Deltastatic=False, lambda2=1e-6, dataset='test', niter=1000, lambda2min=1e-10, lambda2max=1, nPoints=13):

        self.Cstatic = Cstatic
        self.lambda1 = lambda1

        self.Deltastatic = Deltastatic
        self.lambda2 = lambda2

        self.lambda2min = lambda2min
        self.lambda2max = lambda2max
        self.nPoints = nPoints

        self.Dataset = dataset

        self.nIter = niter