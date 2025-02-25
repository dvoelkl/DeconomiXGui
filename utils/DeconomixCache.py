####################################################
#
# File for storing software data during runtime
#
####################################################

from utils.DTD_config import DTDConfig
from utils.ADTD_config import ADTDConfig

class DCXCache:

    def __init__(self, DeconomixFile=None, DTD_config=None, DTDmodel=None, ADTD_config=None):

        # MISC
        self.DeconomixFile = DeconomixFile
        self.DTDTab = None
        self.ADTDTab = None

        # DTD
        self.DTD_config = DTD_config
        self.DTDmodel = DTDmodel
        self.DTD_Y_train = None
        self.DTD_C_train = None
        self.DTD_C_train_est = None

        self.DTD_Y_test = None
        self.DTD_C_test = None
        self.DTD_C_test_est = None

        self.DTD_Y_appl = None
        self.DTD_C_appl_est = None

        # ADTD
        self.ADTD_config = ADTD_config
        self.ADTD_HPS_model = None
        self.ADTDmodel = None
    def clearDTD(self):
        self.DTDTab = None

        self.DTD_config = DTDConfig()
        self.DTDmodel = None
        self.DTD_Y_train = None
        self.DTD_C_train = None
        self.DTD_C_train_est = None

        self.DTD_Y_test = None
        self.DTD_C_test = None
        self.DTD_C_test_est = None

        self.DTD_Y_appl = None
        self.DTD_C_appl_est = None

    def clearADTD(self):
        self.ADTDTab = None

        self.ADTD_config = ADTDConfig()
        self.ADTD_HPS_model = None
        self.ADTDmodel = None

    def clearAll(self):
        self.clearDTD()
        self.clearADTD()
