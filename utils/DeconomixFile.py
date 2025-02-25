####################################################
#
# File for exchanging Deconomix Training and Testing
# data
#
####################################################
import base64
import pickle

import anndata
from anndata import AnnData
import numpy as np
import pandas as pd
import tempfile

__VERSION__ = 1.0

class DeconomixFile:
    '''
    Proprietary file format of Deconomix used for storing and exchanging information about training and test data
    '''

    def __init__(self,
                 X_mat=None,
                 Train=None, Train_Description="Dataset used for Training - You can personalize this text",
                 Test=None, Test_Description="Dataset used for Testing - You can personalize this text",
                 Application=None, Application_Description="Dataset on which the trained model should be applied to - You can personalize this text",
                 Gamma=None,
                 description="",
                 author="",
                 filename=""):
        '''
        Constructor for DeconomixFile
        :param X_mat: Data matrix
        :param Train: Training data
        :param Test: Testing data
        :param Application: Application Data
        :param Gamma: Stored Gamma, if already calculated (might be unused, not sure)
        :param description: File description
        :param author: File author
        '''
        self.Description = description
        self.Author = author

        self.X_mat = X_mat

        self.Train = Train
        self.TrainDesc = Train_Description

        self.Test = Test
        self.TestDesc = Test_Description

        self.Application=Application
        self.ApplicationDesc = Application_Description

        self.Gamma = Gamma

        self.filename = filename

        self.__VERSION__ = __VERSION__

    def binarize(self, file_path):
        '''
        Serializes the object and saves it to the specified path
        :param file_path: Path where the binary file will be saved
        '''
        with open(file_path + ".dcx", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def unbinarize_from_path(file_path):
        with open(file_path, "rb") as file:
            dcx = pickle.load(file)

        if dcx.__VERSION__ != __VERSION__:
            raise DeconomiXFileVersionException("The loaded Version of the DeconomiX file is outdated!", __VERSION__)

        return dcx

    @staticmethod
    def unbinarize(file):
        dcx = pickle.load(file)

        if dcx.__VERSION__ != __VERSION__:
            raise DeconomiXFileVersionException("The loaded Version of the DeconomiX file is outdated!", __VERSION__)

        return dcx

    @staticmethod
    def from_AnnData(X, train, test, application=None,
                              description="", author="", filename=""):
        def parse_contents(contents):
            if contents is not None:
                content_type, content_string = contents.split(',')

                decoded = base64.b64decode(content_string)
                with tempfile.NamedTemporaryFile(delete=True, suffix='.h5ad') as tmp_file:
                    tmp_file.write(decoded)
                    tmp_file_path = tmp_file.name

                    adata = anndata.read_h5ad(tmp_file_path)

                return adata.to_df()
            else:
                return None

        return DeconomixFile(
            X_mat=parse_contents(X),
            Train=parse_contents(train),
            Test=parse_contents(test),
            Application=parse_contents(application),
            description=description,
            author=author,
            filename=filename
        )

    def to_contents_string(self):
        pickled_data = pickle.dumps(self)

        base64_encoded_data = base64.b64encode(pickled_data).decode('utf-8')
        return f"data:application/octet-stream;base64,{base64_encoded_data}"


class DeconomiXFileVersionException(Exception):
    def __init__(self, message, currentVersion):
        super().__init__(message)

        self.currentVersion = currentVersion