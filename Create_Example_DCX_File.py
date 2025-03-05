from utils.DeconomixFile import DeconomixFile
from deconomix.utils import load_example, simulate_data
import pandas as pd
import anndata


test, train = load_example()
X, _, _ = simulate_data(train, 1000, 100)

# Create example DeconomixFile and set Test, Train and Reference
example = DeconomixFile(X_mat=X, Train=train, Test=test, description="Example DeconomiX file for testing the GUI", author="dvo")

# Save file to disc (.dcx file format)
example.binarize(f"./example")


# New example for ADTD
example = DeconomixFile(X_mat=X.drop('B', axis=1, inplace=False), Train=train.drop('B', axis=1, inplace=False), Test=test, description="Example DeconomiX file for testing the GUI", author="dvo")

# Regulate some genes for application
regulated_genes = ['CXCR4', 'CD3D', 'FTH1', 'PTPRCAP', 'IL32', 'SRGN', 'PTPRC']
test_regulated = test.copy()
test_regulated.loc[regulated_genes, "T"] = 2 * test.loc[regulated_genes, "T"]
_, Y_appl, _ = simulate_data(test_regulated, 1000, 100)
example.Application = Y_appl
example.ApplicationDesc = "Simulated Application Data with Gene Upregulation in T Cells"
example.binarize(f"./example_ADTD")

anndata.AnnData(example.X_mat).write_h5ad('AnnDataExample/X.h5ad')
anndata.AnnData(example.Train).write_h5ad('AnnDataExample/Train.h5ad')
anndata.AnnData(example.Test).write_h5ad('AnnDataExample/Test.h5ad')
anndata.AnnData(example.Application).write_h5ad('AnnDataExample/Application.h5ad')


# Load file
#example = DeconomixFile.unbinarize(f"./example.dcx")

print("")