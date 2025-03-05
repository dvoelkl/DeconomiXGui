import argparse
import pandas as pd
from utils.DeconomixFile import DeconomixFile 

def main():
    parser = argparse.ArgumentParser(description="Create a Deconomix file.")
    parser.add_argument("-X", required=True, help="Path to the X matrix CSV file.")
    parser.add_argument("-train", required=True, help="Path to the training CSV file.")
    parser.add_argument("-traindesc", required=False, help="Description of Training File", default="Dataset used for Training")
    parser.add_argument("-test", required=True, help="Path to the testing CSV file.")
    parser.add_argument("-testdesc", required=False, help="Description of Test File", default="Dataset used for Testing")
    parser.add_argument("-app", required=False, help="Path to the application (bulk) CSV file.", default=None)
    parser.add_argument("-appdesc", required=False, help="Description of Application File", default="Bulk data for Application")
    parser.add_argument("-o", required=True, help="Output path for the Deconomix file (without extension).")
    parser.add_argument("-desc", default="", help="Description of the file.")
    parser.add_argument("-author", default="", help="Author of the file.")
    
    args = parser.parse_args()

    # Load CSV files into pandas DataFrames
    X_mat = pd.read_csv(args.X)
    Train = pd.read_csv(args.train)
    Test = pd.read_csv(args.test)
    if args.app is not None:
        Application = pd.read_csv(args.app)
    else:
        Application = None

    ApplicationDesc = args.appdesc

    # Create DeconomixFile object
    deconomix_file = DeconomixFile(
        X_mat=X_mat,
        Train=Train,
        Test=Test,
        Application=Application,
        description=args.desc,
        author=args.author,
        filename=args.o
    )

    deconomix_file.TestDesc = args.testdesc
    deconomix_file.TrainDesc = args.traindesc
    deconomix_file.ApplicationDesc = ApplicationDesc

    # Serialize and save
    deconomix_file.binarize(args.o)
    print(f"Deconomix file saved as {args.o}.dcx")

if __name__ == "__main__":
    main()
