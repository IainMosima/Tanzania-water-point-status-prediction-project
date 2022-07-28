# function to help with importing sets from thr analysis-df
def set_importer(path, y=False):
    import pandas as pd
    set = pd.read_csv(path)
    set.drop('Unnamed: 0', axis=1, inplace=True)

    if y == True:
        return set.squeeze()

    return set