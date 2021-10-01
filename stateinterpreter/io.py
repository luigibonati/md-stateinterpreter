import pandas as pd

__all__ = ["load_dataframe", "plumed_to_pandas"]

def is_plumed_file(filename):
    """
    Check if given file is in PLUMED format.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file

    Returns
    -------
    bool
        wheter is a plumed output file
    """
    headers = pd.read_csv(filename, sep=" ", skipinitialspace=True, nrows=0)
    is_plumed = True if " ".join(headers.columns[:2]) == "#! FIELDS" else False
    return is_plumed


def plumed_to_pandas(filename="./COLVAR"):
    """
    Load a PLUMED file and save it to a dataframe.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file

    Returns
    -------
    df : DataFrame
        Collective variables dataframe
    """
    skip_rows = 1
    # Read header
    headers = pd.read_csv(filename, sep=" ", skipinitialspace=True, nrows=0)
    # Discard #! FIELDS
    headers = headers.columns[2:]
    # Load dataframe and use headers for columns names
    df = pd.read_csv(
        filename,
        sep=" ",
        skipinitialspace=True,
        header=None,
        skiprows=range(skip_rows),
        names=headers,
        comment="#",
    )

    return df


def load_dataframe(data, **kwargs):
    """Load dataframe from object or from file.

    Parameters
    ----------
    data : str or pandas.DataFrame
        input data

    Returns
    -------
    pandas.DataFrame
        Dataframe

    Raises
    ------
    TypeError
        if data is not a valid type
    """
    # check if data is Dataframe
    if type(data) == pd.DataFrame:
        df = data
    # or is a string
    elif type(data) == str:
        filename = data
        # check if file is in PLUMED format
        if is_plumed_file(filename):
            df = plumed_to_pandas(filename)
        # else use read_csv with optional kwargs
        else:
            df = pd.read_csv(filename, **kwargs)
    else:
        raise TypeError(f"{data}: Accepted types are 'pandas.Dataframe' or 'str'")

    return df
