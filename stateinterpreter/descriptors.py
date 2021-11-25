# imports
import numpy as np
import pandas as pd
import mdtraj as md
import itertools
from warnings import warn

from .io import load_dataframe
from ._configs import *



__all__ = ["descriptors_from_traj"]


def descriptors_from_traj(traj_dict, descriptors = ['ca', 'dihedrals', 'hbonds_distances', 'hbonds_contacts'], start=0, stop=None, stride=1):
    """ "Load trajectory with mdtraj.

    Parameters
    ----------
    traj_dict : dict
        dictionary containing trajectory and topology (optional) file
    descriptors : bool or list, default True
        compute list of descriptors. if True compute all descriptors: ['ca','dihedral','hbonds'].
        if False no descriptor is computed.

    """
    traj_file = traj_dict["trajectory"]
    topo_file = traj_dict["topology"] if "topology" in traj_dict else None

    if type(traj_file) == list:
        traj_list = []
        for traj in traj_file:
            tmp_traj = md.load(traj, top=topo_file, stride=stride)
            if stop is not None:
                tmp_traj = tmp_traj[int(start/stride) : int(stop/stride)]
            else: 
                tmp_traj = tmp_traj[int(start/stride) : ]
            traj_list.append(tmp_traj)
            
        traj = md.join(traj_list)
    else:
        traj = md.load(traj_file, top=topo_file, stride=stride)
        if stop is not None:
            traj = traj[int(start/stride) : int(stop/stride)]
        else: 
            traj = traj[int(start/stride) : ]

    return _compute_descriptors(traj, descriptors)


def _compute_descriptors(traj, descriptors):

    """Compute descriptors from trajectory:
    - Dihedral angles
    - CA distances
    - Hydrogen bonds distances
    - Hydrogen bonds contacts

    Parameters
    ----------
    descriptors : string or list
        compute a single descriptor or a list ot them

    Raises
    ------
    KeyError
        Trajectory needs to be set beforehand.
    """

    descr_list = []

    for d in descriptors:
        if d == 'ca':
            descr_list.append(_CA_DISTANCES(traj))
        elif d == 'dihedrals':
            for angle in ['phi', 'psi', 'chi1', 'chi2']:
                descr_list.append(_DIHEDRALS(traj, kind=angle, sincos=True))
        elif d == 'hbonds_distances':
            descr_list.append(_HYDROGEN_BONDS(traj, 'distances'))
        elif d == 'hbonds_contacts':
            descr_list.append(_HYDROGEN_BONDS(traj, 'contacts'))
        else:
            raise KeyError(f"descriptor: {d} not valid. Only 'ca','dihedrals','hbonds' are allowed.")
    
    #descriptors = pd.concat(descr_list, axis=1)
    if __DEV__:
        #print(f"Descriptors: {descriptors.shape}")
        pass
    return descr_list  

# DESCRIPTORS COMPUTATION
def _CA_DISTANCES(traj):
    descriptors_ids = {}
    if __DEV__:
        print(f"Computing CA distances")
    sel = traj.top.select("name CA")

    pairs = [(i, j) for i, j in itertools.combinations(sel, 2)]
    dist = md.compute_distances(traj, np.array(pairs, dtype=int))

    # Labels
    label = lambda i, j: "DIST. %s%s -- %s%s" % (
        traj.top.atom(i),
        "s" if traj.top.atom(i).is_sidechain else "",
        traj.top.atom(j),
        "s" if traj.top.atom(j).is_sidechain else "",
    )

    names = [label(i, j) for (i, j) in pairs]
    for (i,j) in pairs:
        descriptors_ids[label(i, j)] = [i,j]
    return dist, names, descriptors_ids

def _HYDROGEN_BONDS(traj, kind):
    # H-BONDS DISTANCES / CONTACTS (donor-acceptor)
    # find donors (OH or NH)
    if __DEV__:
        print(f"Computing Hydrogen bonds {kind}")

    donors = [
        at_i.index
        for at_i, at_j in traj.top.bonds
        if ((at_i.element.symbol == "O") | (at_i.element.symbol == "N"))
        & (at_j.element.symbol == "H")
    ]
    # Keep unique
    donors = sorted(list(set(donors)))
    if __DEV__:
        print("Donors:", donors)

    # Find acceptors (O r N)
    acceptors = traj.top.select("symbol O or symbol N")
    if __DEV__:
        print("Acceptors:", acceptors)

    # lambda func to avoid selecting interaction within the same residue
    atom_residue = lambda i: str(traj.top.atom(i)).split("-")[0]
    # compute pairs
    pairs = [
        (min(x, y), max(x, y))
        for x in donors
        for y in acceptors
        if (x != y) and (atom_residue(x) != atom_residue(y))
    ]
    # remove duplicates
    pairs = sorted(list(set(pairs)))

    
    # compute distances
    dist = md.compute_distances(traj, pairs)
    if kind == 'distances':
        descriptors_ids = {}
        # labels
        label = lambda i, j: "HB_DIST %s%s -- %s%s" % (
            traj.top.atom(i),
            "s" if traj.top.atom(i).is_sidechain else "",
            traj.top.atom(j),
            "s" if traj.top.atom(j).is_sidechain else "",
        )

        # basename = 'hb_'
        # names = [ basename+str(x)+'-'+str(y) for x,y in  pairs]
        names = [label(x, y) for x, y in pairs]
        for (x,y) in pairs:
            descriptors_ids[label(x,y)] = [x,y]   
        return dist, names, descriptors_ids
    elif kind == 'contacts':
        descriptors_ids = {}
        def contact_function(x, r0=1.0, d0=0, n=6, m=12):
            # (see formula for RATIONAL) https://www.plumed.org/doc-v2.6/user-doc/html/switchingfunction.html
            return (1 - np.power(((x - d0) / r0), n)) / (1 - np.power(((x - d0) / r0), m)) 
        # Compute contacts
        contacts = contact_function(dist, r0=0.35, d0=0, n=6, m=12)
        # labels
        # basename = 'hbc_'
        # names = [ basename+str(x)+'-'+str(y) for x,y in pairs]
        label = lambda i, j: "HB_CONTACT %s%s -- %s%s" % (
            traj.top.atom(i),
            "s" if traj.top.atom(i).is_sidechain else "",
            traj.top.atom(j),
            "s" if traj.top.atom(j).is_sidechain else "",
        )
        names = [label(x, y) for x, y in pairs]
        for (x,y) in pairs:
            descriptors_ids[label(x,y)] = [x,y]

        return contacts, names, descriptors_ids
    else:
        raise KeyError(f'kind="{kind}" not allowed. Valid values: "distances","contacts".')

def _DIHEDRALS(traj, kind, sincos=True):
    # Get topology
    table, _ = traj.top.to_dataframe()

    if kind == "phi":
        dih_idxs, angles = md.compute_phi(traj)
    elif kind == "psi":
        dih_idxs, angles = md.compute_psi(traj)
    elif kind == "chi1":
        dih_idxs, angles = md.compute_chi1(traj)
    elif kind == "chi2":
        dih_idxs, angles = md.compute_chi2(traj)
    else:
        raise KeyError(f'kind="{kind}" not allowed. Supported values: "phi", "psi", "chi1", "chi2"')

    names = []
    descriptors_ids = {}
    for i, idx in enumerate(dih_idxs):
        # find residue id from topology table
        # res = table['resSeq'][idx[0]]
        # name = 'dih_'+kind+'-'+str(res)
        res = table["resName"][idx[0]] + table["resSeq"][idx[0]].astype("str")
        name = "BACKBONE " + kind + " " + res
        if "chi" in kind:
            name = "SIDECHAIN " + kind + " " + res
        names.append(name)
        descriptors_ids[name] = list(idx)
        if sincos:
            for trig_transform in (np.sin, np.cos):
                # names.append('cos_(sin_)'+kind+'-'+str(res))
                name = "BACKBONE " + trig_transform.__name__ + "_" + kind + " " + res
                if "chi" in kind:
                    name = "SIDECHAIN " + "cos_" + kind + " " + res
                names.append(name)
                descriptors_ids[name] = list(idx) 
    if sincos:
        angles = np.hstack([angles, np.sin(angles), np.cos(angles)])
    return angles, names, descriptors_ids

def load_descriptors(descriptors, start = 0, stop = None, stride = 1, **kwargs):
    descriptors = load_dataframe(descriptors, **kwargs)
    ##?? Luigi why no stop parameter?
    descriptors = descriptors.iloc[start::stride, :]
    if "time" in descriptors.columns:
        descriptors = descriptors.drop("time", axis="columns")
    if __DEV__:
        print(f"Descriptors: {descriptors.shape}")
    return descriptors

def sample(descriptors, states_labels, n_configs, regex_filter = '.*', states_subset=None, states_names=None):
    """Sample points from trajectory

    Args:
        n_configs (int): number of points to sample for each metastable state
        regex_filter (str, optional): regex to filter the features. Defaults to '.*'.
        states_subset (list, optional): list of integers corresponding to the metastable states to sample. Defaults to None take all states.
        states_names (list, optional): list of strings corresponding to the name of the states. Defaults to None.

    Returns:
        (configurations, labels), features_names, states_names
    """
    assert len(descriptors) == len(states_labels), "Length mismatch between descriptors and states_labels."
    features = descriptors.filter(regex=regex_filter).columns.values
    config_list = []
    labels = []
    
    if isinstance(states_labels, pd.DataFrame):
        pass 
    elif isinstance(states_labels, np.ndarray):
        states_labels = np.squeeze(states_labels)
        columns = ['labels']
        if states_labels.ndim == 2:
            columns.append('selection')
        states_labels = pd.DataFrame(data=states_labels, columns=columns)
    else:
        raise TypeError(
            f"{states_labels}: Accepted types are 'pandas.Dataframe' or 'numpy.ndarray' "
        )
    if not ('selection' in states_labels):
        states_labels['selection'] = np.ones(len(states_labels), dtype=bool)

    states = dict()
    if states_subset is None:
        states_subset = range(len(states_labels['labels'].unique()))

    for i in states_subset:
        if states_names is None:
            states[i] = i
        else:
            states[i] = states_names[i]

    for label in states_subset:
        #select label
        df = descriptors.loc[ (states_labels['labels'] == label) & (states_labels['selection'] == True)]
        #select descriptors and sample
        replace = False
        if n_configs > len(df):
            warn("The asked number of samples is higher than the possible unique values. Sampling with replacement")
            replace = True
        config_i = df.filter(regex=regex_filter).sample(n=n_configs, replace=replace).values   
        config_list.append(config_i)
        labels.extend([label]*n_configs)
    labels = np.array(labels, dtype=int)
    configurations = np.vstack(config_list)
    return (configurations, labels), features, states