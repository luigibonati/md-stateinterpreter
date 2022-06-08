# imports
import numpy as np
import pandas as pd
import mdtraj as md
import itertools
from warnings import warn

from .utils.io import load_dataframe
from ._configs import *
from .utils._compiled_numerics import contact_function

__all__ = ["compute_descriptors", "load_descriptors"]

def compute_descriptors(traj, descriptors = ['ca', 'dihedrals', 'hbonds_distances', 'hbonds_contacts']):

    """Compute descriptors from trajectory:
    - Dihedral angles
    - CA distances
    - Hydrogen bonds distances
    - Hydrogen bonds contacts
    - Disulfide bonds dihedrals

    Parameters
    ----------
    descriptors : bool or list
        compute list of descriptors. by default compute all the following descriptors: ['ca', 'dihedrals', 'hbonds_distances', 'hbonds_contacts'].

    Raises
    ------
    KeyError
        Trajectory needs to be set beforehand.
    """

    descr_list = []
    #Cache dists for performance
    _cached_dists = False
    if ('hbonds_distances' in descriptors) and ('hbonds_contacts' in descriptors):
        _cached_dists = True
        dist_idx = descriptors.index('hbonds_distances')
        cont_idx = descriptors.index('hbonds_contacts')
        #swap elements so that hb_distances are calculated first and then cached
        if cont_idx < dist_idx:
            get = descriptors[cont_idx], descriptors[dist_idx]        
            descriptors[dist_idx], descriptors[cont_idx] = get
    _raw_data = []
    _feats = []
    _feats_info = {}
    for d in descriptors:
        if d == 'ca':
            res, names, descriptors_ids = _CA_DISTANCES(traj)
        elif d == 'dihedrals':
            for angle in ['phi', 'psi', 'chi1', 'chi2']:
                res, names, descriptors_ids = _DIHEDRALS(traj, kind=angle, sincos=True)
                _raw_data.append(res)
                _feats.extend(names)
                _feats_info.update(descriptors_ids)
        elif d == 'hbonds_distances':
            res, names, descriptors_ids = _HYDROGEN_BONDS(traj, 'distances')
            if _cached_dists:
                _dsts = np.copy(res)
        elif d == 'hbonds_contacts':
            if _cached_dists:
                res, names, descriptors_ids = _HYDROGEN_BONDS(traj, 'contacts', _cached_dists=_dsts)
            else:
                res, names, descriptors_ids = _HYDROGEN_BONDS(traj, 'contacts')
        elif d == 'disulfide':
            res, names, descriptors_ids = _DISULFIDE_DIHEDRALS(traj, sincos=True)
            print(names)
            print(descriptors_ids)
        else:
            raise KeyError(f"descriptor: {d} not valid. Only 'ca', 'dihedrals', 'hbonds_distances', 'hbonds_contacts','disulfide' are allowed.")
        
        if d != 'dihedrals': #(Done previously)
            _raw_data.append(res)
            _feats.extend(names)
            _feats_info.update(descriptors_ids)
    
    df = pd.DataFrame(np.hstack(_raw_data), columns=_feats)    
    if __DEV__:
        print(f"Descriptors: {df.shape}")
    return df, _feats_info  

def load_descriptors(descriptors, start = 0, stop = None, stride = 1, **kwargs):
    descriptors = load_dataframe(descriptors, **kwargs)
    descriptors = descriptors.iloc[start:stop:stride, :]
    if "time" in descriptors.columns:
        descriptors = descriptors.drop("time", axis="columns")
    if __DEV__:
        print(f"Descriptors: {descriptors.shape}")
    return descriptors

# DESCRIPTORS COMPUTATION
def _CA_DISTANCES(traj):
    descriptors_ids = {}
    if __DEV__:
        print(f"Computing CA distances")
    table, _ = traj.top.to_dataframe()
    sel = traj.top.select("name CA")

    pairs = [(i, j) for i, j in itertools.combinations(sel, 2)]
    dist = md.compute_distances(traj, np.array(pairs, dtype=int))

    # Labels
    label = lambda i, j: "DIST %s%s -- %s%s" % (
        traj.top.atom(i),
        "s" if traj.top.atom(i).is_sidechain else "",
        traj.top.atom(j),
        "s" if traj.top.atom(j).is_sidechain else "",
    )

    names = [label(i, j) for (i, j) in pairs]
    for (i,j) in pairs:
        res_i = table["resName"][i] + table["resSeq"][i].astype("str")
        res_j = table["resName"][j] + table["resSeq"][j].astype("str")
        info = {
            'atoms': [i,j],
            'group': res_i + "_" + res_j
        }
        descriptors_ids[label(i, j)] = info
    return dist, names, descriptors_ids

def _HYDROGEN_BONDS(traj, kind, _cached_dists = None):
    # H-BONDS DISTANCES / CONTACTS (donor-acceptor)
    # find donors (OH or NH)
    if __DEV__:
        print(f"Computing Hydrogen bonds {kind}")

    table, _ = traj.top.to_dataframe()
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
    #hbonded = [ at_i.index
    #            for at_i, at_j in traj.top.bonds
    #            if (at_j.element.symbol == "H") 
    #]
    #acceptors = [idx 
    #            for idx in traj.top.select("symbol O or symbol N")
    #            if idx not in hbonded
    #            ]
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
    if _cached_dists is None:
        dist = md.compute_distances(traj, pairs)
    else:
        dist = _cached_dists

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
            res_x = table["resName"][x] + table["resSeq"][x].astype("str")
            res_y = table["resName"][y] + table["resSeq"][y].astype("str")
            info = {
                'atoms': [x,y],
                'group': res_x + "_" + res_y
            }
            descriptors_ids[label(x,y)] = info   
        return dist, names, descriptors_ids
    elif kind == 'contacts':
        descriptors_ids = {}
        # Compute contacts
        contacts = contact_function(dist, r0=0.35, d0=0, n=6, m=12)
        # labels
        # basename = 'hbc_'
        # names = [ basename+str(x)+'-'+str(y) for x,y in pairs]
        label = lambda i, j: "HB_C %s%s -- %s%s" % (
            traj.top.atom(i),
            "s" if traj.top.atom(i).is_sidechain else "",
            traj.top.atom(j),
            "s" if traj.top.atom(j).is_sidechain else "",
        )
        names = [label(x, y) for x, y in pairs]
        for (x,y) in pairs:
            res_x = table["resName"][x] + table["resSeq"][x].astype("str")
            res_y = table["resName"][y] + table["resSeq"][y].astype("str")
            info = {
                'atoms': [x,y],
                'group': res_x + "_" + res_y
            }
            descriptors_ids[label(x,y)] = info  

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
    if sincos:
        sin_names = []
        cos_names = []
    descriptors_ids = {}
    for i, idx in enumerate(dih_idxs):
        # find residue id from topology table
        # res = table['resSeq'][idx[0]]
        # name = 'dih_'+kind+'-'+str(res)
        res = table["resName"][idx[0]] + table["resSeq"][idx[0]].astype("str")
        #name = "BACKBONE " + kind + " " + res
        name = kind + " " + res
        #if "chi" in kind:
        #    name = "SIDECHAIN " + kind + " " + res
        names.append(name)
        info = {
                'atoms': list(idx),
                'group': res
            }
        descriptors_ids[name] = info  
        if sincos:
            for trig_transform in (np.sin, np.cos):
                _trans_name = trig_transform.__name__ + "_"
                # names.append('cos_(sin_)'+kind+'-'+str(res))
                #name = "BACKBONE " + _trans_name + kind + " " + res
                name = _trans_name + kind + " " + res
                #if "chi" in kind:
                #    name = "SIDECHAIN " + _trans_name + kind + " " + res
                #Dirty trick
                eval(_trans_name + "names.append(name)")
                descriptors_ids[name] = info
    if sincos:
        angles = np.hstack([angles, np.sin(angles), np.cos(angles)])
        names = names + sin_names + cos_names
    
    return angles, names, descriptors_ids

def _DISULFIDE_DIHEDRALS(traj, sincos = True):

    table, bonds = traj.top.to_dataframe()

    # filter S atoms belonging to CYS
    s_cys = table[ (table['element'] == 'S') & (table['resName'] == 'CYS') ].index

    # define arrays
    names = []
    angles = []
    descriptors_ids = {}

    # Loop over every pair of S atoms
    for i,j in itertools.combinations(s_cys,2):
        # Check if bond is formed
        d_ij = md.compute_distances(traj[0],[[i,j]])[0][0]
        if d_ij < 0.25:
            # look for C atoms bonded with S
            for k,l,_,_ in bonds:
                if int(k) == i:
                    c_i = int(l)
                elif int(l) == i:
                    c_i = int(k)
                if int(k) == j:
                    c_j = int(l)
                elif int(l) == j:
                    c_j = int(k)

            # compute feature
            group = str(traj.top.atom(i).residue)+'_'+str(traj.top.atom(j).residue)
            
            desc = md.compute_dihedrals(traj,[[c_i,i,j,c_j]])[:,0]

            name = 'DISULFIDE dih '+group 
            descriptors_ids[name] = {'atoms': [c_i,i,j,c_j], 'group' : group}
            angles.append(desc)
            names.append(name)

            if sincos:
                name = 'DISULFIDE sin_dih '+group 
                descriptors_ids[name] = {'atoms': [c_i,i,j,c_j], 'group' : group}
                angles.append(np.sin(desc))
                names.append(name)

                name = 'DISULFIDE cos_dih '+group 
                descriptors_ids[name] = {'atoms': [c_i,i,j,c_j], 'group' : group}
                angles.append(np.cos(desc))
                names.append(name)

    angles = np.asarray(angles).T

    return angles, names, descriptors_ids

