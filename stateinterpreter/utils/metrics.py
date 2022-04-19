import numpy as np

def get_best_reg(classifier):
    # accuracy
    acc = classifier.get_accuracy()
    # number of features
    num = classifier.get_num_features()
    
    # criterion
    score = (1-acc)*100+num

    # return reg which minimizes it
    best_reg = np.argmin(score)
    return classifier._reg[best_reg],acc[best_reg],num[best_reg]

def get_basis_quality(classifier):
    # accuracy
    acc = classifier.get_accuracy()
    # minimum number of features which give 99% accuracy
    num = classifier.get_num_features()
    #idx = np.where( 1-acc < __EPS__ )
    idx = np.where( 1-acc < 0.01 )

    if len(idx[0]) > 0: 
        min_num = np.min( num[ idx ] )
    else:
        min_num = num[0]

    return min_num