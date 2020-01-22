import numpy as np
import pandas as pd
import re

def find_numbers(s, dtype=int):
    return dtype(re.findall(r'[0-9]+', s)[0])

np_find_numbers = np.vectorize(find_numbers)

def load_feature_and_outcome(
    feature_path='./data/sc1_Phase1_GE_FeatureMatrix.tsv',
    phenotype_path='./data/sc1_Phase1_GE_Phenotype.tsv',
    outcome_path='./data/sc1_Phase1_GE_Outcome.tsv',
    shuffle=True,
    patientid_as_feature=False
    ):
    '''
    load feature and outcome data (with phenotype)
    '''

    feature_matrix_frame = pd.read_csv(feature_path, delimiter='\t')
    outcome_frame = pd.read_csv(outcome_path, delimiter='\t')
    phenotype_frame = pd.read_csv(phenotype_path, delimiter='\t')

    maptables = {}
    for phenotype_name in phenotype_frame.columns:
        if phenotype_name != 'PATIENTID':
            phenotype_frame[phenotype_name], maptable = pd.factorize(phenotype_frame[phenotype_name], sort=True)
            maptables[phenotype_name] = maptable


    combined_frame = pd.merge(phenotype_frame, feature_matrix_frame, on='PATIENTID')
    combined_frame = pd.merge(combined_frame, outcome_frame, on='PATIENTID')

    if patientid_as_feature:
        vals = combined_frame['PATIENTID'].values
        combined_frame['PATIENTID'] = np_find_numbers(vals)
    else:
        del combined_frame['PATIENTID']

    data = combined_frame.values

    if shuffle:
        shuffle_ind = np.arange(data.shape[0])
        np.random.shuffle(shuffle_ind)
        data = data[shuffle_ind]

    label, feature = data[:,-1], data[:,:-1]
    gene_names = np.array(list(combined_frame.columns.values[:-1]))

    return gene_names, feature, label, maptables

def load_reference_genes(filename):
    with open(filename) as feature_file:
        ref_genes = feature_file.readlines()

    ref_genes = [item.split()[0] for item in ref_genes]

    # remove duplicates
    ref_genes = sorted(set(ref_genes))

    return ref_genes