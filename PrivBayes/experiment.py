from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

from pathlib import Path
import pandas as pd
import sys

print(sys.argv)
DATASET_NAME, SNP_COUNT, EPSILON, GENERATION_COUNT, IDX = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])


input_data_path = f'../data/cleansed/{DATASET_NAME}/data_{SNP_COUNT}_{IDX}.csv'
input_data = pd.read_csv(input_data_path, index_col = None)


# location of two output files
mode = 'correlated_attribute_mode'
description_file = f'../PrivBayes/descriptions/{DATASET_NAME}_{EPSILON}_{SNP_COUNT}_{IDX}.json'
synthetic_data_file = f'../data/privbayes/{DATASET_NAME}_{EPSILON}_{SNP_COUNT}_{IDX}.csv'


# An attribute is categorical if its domain size is less than this threshold.
# Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
threshold_value = 4

# specify categorical attributes
print(input_data.columns.values)
categorical_attributes = {idx:True for idx in input_data.columns.values if idx != 'SNP_ID'}

# specify which attributes are candidate keys of input dataset.
candidate_keys = {'PATIENT_ID':True}

# A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not 
# change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
# Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.
epsilon = EPSILON

# The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
degree_of_bayesian_network = 2

# Number of tuples generated in synthetic dataset.
num_tuples_to_generate = GENERATION_COUNT 

describer = DataDescriber(category_threshold=threshold_value)
describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data_path, 
                                                        epsilon=epsilon, 
                                                        k=degree_of_bayesian_network,
                                                        attribute_to_is_categorical=categorical_attributes,
                                                        attribute_to_is_candidate_key=candidate_keys)
Path(Path('/'.join(description_file.split('/', -1)[:-1]))).mkdir(parents=True, exist_ok=True)
describer.save_dataset_description_to_file(description_file)

generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
out_path = synthetic_data_file
Path(Path('/'.join(out_path.split('/', -1)[:-1]))).mkdir(parents=True, exist_ok=True)
generator.save_synthetic_data(out_path)