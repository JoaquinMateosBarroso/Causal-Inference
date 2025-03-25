
import numpy as np

def set_par():
    parameter_choices = {}
    parameter_choices['samplesize'] = [1000]
    parameter_choices['cause_region_size'] = [5]
    parameter_choices['effect_region_size'] = [5]
    parameter_choices['cause_region_density'] = [0.7]
    parameter_choices['effect_region_density'] = [0.7]
    parameter_choices['interaction_density'] = [0.9]
    parameter_choices['interaction_strength'] = [(-0.7, 0.7)]
    parameter_choices['test_type'] = ['both']
    parameter_choices['significance'] = [0.01]
    parameter_choices['number_of_random_models_generated'] = [10]
    return parameter_choices