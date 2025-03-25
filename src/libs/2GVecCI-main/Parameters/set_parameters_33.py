


def set_par():
    parameter_choices = {}
    parameter_choices['samplesize'] = [100]
    parameter_choices['cause_region_size'] = list(range(3,31))
    parameter_choices['effect_region_size'] = list(range(3,31))
    parameter_choices['cause_region_density'] = [0.1]
    parameter_choices['effect_region_density'] = [0.1]
    parameter_choices['interaction_density'] = [1.0]
    parameter_choices['interaction_strength'] = [(-0.7, 0.7)]
    parameter_choices['test_type'] = ['both']
    parameter_choices['significance'] = [0.01]
    parameter_choices['number_of_random_models_generated'] = [100]
    return parameter_choices