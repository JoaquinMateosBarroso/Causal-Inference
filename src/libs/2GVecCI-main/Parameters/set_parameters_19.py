


def set_par():
    parameter_choices = {}
    parameter_choices['samplesize'] = [200]
    parameter_choices['cause_region_size'] = [25]
    parameter_choices['effect_region_size'] = [25]
    parameter_choices['cause_region_density'] = [0.1]
    parameter_choices['effect_region_density'] = [0.1]
    parameter_choices['interaction_density'] = [0.1, 0.3, 0.5, 0.7, 0.9]
    parameter_choices['interaction_strength'] = [(-0.7, 0.7)]
    parameter_choices['test_type'] = ['both']
    parameter_choices['significance'] = [0.01]
    parameter_choices['number_of_random_models_generated'] = [50]
    return parameter_choices