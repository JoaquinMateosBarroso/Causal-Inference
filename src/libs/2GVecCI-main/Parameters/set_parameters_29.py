


def set_par():
    parameter_choices = {}
    parameter_choices['samplesize'] = list(range(60,160,10))
    parameter_choices['cause_region_size'] = [30]
    parameter_choices['effect_region_size'] = [30]
    parameter_choices['cause_region_density'] = [0.3]
    parameter_choices['effect_region_density'] = [0.3]
    parameter_choices['interaction_density'] = [0.3]
    parameter_choices['interaction_strength'] = [(-0.7, 0.7)]
    parameter_choices['test_type'] = ['both']
    parameter_choices['significance'] = [0.01]
    parameter_choices['number_of_random_models_generated'] = [100]
    return parameter_choices