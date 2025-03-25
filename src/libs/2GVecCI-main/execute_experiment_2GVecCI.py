import os
import itertools
import Functions_2GVecCI as fcs
import numpy as np
from random import shuffle

#choose parameters in separate file and import here
import Parameters.set_parameters_13 as set_parameters

parameter_choices = set_parameters.set_par()

seed = 0
##choose as XX0 where set_parameters_XX to reproduce results
rng = np.random.RandomState(seed=seed)

configurations = []
config_names = []
for config in itertools.product(*list(parameter_choices.values())):
    configurations.append(config)

shuffle(configurations)


samplesize = parameter_choices['samplesize']

save_file = '2G_VecCI_test_vanillaPC'


summary = fcs.summary(configurations, random_state=rng, test= 'full', ambiguity=0.01,
                      CI_test_method='ParCorr', max_sep_set=None, linear=True, noise_type='gaussian',
                      standardize=True)
summary_dict = summary[0]
#For NONLINEAR(quadratic) interactions in the data, use this code:
# summary = fcs.summary_quadratic(configurations, random_state = rng, test='full', ambiguity = 0.01, CI_test_method = 'GPDC', max_sep_set = None, linear = 'no')
# summary_dict = summary

#For TRACE METHOD use this code:
#summary_dict = fcs.summary_trace_method(configurations, random_state=rng, eps = 0.1)
#summary_dict = summary

#For vanilla PC and linear data use this code:
# summary_dict = fcs.summary_vanilla_PC_linearPC(configurations, random_state=rng)

summary_name = 'summary_' + save_file + '.txt'
with open(summary_name, 'w') as f:
    for key, value in summary_dict.items():
        f.write('%s:%s\n' % (key, value))

######################### plotting ########################


folder_name = 'plots_test/'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

parameter_choices = set_parameters.set_par()

summary_name = 'summary_test' + '.txt'


plot_parameters = []
plot_parameters.append('cause_region_size')
plot_parameters.append('effect_region_size')
plot_parameters.append('cause_region_density')
plot_parameters.append('effect_region_density')
plot_parameters.append('interaction_density')

save_path1 = folder_name + '/average_plots/'

if not os.path.exists(save_path1):
    os.makedirs(save_path1)

save_path2 = folder_name + '/all_plots/'

if not os.path.exists(save_path2):
    os.makedirs(save_path2)

for plot_parameter in plot_parameters:
    fcs.direct_plotting(summary_dict, plot_parameter, full_parameters=parameter_choices, my_path=save_path1, errors=True)
    fixed_aux_list = fcs.get_fixed_aux_parameters(plot_parameter,parameter_choices)
    for aux_config in fixed_aux_list:
        fcs.direct_plotting(summary_dict, plot_parameter, full_parameters=parameter_choices, aux_parameters=aux_config, my_path=save_path2, errors=True)



