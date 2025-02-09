import time
import numpy as np

from tigramite.pcmci import PCMCI
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from create_toy_datasets import generate_toy_data


class PCMCI_Modified(PCMCI):
    def run_pcmciplus_getting_times(self,
                      selected_links=None,
                      link_assumptions=None,
                      tau_min=0,
                      tau_max=1,
                      pc_alpha=0.01,
                      contemp_collider_rule='majority',
                      conflict_resolution=True,
                      reset_lagged_links=False,
                      max_conds_dim=None,
                      max_combinations=1,
                      max_conds_py=None,
                      max_conds_px=None,
                      max_conds_px_lagged=None,
                      fdr_method='none',
                      ):
        """Runs PCMCIplus time-lagged and contemporaneous causal discovery for
        time series.
        This function extracts the times taken at each step.
        Complete documentation in parent function.
        """

        if selected_links is not None:
            raise ValueError("selected_links is DEPRECATED, use link_assumptions instead.")

        # Check if pc_alpha is chosen to optimze over a list
        if pc_alpha is None or isinstance(pc_alpha, (list, tuple, np.ndarray)):
            # Call optimizer wrapper around run_pcmciplus()
            return self._optimize_pcmciplus_alpha(
                        link_assumptions=link_assumptions,
                        tau_min=tau_min,
                        tau_max=tau_max,
                        pc_alpha=pc_alpha,
                        contemp_collider_rule=contemp_collider_rule,
                        conflict_resolution=conflict_resolution,
                        reset_lagged_links=reset_lagged_links,
                        max_conds_dim=max_conds_dim,
                        max_combinations=max_combinations,
                        max_conds_py=max_conds_py,
                        max_conds_px=max_conds_px,
                        max_conds_px_lagged=max_conds_px_lagged,
                        fdr_method=fdr_method)

        elif pc_alpha < 0. or pc_alpha > 1:
            raise ValueError("Choose 0 <= pc_alpha <= 1")

        # Check the limits on tau
        self._check_tau_limits(tau_min, tau_max)
        # Set the link assumption
        _int_link_assumptions = self._set_link_assumptions(link_assumptions, tau_min, tau_max)

        times = dict()
        
        tic = time.time()
        #
        # Phase 1: Get a superset of lagged parents from run_pc_stable
        #
        lagged_parents = self.run_pc_stable(link_assumptions=link_assumptions,
                            tau_min=tau_min,
                            tau_max=tau_max,
                            pc_alpha=pc_alpha,
                            max_conds_dim=max_conds_dim,
                            max_combinations=max_combinations)
        # Extract p- and val-matrix
        p_matrix = self.p_matrix
        val_matrix = self.val_matrix

        toc = time.time()
        times['phase_1'] = toc - tic
        
        tic = time.time()
        
        #
        # Phase 2: PC algorithm with contemp. conditions and MCI tests
        #
        if self.verbosity > 0:
            print("\n##\n## Step 2: PC algorithm with contemp. conditions "
                  "and MCI tests\n##"
                  "\n\nParameters:")
            if link_assumptions is not None:
                print("\nlink_assumptions = %s" % str(_int_link_assumptions))
            print("\nindependence test = %s" % self.cond_ind_test.measure
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max
                  + "\npc_alpha = %s" % pc_alpha
                  + "\ncontemp_collider_rule = %s" % contemp_collider_rule
                  + "\nconflict_resolution = %s" % conflict_resolution
                  + "\nreset_lagged_links = %s" % reset_lagged_links
                  + "\nmax_conds_dim = %s" % max_conds_dim
                  + "\nmax_conds_py = %s" % max_conds_py
                  + "\nmax_conds_px = %s" % max_conds_px
                  + "\nmax_conds_px_lagged = %s" % max_conds_px_lagged
                  + "\nfdr_method = %s" % fdr_method
                  )

        skeleton_results = self._pcmciplus_mci_skeleton_phase(
                            lagged_parents=lagged_parents, 
                            link_assumptions=_int_link_assumptions, 
                            pc_alpha=pc_alpha,
                            tau_min=tau_min, 
                            tau_max=tau_max, 
                            max_conds_dim=max_conds_dim, 
                            max_combinations=None,    # Otherwise MCI step is not consistent
                            max_conds_py=max_conds_py,
                            max_conds_px=max_conds_px, 
                            max_conds_px_lagged=max_conds_px_lagged, 
                            reset_lagged_links=reset_lagged_links, 
                            fdr_method=fdr_method,
                            p_matrix=p_matrix, 
                            val_matrix=val_matrix,
                            )

        toc = time.time()
        times['phase_2'] = toc - tic
        
        tic = time.time()
        
        #
        # Phase 3: Collider orientations (with MCI tests for default majority collider rule)
        #
        colliders_step_results = self._pcmciplus_collider_phase(
                            skeleton_graph=skeleton_results['graph'], 
                            sepsets=skeleton_results['sepsets'], 
                            lagged_parents=lagged_parents, 
                            pc_alpha=pc_alpha, 
                            tau_min=tau_min, 
                            tau_max=tau_max, 
                            max_conds_py=max_conds_py, 
                            max_conds_px=max_conds_px, 
                            max_conds_px_lagged=max_conds_px_lagged,
                            conflict_resolution=conflict_resolution, 
                            contemp_collider_rule=contemp_collider_rule)
        
        toc = time.time()
        times['phase_3'] = toc - tic
        
        tic = time.time()
        
        #
        # Phase 4: Meek rule orientations
        #
        final_graph = self._pcmciplus_rule_orientation_phase(
                            collider_graph=colliders_step_results['graph'],
                            ambiguous_triples=colliders_step_results['ambiguous_triples'], 
                            conflict_resolution=conflict_resolution)

        toc = time.time()
        times['phase_4'] = toc - tic
        
        # Store the parents in the pcmci member
        self.all_lagged_parents = lagged_parents

        return_dict = {
            'graph': final_graph,
            'p_matrix': skeleton_results['p_matrix'],
            'val_matrix': skeleton_results['val_matrix'],
            'sepsets': colliders_step_results['sepsets'],
            'ambiguous_triples': colliders_step_results['ambiguous_triples'],
            }

        # No confidence interval estimation here
        return_dict['conf_matrix'] = None

        # Print the results
        if self.verbosity > 0:
            self.print_results(return_dict, alpha_level=pc_alpha)
        
        # Return the dictionary
        self.results = return_dict
        
        return return_dict, times

from functions_test_toy_data import get_f1

def join_times(times_list: list[dict[str, float]]):
    """
    Join the times from the list of dictionaries
    """
    times = dict()
    for key in times_list[0].keys():
        times[key] = sum([times_dict[key] for times_dict in times_list]) / len(times_list)
    return times


if __name__ == '__main__':
    N_vars = 20
    
    for max_lag in [3, 5, 10, 20, 30, 40]:
        # Repeat 5 times and get the average time and F1 score
        times_list = []
        f1_list = []
        for i in range(5):
            # Generate toy data
            total_posible_edges = N_vars**2 * (max_lag - 1)
            dataset, ground_truth_parents = generate_toy_data(name='1', T=500, N=N_vars,
                                                              dependency_funcs=[lambda x: np.exp(-x**2)], max_lag=max_lag,
                                                              L=int(total_posible_edges * 0.1))
            dataframe = DataFrame(dataset.values, var_names=dataset.columns)
            
            pcmci = PCMCI_Modified(dataframe=dataframe, cond_ind_test=ParCorr(significance='analytic'))
            
            # Get times
            results, times = pcmci.run_pcmciplus_getting_times(tau_max=max_lag)
            times_list.append(times)
            
            # Get F1 score
            predicted_parents = pcmci.return_parents_dict(graph=results['graph'], val_matrix=results['val_matrix'])
            f1_list.append( get_f1(ground_truth_parents, predicted_parents) )
        
        average_times = join_times(times_list)
        average_f1 = sum(f1_list) / len(f1_list)
        
        print('******************************')
        print(f'{max_lag} max_lag ')
        print(f'{average_times=}')
        print(f'F1 score: {average_f1}')

        
        
