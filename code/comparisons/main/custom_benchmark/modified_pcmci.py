from collections import defaultdict
from copy import deepcopy
import time
from bayes_opt import BayesianOptimization
import numpy as np

from tigramite.pcmci import PCMCI, _nested_to_normal, _create_nested_dictionary
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from statsmodels.tsa.stattools import grangercausalitytests


class PCMCI_Modified(PCMCI):
    
    def run_pcmciplus(self,
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
        """
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
        
        #
        # Phase 4: Meek rule orientations
        #
        final_graph = self._pcmciplus_rule_orientation_phase(
                            collider_graph=colliders_step_results['graph'],
                            ambiguous_triples=colliders_step_results['ambiguous_triples'], 
                            conflict_resolution=conflict_resolution)

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
        
        return return_dict

    def _run_pc_stable_single(self, j,
                              link_assumptions_j=None,
                              tau_min=1,
                              tau_max=1,
                              save_iterations=False,
                              pc_alpha=0.2,
                              max_conds_dim=None,
                              max_combinations=1):
        if pc_alpha < 0. or pc_alpha > 1.:
            raise ValueError("Choose 0 <= pc_alpha <= 1")

        # Initialize the dictionaries for the pval_max, val_dict, val_min
        # results
        pval_max = dict()
        val_dict = dict()
        val_min = dict()
        # Initialize the parents values from the selected links, copying to
        # ensure this initial argument is unchanged.
        parents = []
        for itau in link_assumptions_j:
            link_type = link_assumptions_j[itau]
            if itau != (j, 0) and link_type not in ['<--', '<?-']:
                parents.append(itau)

        val_dict = {(p[0], p[1]): None for p in parents}
        pval_max = {(p[0], p[1]): None for p in parents}

        # Define a nested defaultdict of depth 4 to save all information about
        # iterations
        iterations = _create_nested_dictionary(4)
        # Ensure tau_min is at least 1
        tau_min = max(1, tau_min)

        # Loop over all possible condition dimensions
        max_conds_dim = self._set_max_condition_dim(max_conds_dim,
                                                    tau_min, tau_max)
        # Iteration through increasing number of conditions, i.e. from 
        # [0, max_conds_dim] inclusive
        converged = False
        for conds_dim in range(max_conds_dim + 1):
            # (Re)initialize the list of non-significant links
            nonsig_parents = list()
            # Check if the algorithm has converged
            if len(parents) - 1 < conds_dim:
                converged = True
                break
            # Print information about
            if self.verbosity > 1:
                print("\nTesting condition sets of dimension %d:" % conds_dim)
                            
            
            # Iterate through all possible pairs (that have not converged yet)
            
            for index_parent, parent in enumerate(parents):
                # Print info about this link
                if self.verbosity > 1:
                    self._print_link_info(j, index_parent, parent, len(parents))
                # Iterate through all possible combinations
                nonsig = False
                
                
                
                Z = self._optimize_cond_set(parent, parents, j, conds_dim, pc_alpha)
                
                # Perform independence test
                if link_assumptions_j[parent] == '-->':
                    val = 1.
                    pval = 0.
                    dependent = True
                else:
                    val, pval, dependent = self.cond_ind_test.run_test(X=[parent],
                                                Y=[(j, 0)],
                                                Z=Z,
                                                tau_max=tau_max,
                                                alpha_or_thres=pc_alpha,
                                                )
                
                # Keep track of maximum p-value and minimum estimated value
                # for each pair (across any condition)
                val_min[parent] = \
                    min(np.abs(val), val_min.get(parent,
                                                        float("inf")))

                if pval_max[parent] is None or pval > pval_max[parent]:
                    pval_max[parent] = pval
                    val_dict[parent] = val

                # Delete link later and break while-loop if non-significant
                if not dependent: # pval > pc_alpha:
                    nonsig_parents.append((j, parent))
                    nonsig = True
                    break

                # Print the results if needed
                if self.verbosity > 1:
                    self._print_a_pc_result(nonsig,
                                            conds_dim, max_combinations)

            # Remove non-significant links
            for _, parent in nonsig_parents:
                del val_min[parent]
            # Return the parents list sorted by the test metric so that the
            # updated parents list is given to the next cond_dim loop
            parents = self._sort_parents(val_min)
            # Print information about the change in possible parents
            if self.verbosity > 1:
                print("\nUpdating parents:")
                self._print_parents_single(j, parents, val_min, pval_max)

        # Print information about if convergence was reached
        if self.verbosity > 1:
            self._print_converged_pc_single(converged, j, max_conds_dim)
        # Return the results
        return {'parents': parents,
                'val_min': val_min,
                'val_dict': val_dict,
                'pval_max': pval_max,
                'iterations': _nested_to_normal(iterations)}
     
    def _optimize_cond_set(self, parent, parents, j, cond_dim, pc_alpha, tau_max=1):
        # Define the objective function
        def objective_function(lag, i):
            # Discretize the lag and i
            lag, i = int(lag), int(i)
            Z = [(i, -lag)]
            pval, val, dependent = self.cond_ind_test.run_test(X=[parent],
                                                                Y=[(j, 0)],
                                                                Z=Z,
                                                                tau_max=tau_max,
                                                                alpha_or_thres=pc_alpha,
                                                                )
            return -val
        
        # Define parameter bounds
        pbounds = {
            'lag': (1, tau_max + 1),
            'i': (0, max([p[0] for p in parents]) + 1),
        }

        # Initialize the optimizer
        optimizer = BayesianOptimization(
            f=objective_function,  # We'll define the function during optimization
            pbounds=pbounds,
            verbose=0,
            random_state=1,
        )

        # Custom function to suggest a batch of points
        def suggest_batch(optimizer, batch_size):
            suggested_points = []
            for _ in range(batch_size):
                next_point = optimizer.suggest()
                suggested_points.append(next_point)
                # Temporarily add the suggested point with a dummy value to avoid repetition
                optimizer.register(params=next_point, target=-1)
            # Remove the dummy points after suggestion
            for point in suggested_points:
                optimizer._space._cache.pop(point)
            return suggested_points

        # Custom optimization loop with batch processing
        def batch_optimize(optimizer, init_points=5, n_iter=25, batch_size=3):
            # Initial random exploration
            optimizer.maximize(init_points=init_points, n_iter=0)
            # Optimization loop
            for _ in range(n_iter):
                # Suggest a batch of points
                batch = suggest_batch(optimizer, batch_size)
                # Evaluate the objective function for each point in the batch
                for point in batch:
                    target = objective_function(**point)
                    optimizer.register(params=point, target=target)
        
        # batch_optimize(optimizer, init_points=20, n_iter=20, batch_size=cond_dim)
        
        # Run the optimization
        optimizer.maximize(init_points=10, n_iter=5)

        best_cond_set = optimizer.max
        
        i = int(best_cond_set['params']['i'])
        lag = int(best_cond_set['params']['lag'])
        
        return [(i, -lag)]


    


from functions_test_data import get_f1

def join_times(times_list: list[dict[str, float]]):
    """
    Join the times from the list of dictionaries
    """
    times = dict()
    for key in times_list[0].keys():
        times[key] = sum([times_dict[key] for times_dict in times_list]) / len(times_list)
    return times


def test_on_toy_data(T, N_vars, max_lag):
    # Repeat 5 times and get the average time and F1 score
    times_list = []
    f1_list = []
    for i in range(iters_per_parameter):
        # Generate toy data
        total_posible_edges = N_vars**2 * (max_lag - 1)
        dataset, ground_truth_parents = generate_toy_data(name='1', T=T, N=N_vars,
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
    
    return average_times, average_f1
    
import matplotlib.pyplot as plt
if __name__ == '__main__':
    N_vars_list = [10, 25, 50, 100]
    max_lag_list = [3, 5, 10, 30]
    T = 100
    iters_per_parameter = 1
    
    results = dict() # keys are (N_vars, max_lag) and values are dicts
    for N_vars in N_vars_list:
        for max_lag in max_lag_list:
            average_times, average_f1 = test_on_toy_data(T, N_vars, max_lag)
            results[(N_vars, max_lag)] = {'average_times': average_times, 'average_f1': average_f1}
            
            print('******************************')
            print(f'{N_vars} N_vars ')
            print(f'{max_lag} max_lag ')
            print(f'{average_times=}')
            print(f'F1 score: {average_f1}')
    
    # Save the results
    with open('complete_search_results.txt', 'w') as f:
        f.write(results.__str__())
    
    # Show 3d plot of times
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (N_vars, max_lag), result in results.items():
        total_times = {key: sum(times.values()) for key, times in result['average_times'].items()}
        ax.scatter(N_vars, max_lag, total_times, c='r')
    ax.set_xlabel('N_vars')
    ax.set_ylabel('max_lag')
    ax.set_zlabel('Time')
    
    plt.savefig('times_plot.pdf')
    
    # Show 3d plot of F1 scores
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (N_vars, max_lag), result in results.items():
        ax.scatter(N_vars, max_lag, result['average_f1'], c='r')
    ax.set_xlabel('N_vars')
    ax.set_ylabel('max_lag')
    ax.set_zlabel('F1 score')

        
        
