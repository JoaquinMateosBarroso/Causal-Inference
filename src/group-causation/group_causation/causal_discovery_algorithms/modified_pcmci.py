from copy import deepcopy
import time
# from bayes_opt import BayesianOptimization
import numpy as np

from tigramite.pcmci import PCMCI, _nested_to_normal, _create_nested_dictionary

from group_causation.causal_discovery_algorithms.causal_discovery_custom import summarized_causality_multivariate_granger, summarized_causality_univariate_granger


class PCMCI_Modified(PCMCI):
    def run_pcmciplus(self,
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
                      **kwargs,
                      ):

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
                        fdr_method=fdr_method,
                        **kwargs)

        elif pc_alpha < 0. or pc_alpha > 1:
            raise ValueError("Choose 0 <= pc_alpha <= 1")

        # Check the limits on tau
        self._check_tau_limits(tau_min, tau_max)
        # Set the link assumption
        _int_link_assumptions = self._set_link_assumptions(link_assumptions, tau_min, tau_max)


        #
        # Phase 0: Get as selected_links those that have a granger-lagged connection
        #
        if link_assumptions is None:
            link_assumptions = self._remove_nongranger_summarized_links(tau_max, **kwargs)
        
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
                  "\n\nArgs:")
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
        """Lagged PC algorithm for estimating lagged parents of single variable.

        Parameters
        j : int
            Variable index.
        link_assumptions_j : dict
            Dictionary of form {j:{(i, -tau): link_type, ...}, ...} specifying
            assumptions about links. This initializes the graph with entries
            graph[i,j,tau] = link_type. For example, graph[i,j,0] = '-->'
            implies that a directed link from i to j at lag 0 must exist.
            Valid link types are 'o-o', '-->', '<--'. In addition, the middle
            mark can be '?' instead of '-'. Then '-?>' implies that this link
            may not exist, but if it exists, its orientation is '-->'. Link
            assumptions need to be consistent, i.e., graph[i,j,0] = '-->'
            requires graph[j,i,0] = '<--' and acyclicity must hold. If a link
            does not appear in the dictionary, it is assumed absent. That is,
            if link_assumptions is not None, then all links have to be specified
            or the links are assumed absent.
        tau_min : int, optional (default: 1)
            Minimum time lag to test. Useful for variable selection in
            multi-step ahead predictions. Must be greater zero.
        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.
        save_iterations : bool, optional (default: False)
            Whether to save iteration step results such as conditions used.
        pc_alpha : float or None, optional (default: 0.2)
            Significance level in algorithm. If a list is given, pc_alpha is
            optimized using model selection criteria provided in the
            cond_ind_test class as get_model_selection_criterion(). If None,
            a default list of values is used.
        max_conds_dim : int, optional (default: None)
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.
        max_combinations : int, optional (default: 1)
            Maximum number of combinations of conditions of current cardinality
            to test in PC1 step.

        Returns
        parents : list
            List of estimated parents.
        val_min : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum absolute
            test statistic value of a link.
        pval_max : dict
            Dictionary of form {(0, -1):float, ...} containing the maximum
            p-value of a link across different conditions.
        iterations : dict
            Dictionary containing further information on algorithm steps.
        """

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
            tic = time.time()
            
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
                for comb_index, Z in \
                        enumerate(self._iter_conditions(parent, conds_dim,
                                                        parents)):
                    # Break if we try too many combinations
                    if comb_index >= max_combinations:
                        break
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
                    # Print some information if needed
                    if self.verbosity > 1:
                        self._print_cond_info(Z, comb_index, pval, val)
                    # Keep track of maximum p-value and minimum estimated value
                    # for each pair (across any condition)
                    val_min[parent] = \
                        min(np.abs(val), val_min.get(parent,
                                                            float("inf")))

                    if pval_max[parent] is None or pval > pval_max[parent]:
                        pval_max[parent] = pval
                        val_dict[parent] = val

                    # Save the iteration if we need to
                    if save_iterations:
                        a_iter = iterations['iterations'][conds_dim][parent]
                        a_iter[comb_index]['conds'] = deepcopy(Z)
                        a_iter[comb_index]['val'] = val
                        a_iter[comb_index]['pval'] = pval
                    # Delete link later and break while-loop if non-significant
                    if not dependent: #pval > pc_alpha:
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
            
            toc = time.time()
            
            # print(f'X_{j}; Iteration {conds_dim} took {toc - tic} seconds')

        # Print information about if convergence was reached
        if self.verbosity > 1:
            self._print_converged_pc_single(converged, j, max_conds_dim)
        # Return the results
        return {'parents': parents,
                'val_min': val_min,
                'val_dict': val_dict,
                'pval_max': pval_max,
                'iterations': _nested_to_normal(iterations)}

    def _remove_dependent_links(self, tau_max):
        '''
        TODO
        Set the link assumptions from an independence test.
        '''
        link_assumptions = dict()
        
        

    def _remove_nongranger_summarized_links(self,
                                            tau_max,
                                            max_summarized_crosslinks_density,
                                            preselection_alpha):
        """
        Set the link assumptions from the Granger causality test.
        """
        link_assumptions = dict()
        data_matrix = self.dataframe.values[0]
        # Obtain the Granger causality graph
        granger_graph = summarized_causality_multivariate_granger(self.dataframe.values[0], tau_max,
                                                       max_summarized_crosslinks_density, preselection_alpha)
        
        if self.verbosity > 0:
            print(f'{granger_graph.edges()=}')
        
        for j in range(data_matrix.shape[1]):
            # Assume autocausal links always exist
            link_assumptions[j] = {(j, -1): '-->'}
            for i in range(data_matrix.shape[1]):
                if granger_graph.has_edge(i, j):
                    for tau in range(1, tau_max):
                        # Test other cross-links
                        link_assumptions[j][(i, -tau)] = '-->'
        
        return link_assumptions

    
    
    def _optimize_pcmciplus_alpha(self,
                      link_assumptions,
                      tau_min,
                      tau_max,
                      pc_alpha,
                      contemp_collider_rule,
                      conflict_resolution,
                      reset_lagged_links,
                      max_conds_dim,
                      max_combinations,
                      max_conds_py,
                      max_conds_px,
                      max_conds_px_lagged,
                      fdr_method,
                      **kwargs,
                      ):
        """Optimizes pc_alpha in PCMCIplus.

        If a list or None is passed for ``pc_alpha``, the significance level is
        optimized for every graph across the given ``pc_alpha`` values using the
        score computed in ``cond_ind_test.get_model_selection_criterion()``

        Parameters
        See those for run_pcmciplus()

        Returns
        Results for run_pcmciplus() for the optimal pc_alpha.
        """

        if pc_alpha is None:
            pc_alpha_list = [0.001, 0.005, 0.01, 0.025, 0.05]
        else:
            pc_alpha_list = pc_alpha

        if self.verbosity > 0:
            print("\n##\n## Optimizing pc_alpha over " + 
                  "pc_alpha_list = %s" % str(pc_alpha_list) +
                  "\n##")

        results = {}
        score = np.zeros_like(pc_alpha_list)
        for iscore, pc_alpha_here in enumerate(pc_alpha_list):
            # Print statement about the pc_alpha being tested
            if self.verbosity > 0:
                print("\n## pc_alpha = %s (%d/%d):" % (pc_alpha_here,
                                                      iscore + 1,
                                                      score.shape[0]))
            # Get the results for this alpha value
            results[pc_alpha_here] = \
                self.run_pcmciplus(link_assumptions=link_assumptions,
                                    tau_min=tau_min,
                                    tau_max=tau_max,
                                    pc_alpha=pc_alpha_here,
                                    contemp_collider_rule=contemp_collider_rule,
                                    conflict_resolution=conflict_resolution,
                                    reset_lagged_links=reset_lagged_links,
                                    max_conds_dim=max_conds_dim,
                                    max_combinations=max_combinations,
                                    max_conds_py=max_conds_py,
                                    max_conds_px=max_conds_px,
                                    max_conds_px_lagged=max_conds_px_lagged,
                                    fdr_method=fdr_method,
                                    **kwargs)

            # Get one member of the Markov equivalence class of the result
            # of PCMCIplus, which is a CPDAG

            # First create order that is based on some feature of the variables
            # to avoid order-dependence of DAG, i.e., it should not matter
            # in which order the variables appear in dataframe
            # Here we use the sum of absolute val_matrix values incident at j
            val_matrix = results[pc_alpha_here]['val_matrix']
            variable_order = np.argsort(
                                np.abs(val_matrix).sum(axis=(0,2)))[::-1]

            dag = self._get_dag_from_cpdag(
                            cpdag_graph=results[pc_alpha_here]['graph'],
                            variable_order=variable_order)
            

            # Compute the best average score when the model selection
            # is applied to all N variables
            for j in range(self.N):
                parents = []
                for i, tau in zip(*np.where(dag[:,j,:] == "-->")):
                    parents.append((i, -tau))
                score_j = self.cond_ind_test.get_model_selection_criterion(
                        j, parents, tau_max)
                score[iscore] += score_j
            score[iscore] /= float(self.N)

        # Record the optimal alpha value
        optimal_alpha = pc_alpha_list[score.argmin()]

        if self.verbosity > 0:
            print("\n##"+
                  "\n\n## Scores for individual pc_alpha values:\n")
            for iscore, pc_alpha in enumerate(pc_alpha_list):
                print("   pc_alpha = %7s yields score = %.5f" % (pc_alpha, 
                                                                score[iscore]))
            print("\n##\n## Results for optimal " +
                  "pc_alpha = %s\n##" % optimal_alpha)
            self.print_results(results[optimal_alpha], alpha_level=optimal_alpha)

        optimal_results = results[optimal_alpha]
        optimal_results['optimal_alpha'] = optimal_alpha
        return optimal_results

