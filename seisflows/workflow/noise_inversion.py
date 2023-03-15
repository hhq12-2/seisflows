#!/usr/bin/env python3
"""
Ambient Noise Adjoint Tomography Forward Solver based on the workflow proposed
by Wang et al. where synthetic Greens functinos (SGF) are generated by
simulating point forces.

Reference
- "Three‐dimensional sensitivity kernels for multicomponent empirical Green's
  functions from ambient noise: Methodology and application to Adjoint
  tomography."
  Journal of Geophysical Research: Solid Earth 124.6 (2019): 5794-5810.

"""
import os
from seisflows import logger
from seisflows.workflow.inversion import Inversion


class NoiseInversion(Inversion):
    """
    Noise Inversion Workflow
    ------------------------


    Parameters
    ----------
    :type modules: list of module
    :param modules: instantiated SeisFlows modules which should have been
        generated by the function `seisflows.config.import_seisflows` with a
        parameter file generated by seisflows.configure
    :type data_case: str
    :param data_case: How to address 'data' in the workflow, available options:
        'data': real data will be provided by the user in
        `path_data/{source_name}` in the same format that the solver will
        produce synthetics (controlled by `solver.format`) OR
        synthetic': 'data' will be generated as synthetic seismograms using
        a target model provided in `path_model_true`. If None, workflow will
        not attempt to generate data.
    :type stop_after: str
    :param stop_after: optional name of task in task list (use
        `seisflows print tasks` to get task list for given workflow) to stop
        workflow after, allowing user to prematurely stop a workflow to explore
        intermediate results or debug.
    :type export_traces: bool
    :param export_traces: export all waveforms that are generated by the
        external solver to `path_output`. If False, solver traces stored in
        scratch may be discarded at any time in the workflow
    :type export_residuals: bool
    :param export_residuals: export all residuals (data-synthetic misfit) that
        are generated by the external solver to `path_output`. If False,
        residuals stored in scratch may be discarded at any time in the 
        workflow

    Paths
    -----
    :type workdir: str
    :param workdir: working directory in which to perform a SeisFlows workflow.
        SeisFlows internal directory structure will be created here. Default cwd
    :type path_output: str
    :param path_output: path to directory used for permanent storage on disk.
        Results and exported scratch files are saved here.
    :type path_data: str
    :param path_data: path to any externally stored data required by the solver
    :type path_state_file: str
    :param path_state_file: path to a text file used to track the current
        status of a workflow (i.e., what functions have already been completed),
        used for checkpointing and resuming workflows
    :type path_model_init: str
    :param path_model_init: path to the starting model used to calculate the
        initial misfit. Must match the expected `solver_io` format.
    :type path_model_true: str
    :param path_model_true: path to a target model if `case`=='synthetic' and
        a set of synthetic 'observations' are required for workflow.
    :type path_eval_grad: str
    :param path_eval_grad: scratch path to store files for gradient evaluation,
        including models, kernels, gradient and residuals.
    ***
    """
    def __init__(self, kernels=None, **kwargs):
        """
        Set default forward workflow parameters

        :type modules: list
        :param modules: list of sub-modules that will be established as class
            attributes by the setup() function. Should not need to be set by the
            user
        """
        super().__init__(**kwargs)

        if kernels is None:
            self.kernels = ["ZZ", "TT"]
        else:
            self.kernels = kernels

        # Hard code some Forward workflow parameters to ensure that only certain
        # workflow pathways area followed for noise inversions
        self.data_case = "data"
        self.path.model_true = None

    @property
    def task_list(self):
        """
        USER-DEFINED TASK LIST. This property defines a list of class methods
        that take NO INPUT and have NO RETURN STATEMENTS. This defines your
        linear workflow, i.e., these tasks are to be run in order from start to
        finish to complete a workflow.

        This excludes 'check' (which is run during 'import_seisflows') and
        'setup' which should be run separately

        .. note::
            For workflows that require an iterative approach (e.g. inversion),
            this task list will be looped over, so ensure that any setup and
            teardown tasks (run once per workflow, not once per iteration) are
            not included.

        :rtype: list
        :return: list of methods to call in order during a workflow
        """
        return [self.evaluate_initial_misfit,
                self.run_adjoint_simulations,
                self.postprocess_event_kernels,
                self.evaluate_gradient_from_kernels,
                self.initialize_line_search,
                self.perform_line_search,
                self.finalize_iteration
                ]

    def check(self):
        """
        Check that noise workflow has correct parameters set
        """
        super().check()

        # assert(self.solver.source_prefix == "FORCESOLUTION"), \
        #     "noise simulations requires `source_prefix`=='FORCESOLUTION'"

    def run_forward_simulations(self, path_model, **kwargs):
        """
        Performs forward simulation for a single given master station
        """
        # 