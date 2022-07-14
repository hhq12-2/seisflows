#!/usr/bin/env python3
"""
This Solver module is in charge of interacting with external numerical solvers
such as SPECFEM (2D/3D/3D_GLOBE). This SPECFEM base class provides general
functions that work with all versions of SPECFEM. Subclasses will provide
additional capabilities unique to each version of SPECFEM.

.. note::
    The Base class implementation is almost completely SPECFEM2D related.
    However, SPECFEM2D requires a few unique parameters that 3D/3D_GLOBE
    do not. Because of the inheritance architecture of SeisFlows, we do not
    want the 3D and 3D_GLOBE versions to inherit 2D-specific parameters, so
    we need this this more generalized SPECFEM base class.

TODO
    - add in `apply_hess` functionality that was partially written in legacy code
    - move `_initialize_adjoint_traces` to workflow.migration
    - Add density scaling based on Vp?
"""
import os
import sys
import subprocess
from glob import glob

from seisflows import logger
from seisflows.core import Dict
from seisflows.plugins import solver_io as solver_io_dir
from seisflows.tools import msg, unix
from seisflows.tools.utils import get_task_id
from seisflows.tools.specfem import getpar, setpar


class Specfem:
    """
    [solver.specfem] SPECFEM interface shared between 2D/3D/3D_GLOBE
    implementations providing generalized interface to establish SPECFEM
    working directories, call SPECFEM binaries, and keep track of a number of
    parallel processes.

    :type case: str
    :param case: determine the type of workflow we will attempt
        Available: ['DATA': data-synthetic comparisons. Data must be provided
        by the user in `path_data`. 'SYNTHETIC': synthetic-synthetic
        comparisons. `path_model_true` is required to generate target 'data']
    :type data_format: str
    :param data_format: data format for reading traces into memory.
        Available: ['SU' seismic unix format, 'ASCII' human-readable ascii]
    :type materials: str
    :param materials: Material parameters used to define model. Available:
        ['ELASTIC': Vp, Vs, 'ACOUSTIC': Vp, 'ISOTROPIC', 'ANISOTROPIC']
    :type density: bool
    :param density: How to treat density during inversion. If True, updates
        density during inversion. If False, keeps it constant.
    :type attenuation: bool
    :param attenuation: How to treat attenuation during inversion.
        if True, turns on attenuation during forward simulations only. If
        False, attenuation is always set to False. Requires underlying
        attenution (Q_mu, Q_kappa) model
    :type components: str
    :param components: components to consider and tag data with. Should be
        string of letters such as 'RTZ'
    :type solver_io: str
    :param solver_io: format of model/kernel/gradient files expected by the
        numerical solver. Available: ['fortran_binary': default .bin files].
        TODO: ['adios': ADIOS formatted files]
    :type source_prefix: str
    :param source_prefix: prefix of source/event/earthquake files. If None,
        will attempt to guess based on the specific solver chosen.
    :type mpiexec: str
    :param mpiexec: MPI executable used to run parallel processes. Should also
        be defined for the system module
    :type workdir: str
    :param workdir: working directory in which to look for data and store
        results. Defaults to current working directory
    :type path_solver: str
    :param path_solver: scratch path for all solver related tasks
    :type path_data: str
    :param path_data: path to any externally stored data required by the solver
    :type path_specfem_bin: str
    :param path_specfem_bin: path to SPECFEM bin/ directory which
        contains binary executables for running SPECFEM
    :type path_specfem_data: str
    :param path_specfem_data: path to SPECFEM DATA/ directory which must
        contain the CMTSOLUTION, STATIONS and Par_file files used for
        running SPECFEM
    :type path_model_true: str
    :param path_model_true: path to a target model if `case`=='synthetic' and
        a set of synthetic 'observations' are required for workflow.
    :type path_output: str
    :param path_output: shared output directory on disk for more permanent
        storage of solver related files such as traces, kernels, gradients.
    """
    def __init__(self, case="data", data_format="ascii",  materials="acoustic",
                 density=False, nproc=1, ntask=1, attenuation=False,
                 components="ZNE", solver_io="fortran_binary",
                 source_prefix=None, mpiexec=None, workdir=os.getcwd(),
                 path_solver=None, path_eval_grad=None, path_data=None,
                 path_specfem_bin=None, path_specfem_data=None,
                 path_model_init=None, path_model_true=None, path_output=None,
                 **kwargs):
        """Set default SPECFEM interface parameters"""
        self.case = case
        self.data_format = data_format
        self.materials = materials
        self.nproc = nproc
        self.ntask = ntask
        self.density = density
        self.attenuation = attenuation
        self.components = components
        self.solver_io = solver_io
        self.mpiexec = mpiexec
        self.source_prefix = source_prefix or "SOURCE"

        # Define internally used directory structure
        self.path = Dict(
            scratch=path_solver or os.path.join(workdir, "scratch", "solver"),
            eval_grad=path_eval_grad or
                      os.path.join(workdir, "scratch", "evalgrad"),
            data=path_data or os.path.join(workdir, "SFDATA"),
            output=path_output or os.path.join(workdir, "output"),
            specfem_bin=path_specfem_bin,
            specfem_data=path_specfem_data,
            model_init=path_model_init,
            model_true=path_model_true,
        )
        self.path.mainsolver = os.path.join(self.path.scratch, "mainsolver")

        # Establish internally defined parameter system
        self._parameters = []
        if self.density:
            self._parameters.append("rho")

        self._source_names = None
        self._io = getattr(solver_io_dir, self.solver_io)

        # Define available choices for check parameters
        self._available_model_types = ["gll"]
        self._available_materials = [
            "ELASTIC", "ACOUSTIC",  # specfem2d, specfem3d
            "ISOTROPIC", "ANISOTROPIC"  # specfem3d_globe
        ]
        self._available_data_formats = ["ASCII", "SU"]
        self._required_binaries = ["xspecfem2D", "xmeshfem2D", "xcombine_sem",
                                   "xsmooth_sem"]
        self._acceptable_source_prefixes = ["SOURCE", "FORCE", "FORCESOLUTION"]

    def check(self):
        """
        Checks parameter validity for SPECFEM input files and model parameters
        """
        assert(self.case.upper() in ["DATA", "SYNTHETIC"]), \
            f"solver.case must be 'DATA' or 'SYNTHETIC'"

        assert(self.materials.upper() in self._available_materials), \
            f"solver.materials must be in {self._available_materials}"

        if self.data_format.upper() not in self._available_data_formats:
            raise NotImplementedError(
                f"solver.data_format must be {self._available_data_formats}"
            )

        # Make sure we can read in the model/kernel/gradient files
        assert hasattr(solver_io_dir, self.solver_io)
        assert hasattr(self._io, "read_slice"), \
            "IO method has no attribute 'read'"
        assert hasattr(self._io, "write_slice"), \
            "IO method has no attribute 'write'"

        # Check that User has provided appropriate bin/ and DATA/ directories
        for name, dir_ in zip(["bin/", "DATA/"],
                              [self.path.specfem_bin, self.path.specfem_data]):
            assert(dir_ is not None), f"SPECFEM path '{name}' cannot be None"
            assert(os.path.exists(dir_)), f"SPECFEM path '{name}' must exist"

        # Check that the required SPECFEM files are available
        for fid in ["STATIONS", "Par_file"]:
            assert(os.path.exists(os.path.join(self.path.specfem_data, fid))), (
                f"DATA/{fid} does not exist but is required by SeisFlows solver"
            )

        # Make sure source files exist and are appropriately labeled
        assert(self.source_prefix in self._acceptable_source_prefixes)
        assert(glob(os.path.join(self.path.specfem_data,
                                 f"{self.source_prefix}*"))), (
            f"No source files with prefix {self.source_prefix} found in DATA/")

        # Check that required binary files exist which are called upon by solver
        for fid in self._required_binaries:
            assert(os.path.exists(os.path.join(self.path.specfem_bin, fid))), (
                f"bin/{fid} does not exist but is required by SeisFlows solver"
            )

        # Check that model type is set correctly in the Par_file
        model_type = getpar(key="MODEL",
                            file=os.path.join(self.path.specfem_data,
                                              "Par_file"))[1]
        assert(model_type in self._available_model_types), \
            f"{model_type} not in available types {self._available_model_types}"

        # Check that the 'case' variable matches required models
        if self.case.upper() == "SYNTHETIC":
            assert(os.path.exists(self.path.model_true)), (
                f"solver.case == 'synthetic' requires `path_model_true`"
            )

        assert(self.path.model_init is not None and
               os.path.exists(self.path.model_init)), \
            f"`path_model_init` is required for the solver, but does not exist"

        # Check that the number of tasks/events matches the number of events
        self._source_names = self._check_source_names()

    @property
    def taskid(self):
        """
        Returns the currently running process for embarassingly parallelized
        tasks. Task IDs are assigned to the environment by system.run().
        Task IDs are simply integer values from 0 to the number of
        simultaneously running tasks.

        .. note::
            Dependent on environment variable 'SEISFLOWS_TASKID' which is
            assigned by system.run() to each individually running process.

        :rtype: int
        :return: task id for given solver
        """
        return get_task_id()

    @property
    def source_names(self):
        """
        Returns list of source names which should be stored in PAR.SPECFEM_DATA
        Source names are expected to match the following wildcard,
        'PREFIX_*' where PREFIX is something like 'CMTSOLUTION' or 'FORCE'

        .. note::
            Dependent on environment variable 'SEISFLOWS_TASKID' which is
            assigned by system.run() to each individually running process.

        :rtype: list
        :return: list of source names
        """
        if self._source_names is None:
            self._source_names = self._check_source_names()
        return self._source_names

    @property
    def source_name(self):
        """
        Returns name of source currently under consideration

        .. note::
            Dependent on environment variable 'SEISFLOWS_TASKID' which is
            assigned by system.run() to each individually running process.

        :rtype: str
        :return: given source name for given task id
        """
        return self.source_names[self.taskid]

    @property
    def cwd(self):
        """
        Returns working directory currently in use by a running solver instance

        .. note::
            Dependent on environment variable 'SEISFLOWS_TASKID' which is
            assigned by system.run() to each individually running process.

        :rtype: str
        :return: current solver working directory
        """
        return os.path.join(self.path.scratch, self.source_name)

    def data_wildcard(self, comp="?"):
        """
        Returns a wildcard identifier for synthetic data based on SPECFEM2D
        file naming schema. Allows formatting dcomponent e.g.,
        when called by solver.data_filenames.

        .. note::
            SPECFEM3D/3D_GLOBE versions must overwrite this function

        :type comp: str
        :param comp: component formatter, defaults to wildcard '?'
        :rtype: str
        :return: wildcard identifier for channels
        """
        if self.data_format.upper() == "SU":
            return f"U{comp}_file_single.su"
        elif self.data_format.upper() == "ASCII":
            return f"*.?X{comp}.sem?"

    def data_filenames(self, choice="obs"):
        """
        Returns the filenames of SPECFEM2D data, either by the requested
        components or by all available files in the directory.

         .. note::
            SPECFEM3D/3D_GLOBE versions must overwrite this function

        .. note::
            If the glob returns an  empty list, this function exits the
            workflow because filenames should not be empty is they're being
            queried

        :rtype: list
        :return: list of data filenames
        """
        assert(choice in ["obs", "syn", "adj"]), \
            f"choice must be: 'obs', 'syn' or 'adj'"
        unix.cd(os.path.join(self.cwd, "traces", choice))

        filenames = []
        if self.components:
            for comp in self.components:
                filenames = glob(self.data_wildcard(comp=comp.lower()))
        else:
            filenames = glob(self.data_wildcard(comp="?"))

        if not filenames:
            print(msg.cli("The property solver.data_filenames, used to search "
                          "for traces in 'scratch/solver/*/traces' is empty "
                          "and should not be. Please check solver parameters: ",
                          items=[f"data_wildcard: {self.data_wildcard()}"],
                          header="data filenames error", border="=")
                  )
            sys.exit(-1)

        return filenames

    @property
    def model_databases(self):
        """
        The location of model inputs and outputs as defined by SPECFEM2D.
        This is RELATIVE to a SPECFEM2D working directory.

         .. note::
            This path is SPECFEM version dependent so SPECFEM3D/3D_GLOBE
            versions must overwrite this function

        :rtype: str
        :return: path where SPECFEM2D database files are stored
        """
        return "DATA"

    @property
    def kernel_databases(self):
        """
        The location of kernel inputs and outputs as defined by SPECFEM2D
        This is RELATIVE to a SPECFEM2D working directory.

         .. note::
            This path is SPECFEM version dependent so SPECFEM3D/3D_GLOBE
            versions must overwrite this function

        :rtype: str
        :return: path where SPECFEM2D database files are stored
        """
        return "OUTPUT_FILES"

    def setup(self):
        """
        Prepares solver scratch directories for an impending workflow.

        Sets up directory structure expected by SPECFEM and copies or generates
        seismic data to be inverted or migrated

        TODO the .bin during model export assumes GLL file format, more general?
        """
        self._initialize_working_directories()

        # Export the initial model to the SeisFlows output directory
        unix.mkdir(self.path.output)
        for key in self._parameters:
            src = glob(os.path.join(self.path.model_init, f"*{key}.bin"))
            dst = os.path.join(self.path.output, "MODEL_INIT")
            unix.cp(src, dst)

    # def generate_data(self, save_traces=False):
    #     """
    #     Generates observation data to be compared to synthetics. This must
    #     only be run once. If `PAR.CASE`=='data', then real data will be copied
    #     over.
    #  TODO move this to workflow
    #
    #     If `PAR.CASE`=='synthetic' then the external solver will use the
    #     True model to generate 'observed' synthetics. Finally exports traces to
    #     'cwd/traces/obs'
    #
    #     Elif `PAR.CASE`=='DATA', will look in PATH.DATA for directories matching
    #     the given source name and copy ANY files that exist there. e.g., if
    #     source name is '001', you must store waveform data in PATH.DATA/001/*
    #
    #     Also exports observed data to OUTPUT if desired
    #     """
    #     # If synthetic inversion, generate 'data' with solver
    #     if self.case.upper() == "SYNTHETIC":
    #         if self.path.model_true is not None:
    #             if self.taskid == 0:
    #                 logger.info("generating 'data' with MODEL_TRUE")
    #
    #             # Generate synthetic data on the fly using the true model
    #             self.import_model(path_model=self.path.model_true)
    #             self.forward_simulation(
    #                 save_traces=os.path.join("traces", "obs")
    #             )
    #     # If Data provided by user, copy directly into the solver directory
    #     elif self.path.data is not None and os.path.exists(self.path.data):
    #         unix.cp(
    #             src=glob(os.path.join(self.path.data, self.source_name, "*")),
    #             dst=os.path.join(self.cwd, "traces", "obs")
    #         )
    #
    #     # Save observation data to disk
    #     if save_traces:
    #         self._export_traces(
    #             path=os.path.join(self.path.output, "traces", "obs")
    #         )

    def generate_data(self, export_traces=False):
        """
        Generates observation data to be compared to synthetics. This must
        only be run once. If `PAR.CASE`=='data', then real data will be copied
        over.

        If `PAR.CASE`=='synthetic' then the external solver will use the
        True model to generate 'observed' synthetics. Finally exports traces to
        'cwd/traces/obs'

        Elif `PAR.CASE`=='DATA', will look in PATH.DATA for directories matching
        the given source name and copy ANY files that exist there. e.g., if
        source name is '001', you must store waveform data in PATH.DATA/001/*

        :type export_traces: str
        :param export_traces: path to copy and save traces to a more permament
            storage location as waveform stored in scratch/ are liable to be
            deleted or overwritten
        """
        # Basic checks to make sure there are True model files to copy
        assert(self.case.upper() == "SYNTHETIC")
        assert(os.path.exists(self.path.model_true))
        assert(glob(os.path.join(self.path.model_true, "*")))

        # Generate synthetic data on the fly using the true model
        self.import_model(path_model=self.path.model_true)
        self.forward_simulation(
            save_traces=os.path.join(self.cwd, "traces", "obs"),
            export_traces=export_traces
        )

    def import_data(self):
        """
        Import data from an existing directory into the current working
        directory, required if 'observed' waveform data will be provided by
        the User rather than automatically collected (with Pyatoa) or generated
        synthetically (with external solver)
        """
        # Simple checks to make sure we can actually import data
        assert(self.case.upper() == "DATA")
        assert(self.path.data is not None)
        assert(os.path.exists(os.path.join(self.path.data, self.source_name)))
        assert(glob(os.path.join(self.path.data, self.source_name, "*")))

        src = os.path.join(self.path.data, self.source_name, "*")
        dst = os.path.join(self.cwd, "traces", "obs")

        unix.cp(src, dst)

    def forward_simulation(self, executables=None, save_traces=False,
                           export_traces=False):
        """
        Wrapper for SPECFEM binaries: 'xmeshfem?D' 'xgenerate_databases',
                                      'xspecfem?D'

        Calls SPECFEM2D forward solver, exports solver outputs to traces dir

         .. note::
            SPECFEM3D/3D_GLOBE versions must overwrite this function

        :type executables: list or None
        :param executables: list of SPECFEM executables to run, in order, to
            complete a forward simulation. This can be left None in most cases,
            which will select default values based on the specific solver
            being called (2D/3D/3D_GLOBE). It is made an optional parameter
            to keep the function more general for inheritance purposes.
        :type save_traces: str
        :param save_traces: move files from their native SPECFEM output location
            to another directory. This is used to move output waveforms to
            'traces/obs' or 'traces/syn' so that SeisFlows knows where to look
            for them, and so that SPECFEM doesn't overwrite existing files
            during subsequent forward simulations
        :type export_traces: str
        :param export_traces: export traces from the scratch directory to a more
            permanent storage location. i.e., copy files from their original
            location
        """
        if executables is None:
            executables = ["bin/xmeshfem2D", "bin/xspecfem2D"]
        unix.cd(self.cwd)

        setpar(key="SIMULATION_TYPE", val="1", file="DATA/Par_file")
        setpar(key="SAVE_FORWARD", val=".true.", file="DATA/Par_file")

        # Calling subprocess.run() for each of the binary executables listed
        for exc in executables:
            # e.g., fwd_mesher.log
            stdout = f"fwd_{self._exc2log(exc)}.log"
            self._run_binary(executable=exc, stdout=stdout)

        # Work around SPECFEM's version dependent file names
        if self.data_format.upper() == "SU":
            for tag in ["d", "v", "a", "p"]:
                unix.rename(old=f"single_{tag}.su", new="single.su",
                            names=glob(os.path.join("OUTPUT_FILES", "*.su")))

        if export_traces:
            unix.cp(
                src=glob(os.path.join("OUTPUT_FILES", self.data_wildcard())),
                dst=export_traces
            )

        if save_traces:
            unix.mv(
                src=glob(os.path.join("OUTPUT_FILES", self.data_wildcard())),
                dst=save_traces
            )

    def adjoint_simulation(self, executables=None, save_kernels=False,
                           export_kernels=False):
        """
        Wrapper for SPECFEM binary 'xspecfem?D'

        Calls SPECFEM2D adjoint solver, creates the `SEM` folder with adjoint
        traces which is required by the adjoint solver. Renames kernels
        after they have been created from 'alpha' and 'beta' to 'vp' and 'vs',
        respectively.

         .. note::
            SPECFEM3D/3D_GLOBE versions must overwrite this function

        :type executables: list or None
        :param executables: list of SPECFEM executables to run, in order, to
            complete an adjoint simulation. This can be left None in most cases,
            which will select default values based on the specific solver
            being called (2D/3D/3D_GLOBE). It is made an optional parameter
            to keep the function more general for inheritance purposes.
        :type save_kernels: str
        :param save_kernels: move the kernels from their native SPECFEM output
            location to another path. This is used to move kernels to another
            SeisFlows scratch directory so that they are discoverable by
            other modules. The typical location they are moved to is
            path_eval_grad
        :type export_kernels: str
        :param export_kernels: export/copy/save kernels from the scratch
            directory to a more permanent storage location. i.e., copy files
            from their original location. Note that kernel file sizes are LARGE,
            so exporting kernels can lead to massive storage requirements.
        """
        if executables is None:
            executables = ["bin/xspecfem2D"]

        unix.cd(self.cwd)

        setpar(key="SIMULATION_TYPE", val="3", file="DATA/Par_file")
        setpar(key="SAVE_FORWARD", val=".false.", file="DATA/Par_file")

        unix.rm("SEM")
        unix.ln("traces/adj", "SEM")

        # Calling subprocess.run() for each of the binary executables listed
        for exc in executables:
            # e.g., adj_solver.log
            stdout = f"adj_{self._exc2log(exc)}.log"
            logger.info(f"running SPECFEM executable {exc}, log to '{stdout}'")
            self._run_binary(executable=exc, stdout=stdout)

        # Rename kernels to work w/ conflicting name conventions
        unix.cd(self.kernel_databases)
        logger.info(f"renaming event kernels for {self.source_name}")
        for tag in ["alpha", "alpha[hv]", "reg1_alpha", "reg1_alpha[hv]"]:
            names = glob(f"*proc??????_{tag}_kernel.bin")
            unix.rename(old="alpha", new="vp", names=names)

        for tag in ["beta", "beta[hv]", "reg1_beta", "reg1_beta[hv]"]:
            names = glob(f"*proc??????_{tag}_kernel.bin")
            unix.rename(old="beta", new="vs", names=names)

        # Save and export the kernels to user-defined locations
        if export_kernels:
            unix.cp(src=glob("*_kernel.bin"), dst=export_kernels)

        if save_kernels:
            unix.mv(src=glob("*_kernel.bin"), dst=save_kernels)

    def combine(self, input_path, output_path, parameters=None):
        """
        Wrapper for 'xcombine_sem'.
        Sums kernels from individual source contributions to create gradient.

        .. note::
            The binary xcombine_sem simply sums matching databases (.bin)

        .. note::
            It is ASSUMED that this function is being called by
            system.run(single=True) so that we can use the main solver
            directory to perform the kernel summation task

        :type input_path: str
        :param input_path: path to data
        :type output_path: strs
        :param output_path: path to export the outputs of xcombine_sem
        :type parameters: list
        :param parameters: optional list of parameters,
            defaults to `self.parameters`
        """
        unix.cd(self.cwd)

        if parameters is None:
            parameters = self._parameters

        if not os.path.exists(output_path):
            unix.mkdir(output_path)

        # Write the source names into the kernel paths file for SEM/ directory
        with open("kernel_paths", "w") as f:
            f.writelines(
                [os.path.join(input_path, f"{name}\n")
                 for name in self.source_names]
            )

        # Call on xcombine_sem to combine kernels into a single file
        for name in parameters:
            # e.g.: mpiexec bin/xcombine_sem alpha_kernel kernel_paths output/
            exc = f"bin/xcombine_sem {name}_kernel kernel_paths {output_path}"
            # e.g., smooth_vp.log
            stdout = f"{self._exc2log(exc)}_{name}.log"
            self._run_binary(executable=exc, stdout=stdout)

    def smooth(self, input_path, output_path, parameters=None, span_h=0.,
               span_v=0., use_gpu=False):
        """
        Wrapper for SPECFEM binary: xsmooth_sem
        Smooths kernels by convolving them with a 3D Gaussian

        .. note::
            It is ASSUMED that this function is being called by
            system.run(single=True) so that we can use the main solver
            directory to perform the kernel smooth task

        :type input_path: str
        :param input_path: path to data
        :type output_path: str
        :param output_path: path to export the outputs of xcombine_sem
        :type parameters: list
        :param parameters: optional list of parameters,
            defaults to `self.parameters`
        :type span_h: float
        :param span_h: horizontal smoothing length in meters
        :type span_v: float
        :param span_v: vertical smoothing length in meters
        :type use_gpu: bool
        :param use_gpu: whether to use GPU acceleration for smoothing. Requires
            GPU compiled binaries and GPU compute node.
        """
        unix.cd(self.cwd)

        if parameters is None:
            parameters = self._parameters

        if not os.path.exists(output_path):
            unix.mkdir(output_path)

        # Ensure trailing '/' character, required by xsmooth_sem
        input_path = os.path.join(input_path, "")
        output_path = os.path.join(output_path, "")
        if use_gpu:
            use_gpu = ".true"
        else:
            use_gpu = ".false"
        # mpiexec ./bin/xsmooth_sem SMOOTH_H SMOOTH_V name input output use_gpu
        for name in parameters:
            exc = (f"bin/xsmooth_sem {str(span_h)} {str(span_v)} {name}_kernel "
                   f"{input_path} {output_path} {use_gpu}")
            # e.g., combine_vs.log
            stdout = f"{self._exc2log(exc)}_{name}.log"
            self._run_binary(executable=exc, stdout=stdout)

        # Rename output files to remove the '_smooth' suffix which SeisFlows
        # will not recognize
        files = glob(os.path.join(output_path, "*"))
        unix.rename(old="_smooth", new="", names=files)

    def _run_binary(self, executable, stdout="solver.log"):
        """
        Calls MPI solver executable to run solver binaries, used by individual
        processes to run the solver on system. If the external solver returns a
        non-zero exit code (failure), this function will return a negative
        boolean.

        .. note::
            This function ASSUMES it is being run from a SPECFEM working
            directory, i.e., that the executables are located in ./bin/

        .. note::
            This is essentially an error-catching wrapper of subprocess.run()

        :type executable: str
        :param executable: executable function to call. May or may not start
            E.g., acceptable calls for the solver would './bin/xspecfem2D'.
            Also accepts additional command line arguments such as:
            'xcombine_sem alpha_kernel kernel_paths...'
        :type stdout: str
        :param stdout: where to redirect stdout
        :raises SystemExit: If external numerical solver return any failure
            code while running
        """
        # Executable may come with additional sub arguments, we only need to
        # check that the actually executable exists
        if not unix.which(executable.split(" ")[0]):
            print(msg.cli(f"executable '{executable}' does not exist",
                          header="external solver error", border="="))
            sys.exit(-1)

        # Append with mpiexec if we are running with MPI
        if self.mpiexec:
            executable = f"{self.mpiexec} {executable}"

        try:
            with open(stdout, "w") as f:
                subprocess.run(executable, shell=True, check=True, stdout=f)
        except (subprocess.CalledProcessError, OSError) as e:
            print(msg.cli("The external numerical solver has returned a "
                          "nonzero exit code (failure). Consider stopping any "
                          "currently running jobs to avoid wasted "
                          "computational resources. Check 'scratch/solver/"
                          "mainsolver/{stdout}' for the solvers stdout log "
                          "message. The failing command and error message are:",
                          items=[f"exc: {executable}", f"err: {e}"],
                          header="external solver error",
                          border="=")
                  )
            sys.exit(-1)

    @staticmethod
    def _exc2log(exc):
        """
        Very simple conversion utility to get log file names based on binaries.
        e.g., binary 'xspecfem2D' will return 'solver'. Helps keep log file
        naming consistent and generalizable

        :type exc: str
        :param exc: specfem executable, e.g., xspecfem2D, xgenerate_databases
        :rtype: str
        :return: logfile name that matches executable name
        """
        convert_dict = {"specfem": "solver", "meshfem": "mesher",
                        "generate_databases": "mesher", "smooth": "smooth",
                        "combine": "combine"}
        for key, val in convert_dict.items():
            if key in exc:
                return val
        else:
            return "logger"

    def import_model(self, path_model):
        """
        Copy files from given `path_model` into the current working directory
        model database. Used for grabbing starting models (e.g., MODEL_INIT)
        and models that have been perturbed by the optimization library.

        :type path_model: str
        :param path_model: path to an existing starting model
        """
        assert(os.path.exists(path_model)), f"model {path_model} does not exist"
        unix.cd(self.cwd)

        # Copy the model files (ex: proc000023_vp.bin ...) into database dir
        src = glob(os.path.join(path_model, "*"))
        dst = os.path.join(self.cwd, self.model_databases, "")
        unix.cp(src, dst)

    def _initialize_working_directories(self):
        """
        Serial task used to initialize working directories for each of the a
        available sources

        TODO run this with concurrent futures for speedup?
        """
        logger.info(f"initializing {self.ntask} solver directories")
        for source_name in self.source_names:
            cwd = os.path.join(self.path.scratch, source_name)
            self._initialize_working_directory(cwd=cwd)

    def _initialize_working_directory(self, cwd=None):
        """
        Creates directory structure expected by SPECFEM
        (i.e., bin/, DATA/, OUTPUT_FILES/). Copies executables and prepares
        input files.

        Each directory will act as completely independent Specfem working dir.
        This allows for embarrassing parallelization while avoiding the need
        for intra-directory communications, at the cost of temporary disk space.

        .. note::
            Path to binary executables must be supplied by user as SeisFlows has
            no mechanism for automatically compiling from source code.

        :type cwd: str
        :param cwd: optional scratch working directory to intialize. If None,
            will set based on current running seisflows task (self.taskid)
        """
        # Define a constant list of required SPECFEM dir structure, relative cwd
        _required_structure = ["bin", "DATA",
                               "traces/obs", "traces/syn", "traces/adj",
                               self.model_databases, self.kernel_databases]

        # Allow this function to be called on system or in serial
        if cwd is None:
            cwd = self.cwd
            taskid = self.taskid
        else:
            cwd = self.cwd
            _source_name = os.path.basename(cwd)
            taskid = self.source_names.index(_source_name)

        if taskid == 0:
            logger.info(f"initializing {self.ntask} solver directories")

        # Starting from a fresh working directory
        unix.rm(cwd)
        unix.mkdir(cwd)
        for dir_ in _required_structure:
            unix.mkdir(os.path.join(cwd, dir_))

        # Copy existing SPECFEM exectuables into the bin/ directory
        src = glob(os.path.join(self.path.specfem_bin, "*"))
        dst = os.path.join(cwd, "bin", "")
        unix.cp(src, dst)

        # Copy all input DATA/ files except the source files
        src = glob(os.path.join(self.path.specfem_data, "*"))
        src = [_ for _ in src if self.source_prefix not in _]
        dst = os.path.join(cwd, "DATA", "")
        unix.cp(src, dst)

        # Symlink event source specifically, only retain source prefix
        src = os.path.join(self.path.specfem_data,
                           f"{self.source_prefix}_{self.source_name}")
        dst = os.path.join(cwd, "DATA", self.source_prefix)
        unix.ln(src, dst)

        # Symlink TaskID==0 as mainsolver in solver directory for convenience
        if taskid == 0:
            if not os.path.exists(self.path.mainsolver):
                logger.debug(f"symlink {self.source_name} as 'mainsolver'")
                unix.ln(cwd, self.path.mainsolver)

    # def _initialize_adjoint_traces(self):
    #     """
    #     Setup utility: Creates the "adjoint traces" expected by SPECFEM.
    #     This is only done for the 'base' the Preprocess class.
    #
    #     TODO move this into workflow setup
    #
    #     .. note::
    #         Adjoint traces are initialized by writing zeros for all channels.
    #         Channels actually in use during an inversion or migration will be
    #         overwritten with nonzero values later on.
    #     """
    #     preprocess = self.module("preprocess")
    #
    #     if self.par.PREPROCESS.upper() == "DEFAULT":
    #         if self.taskid == 0:
    #             logger.debug(f"intializing {len(self.data_filenames)} "
    #                          f"empty adjoint traces per event")
    #
    #         for filename in self.data_filenames:
    #             st = preprocess.reader(
    #                         path=os.path.join(self.cwd, "traces", "obs"),
    #                         filename=filename
    #                         )
    #             # Zero out data just so we have empty adjoint traces as SPECFEM
    #             # will expect all adjoint sources to have all components
    #             st *= 0
    #
    #             # Write traces back to the adjoint trace directory
    #             preprocess.writer(st=st, filename=filename,
    #                               path=os.path.join(self.cwd, "traces", "adj")
    #                               )

    def _check_source_names(self):
        """
        Determines names of sources by applying wildcard rule to user-supplied
        input files. Source names are only provided up to PAR.NTASK and are
        returned in alphabetical order.

        :rtype: list
        :return: alphabetically ordered list of source names up to PAR.NTASK
        """
        assert(self.path.specfem_data is not None), \
            f"solver source names requires 'solver.path.specfem_data' to exist"
        assert(os.path.exists(self.path.specfem_data)), \
            f"solver source names requires 'solver.path.specfem_data' to exist"

        # Apply wildcard rule and check for available sources, exit if no
        # sources found because then we can't proceed
        wildcard = f"{self.source_prefix}_*"
        fids = sorted(glob(os.path.join(self.path.specfem_data, wildcard)))
        if not fids:
            print(msg.cli("No matching source files when searching PATH for "
                          "the given WILDCARD",
                          items=[f"PATH: {self.path.specfem_data}",
                                 f"WILDCARD: {wildcard}"], header="error"
                          )
                  )
            sys.exit(-1)
        else:
            assert(len(fids) >= self.ntask), (
                f"Number of requested tasks/events {self.ntask} exceeds number "
                f"of available sources {len(fids)}"
            )

        # Create internal definition of sources names by stripping prefixes
        names = [os.path.basename(fid).split("_")[-1] for fid in fids]

        return names[:self.ntask]
