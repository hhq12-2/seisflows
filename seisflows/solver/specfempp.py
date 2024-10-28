#!/usr/bin/env python3
"""
Base class for interaction with SPECFEM++
"""
import os
import yaml
from glob import glob

from seisflows.tools.config import get_task_id, Dict, load_yaml
from seisflows.tools import unix


class SpecfemPP:
    """
    Solver SPECFEMPP
    ----------------
    SPECFEM++-specific alterations to the base SPECFEM module

    Parameters
    ----------
    :type source_prefix: str
    :param source_prefix: Prefix of source files in path SPECFEM_DATA. Defaults
        to 'SOURCE'
    :type multiples: bool
    :param multiples: set an absorbing top-boundary condition

    Paths
    -----
    ***
    """
    __doc__ = Specfem.__doc__ + __doc__

    def __init__(self, **kwargs):
        """Instantiate a SpecfemPP solver interface"""

        self._source_names = []
        pass

    def check(self):
        """
        Checks on parameter validity
        """
        # TO DO
        pass

    def setup(self):
        """
        One-time module setup procedures
        """
        # TO DO
        pass

    def check_model_values(self, path):
        """
        Convenience function to check parameter and model validity for
        chosen Solver model, e.g., no negative velocity values, Vp and Vs 
        satisfty Poisson's ratio, etc. 
        """
        # TO DO
        pass

    @property
    def source_names(self):
        """
        Returns list of source names which should be stored in source.yaml file
        """
        # Parse through source.yaml and read in all sources, or parse through
        # multiple source.yaml files

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
        return self.source_names[get_task_id()]

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

    @property
    def model_databases(self):
        """
        The location of model inputs and outputs as defined by SPECFEM2D.
        This is RELATIVE to a SPECFEM2D working directory.

         .. note::
            This path is SPECFEM version dependent so SPECFEM3D/3D_GLOBE
            versions must overwrite this function

        :rtype: str
        :return: path where SPECFEM2D database files are stored, relative to
            `solver.cwd`
        """
        return "DATABASES_MPI"

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
        src = glob(os.path.join(path_model, f"*{self._ext}"))
        dst = os.path.join(self.cwd, self.model_databases, "")
        unix.cp(src, dst)


    def forward_simulation(self, save_traces=False,
                           export_traces=False, save_forward_arrays=False,
                           flag_save_forward=True, **kwargs):
        """
        Run forward simulation with proper setup and teardown procedures

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
        :type save_forward_arrays: str
        :param save_forward_arrays: relative path (relative to 
            /scratch/solver/<source_name>/<model_database>) to move the forward 
            arrays which are used for adjoint simulations. Mainly used for 
            ambient noise adjoint tomography which requires multiple forward 
            simulations prior to adjoint simulations, putting forward arrays 
            at the risk of overwrite. Normal Users can leave this default.
        :type flag_save_forward: bool
        :param flag_save_forward: whether to turn on the flag for saving the 
            forward arrays which are used for adjoint simulations. Not required 
            if only running forward simulations.
        """
        pass


def getpar(key, file, delim="=", match_partial=False, comment="#",):
    """
    Query a parameter from a SPECFEM++ YAML parameter file
    """
    # Load SPECFEM++ Yaml parameter file, get queried parameter and return
    pass


def setpar(key, val, file, delim="=", **kwargs):
    """
    Overwrite an existing parameter in a SPECFEM++ YAML parameter file
    """
    # Load YAML parameter file, overwrite current value, overwrite file
    pass
