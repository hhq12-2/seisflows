"""
Test any of the utility functions defined in the Tools directory
"""
import dill
import os
import pytest
import numpy as np
from glob import glob
from seisflows import ROOT_DIR
from seisflows.tools.model import Model
from seisflows.tools.config import (Dict, load_yaml, custom_import, get_task_id, 
                                    set_task_id, pickle_function_list)
from seisflows.tools.math import (angle, poissons_ratio)


TEST_DIR = os.path.join(ROOT_DIR, "tests")
TEST_MODEL = os.path.join(TEST_DIR, "test_data", "test_tools", 
                          "test_file_formats")


# =============================================================================
# TEST MODEL CLASS 
# =============================================================================
def test_model_attributes():
    """
    Check that model values are read in correctly and accessible in a way we 
    expect
    """
    m = Model(path=TEST_MODEL, fmt=".bin")

    assert(m.ngll[0] == 40000)
    assert(m.nproc == 1)
    assert("vp" in m.model.keys())
    assert("vs" in m.model.keys())
    assert(m.model.vp[0][0] == 5800.)
    assert(m.model.vs[0][0] == 3500.)


def test_model_functions():
    """
    Test the core merge and split functions of the Model class to ensure
    they give the same results
    """
    m = Model(path=TEST_MODEL, fmt=".bin")

    assert(len(m.merge() == len(m.model.vs[0]) + len(m.model.vp[0])))
    assert(len(m.split()) == len(m.parameters))


def test_model_io(tmpdir):
    """
    Check that saving and loading npz file works
    """
    m = Model(path=TEST_MODEL, fmt=".bin")

    m.save(path=os.path.join(tmpdir, "test.npz"))
    m_new = Model(path=os.path.join(tmpdir, "test.npz"))
    assert(m_new.ngll[0] == m.ngll[0])
    assert(m_new.fmt == m.fmt)

    # Check that writing fortran binary works
    m.write(path=tmpdir)
    assert(len(glob(os.path.join(tmpdir, f"*{m.fmt}"))) == 2)


def test_model_from_input_vector():
    """Check that we can instantiate a model from an input vector"""
    m = Model(path=None)
    m.model = Dict(x=[np.array([-1.2, 1.])])
    assert(m.nproc == 1)
    assert(m.ngll == [2])
    assert(m.parameters == ["x"])

# =============================================================================
# TEST CONFIG FUNCTIONS
# =============================================================================
def test_custom_import():
    """
    Test that importing based on internal modules works for various inputs
    :return:
    """
    with pytest.raises(SystemExit):
        custom_import()
    with pytest.raises(SystemExit):
        custom_import(name="NOT A VALID NAME")

    module = custom_import(name="optimize", module="LBFGS")
    assert(module.__name__ == "LBFGS")
    assert(module.__module__ == "seisflows.optimize.LBFGS")

    # Check one more to be safe
    module = custom_import(name="preprocess", module="default")
    assert(module.__name__ == "Default")
    assert(module.__module__ == "seisflows.preprocess.default")


def test_dict_class():
    """
    Test the functionality of the Dict class to make sure we can access and
    assign values a certain way (modified from GitHub Copilot suggestion)
    """
    # Create a dictionary
    data = {"key1": "value1", "key2": "value2", "key3": "value3"}
    d = Dict(data)

    # Test accessing values
    assert d.key1 == "value1"
    assert d.key2 == "value2"
    assert d.key3 == "value3"

    # Test modifying values
    d.key1 = "new_value1"
    assert d.key1 == "new_value1"

    # Test adding new key-value pairs
    d.key4 = "value4"
    assert d.key4 == "value4"

    # Test deleting key-value pairs
    del d["key2"]
    assert not hasattr(d, "key2")

def test_load_yaml(tmpdir):
    """
    Test the functionality of the load_yaml function to ensure it correctly
    loads YAML files (modified from GitHub Copilot suggestion)
    """
    # Create a temporary YAML file
    yaml_content = f"""
    key1: value1
    key2: value2
    key3:
        - item1
        - item2
        - item3
    key4: inf
    key5: null
    path_test: {tmpdir}/path/to/dir
    """
    yaml_file = os.path.join(tmpdir, "test.yaml")
    with open(yaml_file, "w") as f:
        f.write(yaml_content)

    # Load the YAML file
    config = load_yaml(yaml_file)

    # Test the loaded values
    assert config["key1"] == "value1"
    assert config["key2"] == "value2"
    assert config["key3"] == ["item1", "item2", "item3"]
    # Specific value changes only specified for SeisFlows
    assert config["key4"] == np.inf
    assert config["key5"] is None
    assert config["path_test"] == os.path.join(tmpdir, "path/to/dir")


def test_get_set_task_id():
    """
    Test the functionality of get_task_id and set_task_id functions
    (modified from GitHub Copilot suggestion)
    """
    # Set a task ID
    set_task_id(123)
    # Get the task ID and check if it matches the set value
    assert get_task_id() == 123
    # Set a different task ID
    set_task_id(456)
    # Get th task ID again and check if it matches the new set value
    assert get_task_id() == 456


def test_pickle_function_list(tmpdir):
    """
    Test the functionality of pickle_function_list to ensure it correctly
    pickles a list of functions (modified from GitHub Copilot suggestion) and
    kwargs and that they can be recovered
    """
    # Define a list of functions
    def add(a, b):
        return a + b

    def subtract(a, b):
        return a - b

    function_list = [add, subtract]
    kwargs = {"a": 8, "b": 5}

    # Pickle the function list
    pickle_function_list(function_list, path=tmpdir, **kwargs)

    # These are hardcoded naming conventions from the function
    pickle_file_name = os.path.join(tmpdir, "add_subtract.p")
    pickle_file_kwargs = os.path.join(tmpdir, "add_subtract_kwargs.p")

    # Load the pickled function list
    with open(pickle_file_name, "rb") as f:
        loaded_function_list = dill.load(f)
    
    with open(pickle_file_kwargs, "rb") as f:
        loaded_kwargs = dill.load(f)

    # Test the loaded function list and the loaded kwarg list by checking the
    # values add up
    assert len(loaded_function_list) == len(function_list)
    assert loaded_function_list[0](**loaded_kwargs) == 13
    assert loaded_function_list[1](**loaded_kwargs) == 3

# =============================================================================
# TEST MATH FUNCTIONS
# =============================================================================
def test_angle():
    """
    Test the angle function to ensure it correctly calculates the angle between
    two vectors (modified from GitHub Copilot suggestion)
    """
    # Define two vectors
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])

    # Calculate the angle between the two vectors
    val = angle(v1, v2)

    # Check the calculated angle
    assert np.isclose(val, np.pi / 2)


def test_dot():
    """
    Test the dot function to ensure it correctly calculates the dot product
    between two vectors (modified from GitHub Copilot suggestion)
    """
    # Define two vectors
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])

    # Calculate the dot product between the two vectors
    val = np.dot(v1, v2)

    # Check the calculated dot product
    assert val == 0


def test_poisons_ratio():
    """
    Test the poissons_ratio function to ensure it correctly calculates the
    Poisson's ratio between two vectors (modified from GitHub Copilot suggestion)
    """
    # Define two vectors
    vp = 9.2
    vs = 4.1

    # Calculate the Poisson's ratio between the two vectors
    val = poissons_ratio(vp, vs)

    # Check the calculated Poisson's ratio
    assert val == pytest.approx(0.376, rel=1e-3)

