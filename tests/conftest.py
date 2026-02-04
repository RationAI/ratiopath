import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for deterministic behavior."""
    np.random.seed(42)
