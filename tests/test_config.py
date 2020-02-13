import pytest
from utils.config import Config as cfg


def test_Config_unknown_user():
    """Load configuration with an unknown user"""
    username = "unknown_user_with_a_very_long_name"
    with pytest.raises(KeyError, match=username + ": User path not defined, please add path in utils.config."):
        cfg(username="unknown_user_with_a_very_long_name")

def test_Config_this_machine():
    """Load configuration on this machine (path should be added)"""
    Config = cfg()
    Config.results_folder
