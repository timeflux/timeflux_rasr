import numpy as np


def test_simulation():
    from timeflux_rasr.helpers.simulation import (
        add_artifacts,
        clean_eeg,
        from_epochs,
        mixing_matrix,
        to_epochs,
    )

    mixing = mixing_matrix(n_channels=8, n_sources=10, random_state=0)
    X = clean_eeg(2500, 250, mixing, random_state=1)
    assert X.shape == (8, 2500)
    Xa, mask, kinds = add_artifacts(X, 250, 125, rate=0.5, random_state=2)
    assert mask.sum() == 125 * len(kinds)
    assert np.allclose(Xa[:, ~mask], X[:, ~mask])
    assert not np.allclose(Xa[:, mask], X[:, mask])
    E = to_epochs(X, 125)
    assert E.shape == (20, 125, 8)
    np.testing.assert_array_equal(from_epochs(E), X)
