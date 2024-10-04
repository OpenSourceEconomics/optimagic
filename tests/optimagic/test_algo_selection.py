from optimagic import algos


def test_dfols_is_present():
    assert hasattr(algos, "nag_dfols")
    assert hasattr(algos.Bounded, "nag_dfols")
    assert hasattr(algos.LeastSquares, "nag_dfols")
    assert hasattr(algos.Local, "nag_dfols")
    assert hasattr(algos.Bounded.Local.LeastSquares, "nag_dfols")
    assert hasattr(algos.Local.Bounded.LeastSquares, "nag_dfols")
    assert hasattr(algos.LeastSquares.Bounded.Local, "nag_dfols")
