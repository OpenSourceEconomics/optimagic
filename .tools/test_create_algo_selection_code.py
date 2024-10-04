from algo_selector import _generate_category_combinations


def test_generate_category_combinations():
    categories = ["a", "b", "c"]
    got = _generate_category_combinations(categories)
    expected = [
        ("a", "b", "c"),
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a",),
        ("b",),
        ("c",),
    ]
    assert got == expected
