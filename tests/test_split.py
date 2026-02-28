import numpy as np

from ratiopath.model_selection.split import (
    StratifiedGroupShuffleSplit,
    train_test_split,
)


def test_train_test_split_with_groups_and_stratify():
    x = np.arange(12).reshape(6, 2)
    y = np.array([0, 0, 1, 1, 0, 1])
    groups = np.array([1, 1, 2, 2, 3, 3])

    # include groups as one of the arrays so we can inspect split groups
    x_train, x_test, y_train, y_test, g_train, g_test = train_test_split(
        x, y, groups, test_size=0.33, random_state=0, stratify=y, groups=groups
    )

    # ensure groups do not overlap between train and test
    assert set(g_train).isdisjoint(set(g_test))

    # ensure stratification roughly preserved in the test set
    prop_full = (y == 0).sum() / len(y)
    prop_test = (y_test == 0).sum() / len(y_test)
    assert abs(prop_full - prop_test) <= 0.34


def test_train_test_split_with_groups_no_stratify():
    x = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 0, 1, 0])
    groups = np.array([1, 1, 2, 2, 3])

    x_train, x_test, y_train, y_test, g_train, g_test = train_test_split(
        x, y, groups, test_size=0.4, random_state=1, groups=groups
    )

    assert set(g_train).isdisjoint(set(g_test))


def test_stratified_group_shuffle_split_splits():
    x = np.arange(12).reshape(6, 2)
    y = np.array([0, 0, 1, 1, 0, 1])
    groups = np.array([1, 1, 2, 2, 3, 3])

    sgss = StratifiedGroupShuffleSplit(n_splits=5, test_size=0.33, random_state=42)

    for train_idx, test_idx in sgss.split(x, y, groups=groups):
        # groups should be non-overlapping
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)

        # indices should cover all samples
        assert len(train_idx) + len(test_idx) == len(x)

        # test must contain at least one sample
        assert len(test_idx) > 0
