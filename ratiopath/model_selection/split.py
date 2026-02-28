import numbers
from collections.abc import Iterator
from itertools import chain
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.model_selection import (
    BaseShuffleSplit,
    GroupShuffleSplit,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
)
from sklearn.model_selection._split import GroupsConsumerMixin, _validate_shuffle_split
from sklearn.utils._array_api import get_namespace_and_device, move_to
from sklearn.utils._indexing import _safe_indexing
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params
from sklearn.utils.validation import _num_samples, check_random_state, indexable


ArrayLike: TypeAlias = np.typing.ArrayLike
MatrixLike: TypeAlias = np.ndarray | pd.DataFrame | spmatrix
Int: TypeAlias = int | np.int8 | np.int16 | np.int32 | np.int64
Float: TypeAlias = float | np.float16 | np.float32 | np.float64


class StratifiedGroupShuffleSplit(GroupsConsumerMixin, BaseShuffleSplit):
    """Stratified shuffle split with non-overlapping groups.

    Provides train/test indices to split data such that both stratification
    (preserving class distribution) and grouping (non-overlapping groups between
    splits) are maintained.

    This splitter combines the functionality of StratifiedShuffleSplit and
    GroupShuffleSplit. It attempts to create folds which preserve the percentage
    of samples from each class while ensuring that samples from the same group
    do not appear in both train and test sets.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters:
    n_splits: Number of re-shuffling & splitting iterations.
    test_size: If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the test split. If int, represents the absolute number
        of test samples. If None, the value is set to the complement of the train size.
    train_size: If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the train split. If int, represents the absolute
        number of train samples. If None, the value is automatically set to the
        complement of the test size.
    random_state: Controls the randomness of the training and testing indices. Pass an
        int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples:
        >>> import numpy as np
        >>> from ratiopath.model_selection import StratifiedGroupShuffleSplit
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        >>> y = np.array([0, 0, 1, 1, 0, 1])
        >>> groups = np.array([1, 1, 2, 2, 3, 3])
        >>> sgss = StratifiedGroupShuffleSplit(n_splits=2, random_state=42)
        >>> for train_index, test_index in sgss.split(X, y, groups):
        ...     print(f"Train: {train_index}, Test: {test_index}")
        Train: [0 1 2 3], Test: [4 5]
        Train: [2 3 4 5], Test: [0 1]

    Notes:
        The implementation finds the best stratification split by trying multiple splits
        and selecting the one that minimizes the difference between the class
        distributions in the original data and the test split.

        Groups appear exactly once in the test set across all splits.
    """

    def __init__(
        self,
        n_splits: Int = 5,
        *,
        test_size: None | Float = None,
        train_size: None | Float = None,
        random_state: np.random.RandomState | None | Int = None,
    ) -> None:
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.2

    @staticmethod
    def _get_distribution(labels: ArrayLike) -> np.ndarray:
        _, counts = np.unique(labels, return_counts=True)
        return counts / counts.sum()

    def split(
        self,
        X: list[str] | MatrixLike,  # noqa: N803
        y: ArrayLike,
        groups: Any = None,
    ) -> Iterator[Any]:
        """Generate indices to split data into training and test set.

        Parameters:
            X: Training data, where ``n_samples`` is the number of samples and
                ``n_features`` is the number of features.
            y: The target variable for supervised learning problems. Stratification is
                done based on the y labels.
            groups: Group labels for the samples used while splitting the dataset into
                train and test set. Must be provided.

        Yields:
            train: The training set indices for that split.
            test: The testing set indices for that split.
        """
        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(
            n_samples, self.test_size, self.train_size, self._default_test_size
        )

        flipped = False
        if n_test > n_train:
            # Approximation using folds is terrible when the test set is larger than the train set
            n_test, n_train = n_train, n_test
            flipped = True

        n_splits = round(n_samples / n_test)
        rng = check_random_state(self.random_state)
        y = np.asarray(y)

        data_distribution = self._get_distribution(y)
        min_diff: Float | None = None
        train_index: np.ndarray | None = None
        test_index: np.ndarray | None = None

        for _ in range(self.n_splits):
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rng)

            for curr_train_index, curr_test_index in cv.split(X=X, y=y, groups=groups):
                test_distribution = self._get_distribution(y[curr_test_index])

                if len(test_distribution) == len(data_distribution):
                    diff = np.abs(test_distribution - data_distribution).sum()
                else:
                    diff = float("inf")

                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    train_index = curr_train_index
                    test_index = curr_test_index

            if flipped:
                train_index, test_index = test_index, train_index
            yield train_index, test_index


# https://github.com/scikit-learn/scikit-learn/blob/d3898d9d5/sklearn/model_selection/_split.py#L2757
@validate_params(
    {
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "random_state": ["random_state"],
        "shuffle": ["boolean"],
        "stratify": ["array-like", None],
        "groups": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def train_test_split(
    *arrays,
    test_size: None | Float = None,
    train_size: None | Float = None,
    random_state: np.random.RandomState | None | Int = None,
    shuffle: bool = True,
    stratify: None | ArrayLike = None,
    groups: None | ArrayLike = None,
) -> list:
    """Split arrays or matrices into random train and test subsets.

    This is an extended version of ``sklearn.model_selection.train_test_split`` that
    adds support for stratified splits with non-overlapping groups. When both
    ``stratify`` and ``groups`` are provided, uses ``StratifiedGroupShuffleSplit`` to
    ensure both class distributions and group separation are preserved.

    Parameters:
        *arrays: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas
            dataframes.
        test_size: If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the absolute
            number of test samples. If None, the value is set to the complement of the
            train size. If ``train_size`` is also None, it will be set to 0.25.
        train_size: If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the train split. If int, represents the
            absolute number of train samples. If None, the value is automatically set to
            the complement of the test size.
        random_state: Controls the randomness of the training and testing indices. Pass
            an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
        shuffle: Whether or not to shuffle the data before splitting. If False, stratify
            must be None.
        stratify: If not None, data is split in a stratified fashion, using this as the
            class labels. For binary or multiclass classification, this ensures that the
            test and training sets have approximately the same percentage of samples of
            each target class as the complete set.
        groups: Group labels for the samples used while splitting the dataset into train
            and test set. When provided with ``stratify``, ensures both stratification
            and non-overlapping groups are maintained.

    Returns:
        splitting: List containing train-test split of inputs. If ``shuffle=False``, the
            ``train`` arrays will have shape ``[0:split_point]`` and ``test`` arrays
            will have shape ``[split_point:n_samples]`` for each input.

    Examples:
        >>> import numpy as np
        >>> from ratiopath.model_selection import train_test_split
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([0, 0, 1, 1])
        >>> groups = np.array([1, 1, 2, 2])
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.25, random_state=42, stratify=y, groups=groups
        ... )
        >>> X_train
        array([[1, 2],
            [5, 6],
            [7, 8]])
        >>> X_test
        array([[3, 4]])

    Notes:
        When ``shuffle=True`` and both ``stratify`` and ``groups`` are provided, uses
        ``StratifiedGroupShuffleSplit`` to split the data, ensuring that:

        * The class distribution is preserved in train and test sets
        * No group appears in both train and test sets

        When only one of ``stratify`` or ``groups`` is provided, uses the appropriate
        single-constraint splitter.

        When ``shuffle=False``, a stratified split is not supported and ``stratify``
        must be None.
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        # Just this branch is different from sklearn's implementation
        if groups is not None:
            if stratify is not None:
                cvclass = StratifiedGroupShuffleSplit
            else:
                cvclass = GroupShuffleSplit
        else:
            cvclass = StratifiedShuffleSplit if stratify is not None else ShuffleSplit

        # It is safer to pass fractions, because some splitters calculate n_samplers
        # as number of groups, not samples
        cv = cvclass(
            test_size=n_test / n_samples,
            train_size=n_train / n_samples,
            random_state=random_state,
        )

        train, test = next(cv.split(X=arrays[0], y=stratify, groups=groups))

    xp, _, device = get_namespace_and_device(arrays[0])
    train, test = move_to(train, test, xp=xp, device=device)

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )
