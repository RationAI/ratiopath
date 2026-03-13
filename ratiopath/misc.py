import inspect
from collections.abc import Callable
from typing import cast


def safely_instantiate[**P, T](
    cls: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Instantiate a callable while filtering unsupported keyword arguments.

    Args:
        cls: Target callable/class to instantiate.
        *args: Positional arguments forwarded to the target.
        **kwargs: Keyword arguments; unsupported ones are dropped if the target
            does not accept ``**kwargs``.
    """
    params = inspect.signature(cls).parameters

    # Check if the target accepts **kwargs natively
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    safe_kwargs = (
        kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in params}
    )
    return cast("Callable[..., T]", cls)(*args, **safe_kwargs)
