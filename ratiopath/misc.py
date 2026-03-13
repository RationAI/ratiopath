import inspect
from collections.abc import Callable
from typing import cast


def safely_instantiate[**P, T](
    cls: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    params = inspect.signature(cls).parameters

    # Check if the target accepts **kwargs natively
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    safe_kwargs = (
        kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in params}
    )
    return cast("Callable[..., T]", cls)(*args, **safe_kwargs)
