from typing import Any


class BaseCompletion:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
