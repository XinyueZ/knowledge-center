from rich.pretty import pprint
from typing import Any

VERBOSE = True


def pretty_print(title: str = "Untitled", content: Any = None):
    if not VERBOSE:
        return
    print(f"-- {title} --")
    pprint(content)
