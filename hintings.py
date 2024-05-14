#!/usr/bin/env python3
from typing import Any, Callable, Iterable, Literal

# Simple type aliases
AnyType = Any
CallableType = Callable

# Complex type aliases
RowType = list[tuple[int, ...]]
FrameType = list[RowType]
FramesType = Iterable[FrameType]

# Others
EffectModeType = Literal["ascii", "binary", "short", "standard", "long"]
RenderModeType = Literal["frame", "ascii", "gray", "rgba"]
