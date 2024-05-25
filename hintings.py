#!/usr/bin/env python3
from typing import Any, Callable, Iterable, Literal

# Simple type aliases
AnyType = Any
CallableType = Callable

# Complex type aliases
RowType = list[tuple[int, ...]]
FrameType = list[RowType]
FramesType = Iterable[FrameType]
Point3DType = tuple[float, float, float]
RotationType = tuple[float, float, float]
VertexType = tuple[float, float, float]
TriangleType = tuple[VertexType, VertexType, VertexType]

# Others
EffectModeType = Literal["ascii", "binary", "short", "standard", "long"]
