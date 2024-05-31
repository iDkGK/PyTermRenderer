#!/usr/bin/env python3
from typing import Any, Callable, Iterable, Literal

# Simple type aliases
AnyType = Any
CallableType = Callable

# Complex type aliases
RowType = list[tuple[int, ...]]
FrameType = list[RowType]
FramesType = Iterable[FrameType]
ImageType = list[RowType]
ImagesType = Iterable[ImageType]
Point3DType = tuple[float, float, float]
RotationType = tuple[float, float]
Vertex3DType = tuple[float, float, float]
Triangle3DType = tuple[Vertex3DType, Vertex3DType, Vertex3DType]

# Others
EffectModeType = Literal["ascii", "binary", "short", "standard", "long"]
