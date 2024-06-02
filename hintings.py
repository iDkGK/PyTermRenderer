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
TriangleVerticesType = tuple[Vertex3DType, Vertex3DType, Vertex3DType]
Texture2DType = tuple[float, float]
TriangleTexturesType = tuple[Texture2DType, Texture2DType, Texture2DType]
Normal3DType = tuple[float, float, float]
TriangleNormalsType = tuple[Normal3DType, Normal3DType, Normal3DType]
FrustumBorderType = tuple[  # left right top bottom near far
    float, float, float, float, float, float
]
Vertex3DTexture2DNormal3DType = tuple[  # X Y Z U V X Y Z
    float, float, float, float, float, float, float, float
]
PixelCoordinateType = tuple[int, int]  # X Y on camera screen
PixelDataType = tuple[int, int, int, int, int]  # X Y R G B

# Others
EffectModeType = Literal["ascii", "binary", "short", "standard", "long"]
