import math
import sys
import time
import warnings
from pathlib import Path
from threading import Lock

from decoder import PNG
from hintings import (
    ScreenPoint2DType,
    FrameType,
    FrustumBorderType,
    ImageType,
    Normal3DType,
    ScreenPixelDataType,
    Point3DType,
    RotationType,
    RowType,
    Texture3DType,
    TriangleNormalsType,
    TriangleTexturesType,
    TriangleVerticesType,
    Vertex2DType,
    Vertex3DTexture3DNormal3DType,
    Vertex3DType,
)


# Render helper functions
def get_line_bresenham(
    vertex1: Vertex3DType,
    vertex2: Vertex3DType,
    frustum_border: FrustumBorderType,
) -> dict[ScreenPoint2DType, ScreenPixelDataType]:
    # Bresenham line algorithm
    line: dict[ScreenPoint2DType, ScreenPixelDataType] = {}
    left, right, top, bottom, near, far = frustum_border
    x1, y1, z1 = vertex1
    x2, y2, z2 = vertex2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    destination = (x2, y2)
    delta_x = abs(x2 - x1)
    delta_y = abs(y2 - y1)
    delta_z = abs(z2 - z1)
    steps = max(delta_x, delta_y)
    step_x = 1 if x1 < x2 else -1
    step_y = 1 if y1 < y2 else -1
    step_z = 0.0 if steps == 0.0 else (delta_z / steps if z1 < z2 else -delta_z / steps)
    error = delta_x - delta_y
    while True:
        interpolation = (x1, y1)
        if left < x1 < right and bottom < y1 < top and near < z1 < far:
            line[interpolation] = (255, 255, 255, 255, 9608)
        if interpolation == destination:
            break
        double_error = 2 * error
        if double_error > -delta_y:
            error -= delta_y
            x1 += step_x
        if double_error < delta_x:
            error += delta_x
            y1 += step_y
        z1 += step_z
    return line


def get_line_bresenham_xy(
    vertex1: Vertex2DType,
    vertex2: Vertex2DType,
    frustum_border: FrustumBorderType,
) -> dict[ScreenPoint2DType, ScreenPixelDataType]:
    # Bresenham line algorithm
    line: dict[ScreenPoint2DType, ScreenPixelDataType] = {}
    left, right, top, bottom, *_ = frustum_border
    x1, y1 = vertex1
    x2, y2 = vertex2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    destination = (x2, y2)
    delta_x = abs(x2 - x1)
    delta_y = abs(y2 - y1)
    step_x = 1 if x1 < x2 else -1
    step_y = 1 if y1 < y2 else -1
    error = delta_x - delta_y
    while True:
        interpolation = (x1, y1)
        if left < x1 < right and bottom < y1 < top:
            line[interpolation] = (255, 255, 255, 255, 9608)
        if interpolation == destination:
            break
        double_error = 2 * error
        if double_error > -delta_y:
            error -= delta_y
            x1 += step_x
        if double_error < delta_x:
            error += delta_x
            y1 += step_y
    return line


def get_line_sweepline(
    xsony_long: dict[int, list[int]],
    xsony_short: dict[int, list[int]],
    frustum_border: FrustumBorderType,
):
    sweep_line: dict[ScreenPoint2DType, ScreenPixelDataType] = {}
    for y_shared, xs_short in xsony_short.items():
        if y_shared not in xsony_long:
            continue
        xs_long_sorted = sorted(xsony_long[y_shared])
        x_long_min, x_long_max = xs_long_sorted[0], xs_long_sorted[-1]
        xs_short_sorted = sorted(xs_short)
        x_short_min, x_short_max = xs_short_sorted[0], xs_short_sorted[-1]
        if abs(x_long_min - x_short_max) < abs(x_long_max - x_short_min):
            sweep_line |= get_line_bresenham_xy(
                (x_long_min, y_shared), (x_short_max, y_shared), frustum_border
            )
        else:
            sweep_line |= get_line_bresenham_xy(
                (x_long_max, y_shared), (x_short_min, y_shared), frustum_border
            )
    return sweep_line


def get_textured_line_bresenham(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType,
    texture_size: tuple[float, float],
) -> dict[ScreenPoint2DType, ScreenPixelDataType]:
    # Bresenham line algorithm
    line: dict[ScreenPoint2DType, ScreenPixelDataType] = {}
    left, right, top, bottom, near, far = frustum_border
    x_a, y_a, z_a, u_a, v_a, *_ = vertex_texture_normal_a
    x_b, y_b, z_b, u_b, v_b, *_ = vertex_texture_normal_b
    x_c, y_c, _, u_c, v_c, *_ = vertex_texture_normal_c
    x_a, y_a = int(x_a), int(y_a)
    x_b, y_b = int(x_b), int(y_b)
    x_c, y_c = int(x_c), int(y_c)
    x, y, z = x_a, y_a, z_a
    texture_width, texture_height = texture_size
    destination = (x_b, y_b)
    delta_x = abs(x_b - x_a)
    delta_y = abs(y_b - y_a)
    delta_z = abs(z_b - z_a)
    steps = max(delta_x, delta_y)
    step_x = 1 if x_a < x_b else -1
    step_y = 1 if y_a < y_b else -1
    step_z = (
        0.0 if steps == 0.0 else (delta_z / steps if z_a < z_b else -delta_z / steps)
    )
    error = delta_x - delta_y
    while True:
        interpolation = (x, y)
        if left < x < right and bottom < y < top and near < z < far:
            denominator = (y_b - y_c) * (x_a - x_c) + (x_c - x_b) * (y_a - y_c)
            if denominator == 0.0:
                alpha = 0.0
                beta = 0.0
                gamma = 1.0
            else:
                alpha = ((y_b - y_c) * (x - x_c) + (x_c - x_b) * (y - y_c)) / (
                    denominator
                )
                beta = ((y_c - y_a) * (x - x_c) + (x_a - x_c) * (y - y_c)) / (
                    denominator
                )
                gamma = 1 - alpha - beta
            u = alpha * u_a + beta * u_b + gamma * u_c
            v = 1 - alpha * v_a - beta * v_b - gamma * v_c
            texture_x = min(max(0, int(u * texture_width)), int(texture_width))
            texture_y = min(max(0, int(v * texture_height)), int(texture_height))
            r, g, b, a = texture_image[texture_y][texture_x]
            line[interpolation] = (r, g, b, a, 9608)
        if interpolation == destination:
            break
        double_error = 2 * error
        if double_error > -delta_y:
            error -= delta_y
            x += step_x
        if double_error < delta_x:
            error += delta_x
            y += step_y
        z += step_z
    return line


def get_textured_line_bresenham_xy(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    vertex1: Vertex2DType,
    vertex2: Vertex2DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType,
    texture_size: tuple[float, float],
) -> dict[ScreenPoint2DType, ScreenPixelDataType]:
    # Bresenham line algorithm
    line: dict[ScreenPoint2DType, ScreenPixelDataType] = {}
    left, right, top, bottom, *_ = frustum_border
    x_a, y_a, _, u_a, v_a, *_ = vertex_texture_normal_a
    x_b, y_b, _, u_b, v_b, *_ = vertex_texture_normal_b
    x_c, y_c, _, u_c, v_c, *_ = vertex_texture_normal_c
    x1, y1 = vertex1
    x2, y2 = vertex2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x, y = x1, y1
    texture_width, texture_height = texture_size
    destination = (x2, y2)
    delta_x = abs(x2 - x1)
    delta_y = abs(y2 - y1)
    step_x = 1 if x1 < x2 else -1
    step_y = 1 if y1 < y2 else -1
    error = delta_x - delta_y
    while True:
        interpolation = (x, y)
        if left < x < right and bottom < y < top:
            denominator = (y_b - y_c) * (x_a - x_c) + (x_c - x_b) * (y_a - y_c)
            if denominator == 0.0:
                alpha = 0.0
                beta = 0.0
                gamma = 1.0
            else:
                alpha = ((y_b - y_c) * (x - x_c) + (x_c - x_b) * (y - y_c)) / (
                    denominator
                )
                beta = ((y_c - y_a) * (x - x_c) + (x_a - x_c) * (y - y_c)) / (
                    denominator
                )
                gamma = 1 - alpha - beta
            u = alpha * u_a + beta * u_b + gamma * u_c
            v = 1 - alpha * v_a - beta * v_b - gamma * v_c
            texture_x = min(max(0, int(u * texture_width)), int(texture_width))
            texture_y = min(max(0, int(v * texture_height)), int(texture_height))
            r, g, b, a = texture_image[texture_y][texture_x]
            line[interpolation] = (r, g, b, a, 9608)
        if interpolation == destination:
            break
        double_error = 2 * error
        if double_error > -delta_y:
            error -= delta_y
            x += step_x
        if double_error < delta_x:
            error += delta_x
            y += step_y
    return line


def get_textured_line_sweepline(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    xsony_long: dict[int, list[int]],
    xsony_short: dict[int, list[int]],
    frustum_border: FrustumBorderType,
    texture_image: ImageType,
    texture_size: tuple[float, float],
):
    sweep_line: dict[ScreenPoint2DType, ScreenPixelDataType] = {}
    for y_shared, xs_short in xsony_short.items():
        if y_shared not in xsony_long:
            continue
        xs_long_sorted = sorted(xsony_long[y_shared])
        x_long_min, x_long_max = xs_long_sorted[0], xs_long_sorted[-1]
        xs_short_sorted = sorted(xs_short)
        x_short_min, x_short_max = xs_short_sorted[0], xs_short_sorted[-1]
        if abs(x_long_min - x_short_max) < abs(x_long_max - x_short_min):
            sweep_line |= get_textured_line_bresenham_xy(
                vertex_texture_normal_a,
                vertex_texture_normal_b,
                vertex_texture_normal_c,
                (x_long_min, y_shared),
                (x_short_max, y_shared),
                frustum_border,
                texture_image,
                texture_size,
            )
        else:
            sweep_line |= get_textured_line_bresenham_xy(
                vertex_texture_normal_a,
                vertex_texture_normal_b,
                vertex_texture_normal_c,
                (x_long_max, y_shared),
                (x_short_min, y_shared),
                frustum_border,
                texture_image,
                texture_size,
            )
    return sweep_line


# Render functions
def render_mesh_line_no_culling(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType | None,
    texture_size: tuple[float, float] | None,
) -> dict[ScreenPoint2DType, ScreenPixelDataType]:
    x_a, y_a, z_a, *_ = vertex_texture_normal_a
    x_b, y_b, z_b, *_ = vertex_texture_normal_b
    x_c, y_c, z_c, *_ = vertex_texture_normal_c
    vertex_a = (x_a, y_a, z_a)
    vertex_b = (x_b, y_b, z_b)
    vertex_c = (x_c, y_c, z_c)
    return (
        get_line_bresenham(vertex_a, vertex_b, frustum_border)
        | get_line_bresenham(vertex_b, vertex_c, frustum_border)
        | get_line_bresenham(vertex_c, vertex_a, frustum_border)
    )


def render_mesh_line_backface_culling(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType | None,
    texture_size: tuple[float, float] | None,
) -> dict[ScreenPoint2DType, ScreenPixelDataType]:
    x_a, y_a, z_a, *_ = vertex_texture_normal_a
    x_b, y_b, z_b, *_ = vertex_texture_normal_b
    x_c, y_c, z_c, *_ = vertex_texture_normal_c
    v_ab_x, v_ab_y = x_b - x_a, y_b - y_a
    v_bc_x, v_bc_y = x_c - x_b, y_c - y_b
    vertex_a = (x_a, y_a, z_a)
    vertex_b = (x_b, y_b, z_b)
    vertex_c = (x_c, y_c, z_c)
    if v_ab_y * v_bc_x - v_ab_x * v_bc_y < 0:
        return {}
    return (
        get_line_bresenham(vertex_a, vertex_b, frustum_border)
        | get_line_bresenham(vertex_b, vertex_c, frustum_border)
        | get_line_bresenham(vertex_c, vertex_a, frustum_border)
    )


def render_untextured_model(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType | None,
    texture_size: tuple[float, float] | None,
) -> dict[ScreenPoint2DType, ScreenPixelDataType]:
    x_a, y_a, z_a, *_ = vertex_texture_normal_a
    x_b, y_b, z_b, *_ = vertex_texture_normal_b
    x_c, y_c, z_c, *_ = vertex_texture_normal_c
    v_ab_x, v_ab_y = x_b - x_a, y_b - y_a
    v_bc_x, v_bc_y = x_c - x_b, y_c - y_b
    if v_ab_y * v_bc_x - v_ab_x * v_bc_y < 0:
        return {}
    # Sort by y coordinate
    vertex_texture_normal_a, vertex_texture_normal_b, vertex_texture_normal_c = sorted(
        (vertex_texture_normal_a, vertex_texture_normal_b, vertex_texture_normal_c),
        key=lambda _: _[1],
        reverse=True,
    )
    x_a, y_a, z_a, *_ = vertex_texture_normal_a
    x_b, y_b, z_b, *_ = vertex_texture_normal_b
    x_c, y_c, z_c, *_ = vertex_texture_normal_c
    vertex_a = (x_a, y_a, z_a)
    vertex_b = (x_b, y_b, z_b)
    vertex_c = (x_c, y_c, z_c)
    # Longest line
    line_ac = get_line_bresenham(vertex_a, vertex_c, frustum_border)
    # Other lines
    line_ab = get_line_bresenham(vertex_a, vertex_b, frustum_border)
    line_bc = get_line_bresenham(vertex_b, vertex_c, frustum_border)
    # Longest line xs on y
    xsony_ac: dict[int, list[int]] = {}
    for x, y in line_ac:
        xsony_ac.setdefault(y, [])
        xsony_ac[y].append(x)
    # Other lines xs on y
    xsony_ab: dict[int, list[int]] = {}
    for x, y in line_ab:
        xsony_ab.setdefault(y, [])
        xsony_ab[y].append(x)
    xsony_bc: dict[int, list[int]] = {}
    for x, y in line_bc:
        xsony_bc.setdefault(y, [])
        xsony_bc[y].append(x)
    # Sweep line algorithm
    sweep_line_ab = get_line_sweepline(xsony_ac, xsony_ab, frustum_border)
    sweep_line_bc = get_line_sweepline(xsony_ac, xsony_bc, frustum_border)
    return line_ac | line_ab | line_bc | sweep_line_ab | sweep_line_bc


def render_textured_model(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType | None,
    texture_size: tuple[float, float] | None,
) -> dict[ScreenPoint2DType, ScreenPixelDataType]:
    if texture_image is None or texture_size is None:
        return {}
    x_a, y_a, *_ = vertex_texture_normal_a
    x_b, y_b, *_ = vertex_texture_normal_b
    x_c, y_c, *_ = vertex_texture_normal_c
    v_ab_x, v_ab_y = x_b - x_a, y_b - y_a
    v_bc_x, v_bc_y = x_c - x_b, y_c - y_b
    if v_ab_y * v_bc_x - v_ab_x * v_bc_y < 0:
        return {}
    # Sort by y coordinate
    vertex_texture_normal_a, vertex_texture_normal_b, vertex_texture_normal_c = sorted(
        (vertex_texture_normal_a, vertex_texture_normal_b, vertex_texture_normal_c),
        key=lambda _: _[1],
        reverse=True,
    )
    # Longest line
    line_ac = get_textured_line_bresenham(
        vertex_texture_normal_a,
        vertex_texture_normal_c,
        vertex_texture_normal_b,
        frustum_border,
        texture_image,
        texture_size,
    )
    # Other lines
    line_ab = get_textured_line_bresenham(
        vertex_texture_normal_a,
        vertex_texture_normal_b,
        vertex_texture_normal_c,
        frustum_border,
        texture_image,
        texture_size,
    )
    line_bc = get_textured_line_bresenham(
        vertex_texture_normal_b,
        vertex_texture_normal_c,
        vertex_texture_normal_a,
        frustum_border,
        texture_image,
        texture_size,
    )
    # Longest line xs on y
    xsony_ac: dict[int, list[int]] = {}
    for x, y in line_ac:
        xsony_ac.setdefault(y, [])
        xsony_ac[y].append(x)
    # Other lines xs on y
    xsony_ab: dict[int, list[int]] = {}
    for x, y in line_ab:
        xsony_ab.setdefault(y, [])
        xsony_ab[y].append(x)
    xsony_bc: dict[int, list[int]] = {}
    for x, y in line_bc:
        xsony_bc.setdefault(y, [])
        xsony_bc[y].append(x)
    # Sweep line algorithm
    sweep_line_ab = get_textured_line_sweepline(
        vertex_texture_normal_a,
        vertex_texture_normal_b,
        vertex_texture_normal_c,
        xsony_ac,
        xsony_ab,
        frustum_border,
        texture_image,
        texture_size,
    )
    sweep_line_bc = get_textured_line_sweepline(
        vertex_texture_normal_a,
        vertex_texture_normal_b,
        vertex_texture_normal_c,
        xsony_ac,
        xsony_bc,
        frustum_border,
        texture_image,
        texture_size,
    )
    return line_ac | line_ab | line_bc | sweep_line_ab | sweep_line_bc


class Object(object):
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._name = ""
        self._triangles: set[
            tuple[
                TriangleVerticesType,
                TriangleTexturesType,
                TriangleNormalsType,
            ]
        ] = set()
        self._texture: PNG | None = None
        # Parse file data and retrieve triangles vertices
        vertices: list[Vertex3DType] = []
        textures: list[Texture3DType] = []
        normals: list[Normal3DType] = []
        faces: list[tuple[tuple[int, ...], ...]] = []
        for line in Path(filepath).read_text().strip().splitlines():
            data_type, *data = line.strip().split()
            if data_type == "mtllib":
                pass
            elif data_type == "o":
                self._name = " ".join(data)
            elif data_type == "v":
                x, y, z, *_ = map(float, data)
                vertices.append((x, y, z))
            elif data_type == "vn":
                x, y, z, *_ = map(float, data)
                normals.append((x, y, z))
            elif data_type == "vt":
                u, *vw = map(float, data)
                if len(vw) == 2:
                    v, w = vw
                elif len(vw) == 1:
                    (v,), w = vw, 0.0
                else:
                    v = w = 0.0
                textures.append((u, v, w))
            elif data_type == "s":
                (data,) = data
                if data == "off":
                    group_number = None  # type: ignore
                else:
                    group_number = int(data)  # type: ignore
            elif data_type == "usemtl":
                model_filepath = Path(filepath)
                material_filepath = model_filepath.parent / (
                    "materials/%s.png" % model_filepath.stem
                )
                self._texture = PNG(material_filepath.as_posix()).decode()
            elif data_type == "f":
                face_vertices_indices: list[int] | tuple[int, ...] = []
                face_textures_indices: list[int] | tuple[int, ...] = []
                face_normals_indices: list[int] | tuple[int, ...] = []
                for part in data:
                    v_index, vt_index, vn_index = part.split("/")
                    face_vertices_indices.append(int(v_index) - 1)
                    face_textures_indices.append(int(vt_index) - 1)
                    face_normals_indices.append(int(vn_index) - 1)
                faces.append(
                    (
                        tuple(face_vertices_indices),
                        tuple(face_textures_indices),
                        tuple(face_normals_indices),
                    )
                )
        for face_vertices_indices, face_textures_indices, face_normals_indices in faces:
            face_vertex_a_index, face_vertex_b_index, face_vertex_c_index, *_ = (
                face_vertices_indices
            )
            face_texture_a_index, face_texture_b_index, face_texture_c_index, *_ = (
                face_textures_indices
            )
            face_normal_a_index, face_normal_b_index, face_normal_c_index, *_ = (
                face_normals_indices
            )
            self._triangles.add(
                (
                    (
                        vertices[face_vertex_a_index],
                        vertices[face_vertex_b_index],
                        vertices[face_vertex_c_index],
                    ),
                    (
                        textures[face_texture_a_index],
                        textures[face_texture_b_index],
                        textures[face_texture_c_index],
                    ),
                    (
                        normals[face_normal_a_index],
                        normals[face_normal_b_index],
                        normals[face_normal_c_index],
                    ),
                )
            )

    # Properties
    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def name(self) -> str:
        return self._name

    @property
    def triangles(self) -> set[
        tuple[
            TriangleVerticesType,
            TriangleTexturesType,
            TriangleNormalsType,
        ]
    ]:
        return self._triangles

    @property
    def texture(self) -> PNG | None:
        return self._texture

    # Update methods
    def update(self, delta_time: float = 0.0) -> None:
        triangles: set[
            tuple[
                TriangleVerticesType,
                TriangleTexturesType,
                TriangleNormalsType,
            ]
        ] = set()
        sin_θ = math.sin(math.radians(270 * delta_time))
        cos_θ = math.cos(math.radians(270 * delta_time))
        stroke = (time.perf_counter_ns() / 1e9 % 2.0 - 1.0) * 4 * delta_time
        for vertices, textures, normals in self._triangles:
            # Unpacking
            (
                (vertex_a_x, vertex_a_y, vertex_a_z),
                (vertex_b_x, vertex_b_y, vertex_b_z),
                (vertex_c_x, vertex_c_y, vertex_c_z),
            ) = vertices
            # Rotation
            (
                (vertex_a_x, vertex_a_y, vertex_a_z),
                (vertex_b_x, vertex_b_y, vertex_b_z),
                (vertex_c_x, vertex_c_y, vertex_c_z),
            ) = (
                (
                    vertex_a_x * cos_θ - vertex_a_z * sin_θ,
                    vertex_a_y,
                    vertex_a_x * sin_θ + vertex_a_z * cos_θ,
                ),
                (
                    vertex_b_x * cos_θ - vertex_b_z * sin_θ,
                    vertex_b_y,
                    vertex_b_x * sin_θ + vertex_b_z * cos_θ,
                ),
                (
                    vertex_c_x * cos_θ - vertex_c_z * sin_θ,
                    vertex_c_y,
                    vertex_c_x * sin_θ + vertex_c_z * cos_θ,
                ),
            )
            # Position
            (
                (vertex_a_x, vertex_a_y, vertex_a_z),
                (vertex_b_x, vertex_b_y, vertex_b_z),
                (vertex_c_x, vertex_c_y, vertex_c_z),
            ) = (
                (vertex_a_x, vertex_a_y + stroke, vertex_a_z),
                (vertex_b_x, vertex_b_y + stroke, vertex_b_z),
                (vertex_c_x, vertex_c_y + stroke, vertex_c_z),
            )
            # Repacking
            vertices = (
                (vertex_a_x, vertex_a_y, vertex_a_z),
                (vertex_b_x, vertex_b_y, vertex_b_z),
                (vertex_c_x, vertex_c_y, vertex_c_z),
            )
            triangles.add((vertices, textures, normals))
        self._triangles = triangles

    # Camera-related methods
    def show_to(self, camera: "Camera") -> None:
        camera.show_object(self)

    def hide_from(self, camera: "Camera") -> None:
        camera.hide_object(self)


class CameraScreenTooSmallError(Exception):
    pass


class Camera(object):
    def __init__(
        self,
        *,
        screen_size: tuple[int, int],
        field_of_view: float,
        near_plane: float,
        far_plane: float,
        coordinate: Point3DType,
        rotation: RotationType,
        move_speed: float = 5.0,
        dash_speed: float = 10.0,
        controllable: bool = True,
    ) -> None:
        self._screen_width, self._screen_height = screen_size
        self._screen_width = (self._screen_width // 2) * 2 or 2
        self._half_width, self._half_height = (
            self._screen_width // 2,
            self._screen_height // 2,
        )
        if self._screen_height < 0:
            raise CameraScreenTooSmallError(
                "camera screen is too small to render objects."
            )
        self._field_of_view = field_of_view
        self._focal = (
            max(self._screen_width, self._screen_height)
            / math.tan(math.radians(field_of_view / 2.0))
            / 2
        )
        self._near_plane = max(0, near_plane)
        self._far_plane = max(near_plane, far_plane)
        self._frustum_border = (
            -self._half_width,
            self._screen_width - self._half_width,
            self._screen_height - self._half_height,
            -self._half_height,
            self._near_plane,
            self._far_plane,
        )
        self._x, self._y, self._z = coordinate
        self._original_coordinate = coordinate
        self._yaw, self._pitch = rotation
        self._original_rotation = rotation
        self._move_speed = abs(move_speed)
        self._dash_speed = abs(dash_speed)
        self._controllable = controllable
        self._objects: set[Object] = set()
        self._pixels: dict[ScreenPoint2DType, ScreenPixelDataType] = {}
        self._information: list[str] = []
        self._delta_time = 0.0
        # Register controller
        if not self._controllable:
            return
        from controller import KeyboardListener

        try:
            import keyboard

            keyboard.press("enter")
        except ImportError:
            keyboard = None
            warnings.warn(
                "no support for third-party keyboard library. "
                "If it's on Linux, install it via `pip install keyboard`. "
                "Then running the `main.py` as root. "
            )
            warnings.warn("Using custom KeyboardListener for camera controlling.")
        try:
            import mouse  # type: ignore

            mouse.move(self._screen_width, self._screen_height)  # type: ignore
        except ImportError:
            mouse = None
            warnings.warn(
                "no support for third-party mouse library. "
                "If it's on Linux, install it via `pip install mouse`. "
                "Then running the `main.py` as root. "
            )
            warnings.warn("Using custom KeyboardListener for camera controlling.")
        # keyboard = mouse = None
        if keyboard is None or mouse is None:
            keyboard_listener = KeyboardListener()
        else:
            keyboard_listener = None

        # Quit and reset state
        self._quit_state = False
        self._reset_state = False

        # Miscellaneous
        self._display_information_state = False
        render_modes = (
            "Mesh Line (No Culling)",
            "Mesh Line (Back-face Culling)",
            "Model without Texture",
            "Model with Texture",
        )
        render_functions = (
            render_mesh_line_no_culling,
            render_mesh_line_backface_culling,
            render_untextured_model,
            render_textured_model,
        )
        render_mode_index = 0
        self._selected_render_mode = render_modes[render_mode_index]
        self._selected_render_function = render_functions[render_mode_index]

        # With third-party modules
        # Move state
        self._dash_state = False
        self._move_forward_state = False
        self._move_backward_state = False
        self._move_leftward_state = False
        self._move_rightward_state = False
        self._move_upward_state = False
        self._move_downward_state = False
        # Keyboard
        if keyboard is not None:
            from keyboard import KEY_DOWN, KEY_UP, KeyboardEvent

            active_states = {KEY_DOWN: True, KEY_UP: False, None: False}
            time_counter = time.perf_counter_ns()

            def quit(event: KeyboardEvent) -> None:
                self._quit_state = active_states.get(event.event_type, False)

            def reset(event: KeyboardEvent) -> None:
                self._reset_state = active_states.get(event.event_type, False)

            def switch_display_information(event: KeyboardEvent) -> None:
                nonlocal time_counter
                # Timeout using millisecond
                if (time.perf_counter_ns() - time_counter) / 1e6 > 200:
                    time_counter = time.perf_counter_ns()
                    self._display_information_state = (
                        not self._display_information_state
                    )

            def change_render_mode(event: KeyboardEvent) -> None:
                nonlocal time_counter, render_mode_index
                # Timeout using millisecond
                if (time.perf_counter_ns() - time_counter) / 1e6 > 200:
                    time_counter = time.perf_counter_ns()
                    render_mode_index = (render_mode_index + 1) % len(render_functions)
                    self._selected_render_mode = render_modes[render_mode_index]
                    self._selected_render_function = render_functions[render_mode_index]

            def dash(event: KeyboardEvent) -> None:
                self._dash_state = active_states.get(event.event_type, False)

            def move_forward(event: KeyboardEvent) -> None:
                self._move_forward_state = active_states.get(event.event_type, False)

            def move_backward(event: KeyboardEvent) -> None:
                self._move_backward_state = active_states.get(event.event_type, False)

            def move_leftward(event: KeyboardEvent) -> None:
                self._move_leftward_state = active_states.get(event.event_type, False)

            def move_rightward(event: KeyboardEvent) -> None:
                self._move_rightward_state = active_states.get(event.event_type, False)

            def move_upward(event: KeyboardEvent) -> None:
                self._move_upward_state = active_states.get(event.event_type, False)

            def move_downward(event: KeyboardEvent) -> None:
                self._move_downward_state = active_states.get(event.event_type, False)

            # Quiting & resetting
            keyboard.hook_key("escape", quit)
            keyboard.hook_key("r", reset)
            # Miscellaneous
            keyboard.hook_key("i", switch_display_information)
            keyboard.hook_key("p", change_render_mode)
            # Position
            keyboard.hook_key("shift", dash)
            keyboard.hook_key("w", move_forward)
            keyboard.hook_key("s", move_backward)
            keyboard.hook_key("a", move_leftward)
            keyboard.hook_key("d", move_rightward)
            keyboard.hook_key("space", move_upward)
            keyboard.hook_key("ctrl", move_downward)
        # Mouse
        if mouse is not None:
            from mouse import ButtonEvent, MoveEvent, WheelEvent  # type: ignore

            rotate_lock = Lock()
            mouse.move(self._screen_width, self._screen_height)  # type: ignore

            def rotate(event: ButtonEvent | WheelEvent | MoveEvent) -> None:
                if not rotate_lock.acquire(blocking=False):
                    return
                while True:
                    if mouse.get_position() != (self._screen_width, self._screen_height):  # type: ignore
                        mouse.move(self._screen_width, self._screen_height)  # type: ignore
                    else:
                        break
                if type(event) == MoveEvent:
                    self._rotate(
                        yaw=-(event.y - self._screen_height) / 72,  # type: ignore
                        pitch=+(event.x - self._screen_width) / 72,  # type: ignore
                    )
                rotate_lock.release()

            # Rotation
            mouse.hook(rotate)  # type: ignore
        # With custom `KeyboardListener` as fallback
        if keyboard_listener is not None:

            # Quit and reset state
            def stop_and_quit_legacy() -> None:
                keyboard_listener.stop()
                self._quit_state = True

            # Miscellaneous state
            def switch_display_information_legacy() -> None:
                self._display_information_state = not self._display_information_state

            def change_render_mode_legacy() -> None:
                nonlocal render_mode_index
                render_mode_index = (render_mode_index + 1) % len(render_functions)
                self._selected_render_mode = render_modes[render_mode_index]
                self._selected_render_function = render_functions[render_mode_index]

            # Quiting & resetting
            keyboard_listener.register("\x1b", stop_and_quit_legacy)
            keyboard_listener.register("r", self._reset)
            # Miscellaneous
            keyboard_listener.register("i", switch_display_information_legacy)
            keyboard_listener.register("p", change_render_mode_legacy)
            # Position
            keyboard_listener.register("W", self._dash_forward)
            keyboard_listener.register("w", self._move_forward)
            keyboard_listener.register("S", self._dash_backward)
            keyboard_listener.register("s", self._move_backward)
            keyboard_listener.register("A", self._dash_leftward)
            keyboard_listener.register("a", self._move_leftward)
            keyboard_listener.register("D", self._dash_rightward)
            keyboard_listener.register("d", self._move_rightward)
            keyboard_listener.register(" ", self._move_upward)
            keyboard_listener.register("\r", self._move_downward)
            # Rotation
            keyboard_listener.register("8", self._yaw_forward)
            keyboard_listener.register("2", self._yaw_reverse)
            keyboard_listener.register("6", self._pitch_forward)
            keyboard_listener.register("4", self._pitch_reverse)

    # Reset methods
    def _reset(self) -> None:
        self._x, self._y, self._z = self._original_coordinate
        self._yaw, self._pitch = self._original_rotation

    # Move methods
    def _move_forward(self) -> None:
        self._x += self._move_speed * self._vector_x * self._delta_time
        self._z += self._move_speed * self._vector_z * self._delta_time

    def _move_backward(self) -> None:
        self._x -= self._move_speed * self._vector_x * self._delta_time
        self._z -= self._move_speed * self._vector_z * self._delta_time

    def _move_leftward(self) -> None:
        self._x -= self._move_speed * self._vector_z * self._delta_time
        self._z += self._move_speed * self._vector_x * self._delta_time

    def _move_rightward(self) -> None:
        self._x += self._move_speed * self._vector_z * self._delta_time
        self._z -= self._move_speed * self._vector_x * self._delta_time

    def _move_upward(self) -> None:
        self._y += self._move_speed * self._delta_time

    def _move_downward(self) -> None:
        self._y -= self._move_speed * self._delta_time

    def _dash_forward(self) -> None:
        self._x += self._dash_speed * self._vector_x * self._delta_time
        self._z += self._dash_speed * self._vector_z * self._delta_time

    def _dash_backward(self) -> None:
        self._x -= self._dash_speed * self._vector_x * self._delta_time
        self._z -= self._dash_speed * self._vector_z * self._delta_time

    def _dash_leftward(self) -> None:
        self._x -= self._dash_speed * self._vector_z * self._delta_time
        self._z += self._dash_speed * self._vector_x * self._delta_time

    def _dash_rightward(self) -> None:
        self._x += self._dash_speed * self._vector_z * self._delta_time
        self._z -= self._dash_speed * self._vector_x * self._delta_time

    def _dash_upward(self) -> None:
        self._y += self._dash_speed * self._delta_time

    def _dash_downward(self) -> None:
        self._y -= self._dash_speed * self._delta_time

    # Rotate methods
    def _yaw_forward(self) -> None:
        self._yaw += 1.0
        self._yaw = max(min(self._yaw, 90.0), -90.0)

    def _yaw_reverse(self) -> None:
        self._yaw -= 1.0
        self._yaw = max(min(self._yaw, 90.0), -90.0)

    def _pitch_forward(self) -> None:
        self._pitch += 1.0
        self._pitch %= 360.0

    def _pitch_reverse(self) -> None:
        self._pitch -= 1.0
        self._pitch %= 360.0

    def _rotate(
        self,
        *,
        yaw: float = 0.0,
        pitch: float = 0.0,
    ) -> None:
        self._yaw += yaw
        self._pitch += pitch
        self._yaw = max(min(self._yaw, 90.0), -90.0)
        self._pitch %= 360.0

    # Update methods
    def _update_position(self, delta_time: float) -> None:
        if not self._controllable:
            return
        self._delta_time = delta_time
        if self._quit_state:
            sys.exit(0)
        if self._reset_state:
            self._reset()
            return
        if self._move_forward_state:
            if self._dash_state:
                self._dash_forward()
            else:
                self._move_forward()
        if self._move_backward_state:
            if self._dash_state:
                self._dash_backward()
            else:
                self._move_backward()
        if self._move_leftward_state:
            if self._dash_state:
                self._dash_leftward()
            else:
                self._move_leftward()
        if self._move_rightward_state:
            if self._dash_state:
                self._dash_rightward()
            else:
                self._move_rightward()
        if self._move_upward_state:
            if self._dash_state:
                self._dash_upward()
            else:
                self._move_upward()
        if self._move_downward_state:
            if self._dash_state:
                self._dash_downward()
            else:
                self._move_downward()

    def _update_trigonometrics(self, delta_time: float) -> None:
        yaw_radians = math.radians(-self._yaw)
        pitch_radians = math.radians(self._pitch)
        self._sin_yaw = math.sin(yaw_radians)
        self._cos_yaw = math.cos(yaw_radians)
        self._sin_pitch = math.sin(pitch_radians)
        self._cos_pitch = math.cos(pitch_radians)

    def _update_vector(self, delta_time: float) -> None:
        self._vector_x = self._sin_pitch  # X component
        self._vector_y = 0.0  # Y component
        self._vector_z = self._cos_pitch  # Z component

    def _update_objects(self, delta_time: float) -> None:
        # Iteration over all triangles. Assuming that every triangle is ▲abc
        self._pixels: dict[ScreenPoint2DType, ScreenPixelDataType] = {}
        for obj in self._objects:
            if obj.texture is None:
                texture_image = None
                texture_size = None
            else:
                texture_image = obj.texture.image_data
                texture_width, texture_height = obj.texture.image_size
                texture_size = (texture_width - 1e-8, texture_height - 1e-8)
            for vertices, textures, normals in obj.triangles:
                (
                    (vertex_a_x, vertex_a_y, vertex_a_z),
                    (vertex_b_x, vertex_b_y, vertex_b_z),
                    (vertex_c_x, vertex_c_y, vertex_c_z),
                ) = vertices
                (
                    (texture_a_u, texture_a_v, texture_a_w),
                    (texture_b_u, texture_b_v, texture_b_w),
                    (texture_c_u, texture_c_v, texture_c_w),
                ) = textures
                (
                    (normal_a_x, normal_a_y, normal_a_z),
                    (normal_b_x, normal_b_y, normal_b_z),
                    (normal_c_x, normal_c_y, normal_c_z),
                ) = normals
                # Position
                # Using vector for relative position
                (vertex_a_x, vertex_a_y, vertex_a_z) = (
                    vertex_a_x - self._x,
                    vertex_a_y - self._y,
                    vertex_a_z - self._z,
                )
                (vertex_b_x, vertex_b_y, vertex_b_z) = (
                    vertex_b_x - self._x,
                    vertex_b_y - self._y,
                    vertex_b_z - self._z,
                )
                (vertex_c_x, vertex_c_y, vertex_c_z) = (
                    vertex_c_x - self._x,
                    vertex_c_y - self._y,
                    vertex_c_z - self._z,
                )
                # Rotation
                # Y-axis rotation that affects X/Z coordinates
                vertex_a_x, vertex_a_z = (
                    vertex_a_x * self._cos_pitch - vertex_a_z * self._sin_pitch,
                    vertex_a_x * self._sin_pitch + vertex_a_z * self._cos_pitch,
                )
                vertex_b_x, vertex_b_z = (
                    vertex_b_x * self._cos_pitch - vertex_b_z * self._sin_pitch,
                    vertex_b_x * self._sin_pitch + vertex_b_z * self._cos_pitch,
                )
                vertex_c_x, vertex_c_z = (
                    vertex_c_x * self._cos_pitch - vertex_c_z * self._sin_pitch,
                    vertex_c_x * self._sin_pitch + vertex_c_z * self._cos_pitch,
                )
                # X-axis rotation that affects Y/Z coordinates
                vertex_a_y, vertex_a_z = (
                    vertex_a_y * self._cos_yaw + vertex_a_z * self._sin_yaw,
                    -vertex_a_y * self._sin_yaw + vertex_a_z * self._cos_yaw,
                )
                vertex_b_y, vertex_b_z = (
                    vertex_b_y * self._cos_yaw + vertex_b_z * self._sin_yaw,
                    -vertex_b_y * self._sin_yaw + vertex_b_z * self._cos_yaw,
                )
                vertex_c_y, vertex_c_z = (
                    vertex_c_y * self._cos_yaw + vertex_c_z * self._sin_yaw,
                    -vertex_c_y * self._sin_yaw + vertex_c_z * self._cos_yaw,
                )
                # Simple near/far plane culling
                # TODO: implement advanced near/far plane culling
                if not (
                    vertex_a_z > self._near_plane
                    and vertex_b_z > self._near_plane
                    and vertex_c_z > self._near_plane
                ) or (
                    vertex_a_z > self._far_plane
                    and vertex_b_z > self._far_plane
                    and vertex_c_z > self._far_plane
                ):
                    continue
                # Triangle vertices projected on camera screen
                vertex_texture_normal_a = (
                    self._focal * vertex_a_x / vertex_a_z,
                    self._focal * vertex_a_y / vertex_a_z,
                    vertex_a_z,
                    texture_a_u,
                    texture_a_v,
                    texture_a_w,
                    normal_a_x,
                    normal_a_y,
                    normal_a_z,
                )
                vertex_texture_normal_b = (
                    self._focal * vertex_b_x / vertex_b_z,
                    self._focal * vertex_b_y / vertex_b_z,
                    vertex_b_z,
                    texture_b_u,
                    texture_b_v,
                    texture_b_w,
                    normal_b_x,
                    normal_b_y,
                    normal_b_z,
                )
                vertex_texture_normal_c = (
                    self._focal * vertex_c_x / vertex_c_z,
                    self._focal * vertex_c_y / vertex_c_z,
                    vertex_c_z,
                    texture_c_u,
                    texture_c_v,
                    texture_c_w,
                    normal_c_x,
                    normal_c_y,
                    normal_c_z,
                )
                self._pixels |= self._selected_render_function(
                    vertex_texture_normal_a,
                    vertex_texture_normal_b,
                    vertex_texture_normal_c,
                    self._frustum_border,
                    texture_image,
                    texture_size,
                )

    def _update_infomation(self, delta_time: float) -> None:
        if self._display_information_state:
            self._information = [
                *(("".rjust(self._screen_width),) * (self._screen_height - 4)),
                ("FOV: %f" % self._field_of_view).rjust(self._screen_width),
                ("Rendering Mode: %s" % self._selected_render_mode).rjust(
                    self._screen_width
                ),
                ("Rotation (Yaw, Pitch): (%f, %f)" % (self._yaw, self._pitch)).rjust(
                    self._screen_width
                ),
                (
                    "Coordinate (X, Y, Z): (%f, %f, %f)" % (self._x, self._y, self._z)
                ).rjust(self._screen_width),
            ]
        else:
            self._information = [
                *(("".rjust(self._screen_width),) * (self._screen_height)),
            ]

    def update(self, delta_time: float = 0.0) -> None:
        self._update_position(delta_time)
        self._update_trigonometrics(delta_time)
        self._update_vector(delta_time)
        self._update_objects(delta_time)
        self._update_infomation(delta_time)

    # Draw methods
    def get_frame(self) -> FrameType:
        frame: FrameType = []
        for y in range(0, self._screen_height):
            row: RowType = []
            for x in range(0, self._screen_width):
                character = self._information[y][x]
                if character != " ":
                    pixel = (255, 255, 255, 255, ord(character))
                else:
                    pixel = self._pixels.get(
                        ((x - self._half_width) // 2, y - self._half_height),
                        (255, 255, 255, 255, 32),
                    )
                row.append(pixel)
            frame.append(row)
        frame.reverse()
        return frame

    # Objects-related methods
    def show_object(self, obj: Object) -> None:
        if obj not in self._objects:
            self._objects.add(obj)

    def hide_object(self, obj: Object) -> None:
        if obj in self._objects:
            self._objects.remove(obj)


class SmoothCamera(Camera):
    def __init__(
        self,
        *,
        screen_size: tuple[int, int],
        field_of_view: float,
        near_plane: float,
        far_plane: float,
        coordinate: Point3DType,
        rotation: RotationType,
        move_acceleration: float = 10.0,
        dash_acceleration: float = 20.0,
        inertia_ratio: float = 0.8,
        controllable: bool = True,
    ) -> None:
        self._speed_x = 0.0
        self._speed_y = 0.0
        self._speed_z = 0.0
        self._move_acceleration = abs(move_acceleration)
        self._dash_acceleration = abs(dash_acceleration)
        self._inertia_ratio = max(min(inertia_ratio, 1.0), 0.0) / 200.0
        super(SmoothCamera, self).__init__(
            screen_size=screen_size,
            field_of_view=field_of_view,
            near_plane=near_plane,
            far_plane=far_plane,
            coordinate=coordinate,
            rotation=rotation,
            move_speed=0.0,
            dash_speed=0.0,
            controllable=controllable,
        )

    # Move methods
    def _move_forward(self) -> None:
        self._speed_x += self._move_acceleration * self._vector_x * self._delta_time
        self._speed_z += self._move_acceleration * self._vector_z * self._delta_time

    def _move_backward(self) -> None:
        self._speed_x -= self._move_acceleration * self._vector_x * self._delta_time
        self._speed_z -= self._move_acceleration * self._vector_z * self._delta_time

    def _move_leftward(self) -> None:
        self._speed_x -= self._move_acceleration * self._vector_z * self._delta_time
        self._speed_z += self._move_acceleration * self._vector_x * self._delta_time

    def _move_rightward(self) -> None:
        self._speed_x += self._move_acceleration * self._vector_z * self._delta_time
        self._speed_z -= self._move_acceleration * self._vector_x * self._delta_time

    def _move_upward(self) -> None:
        self._speed_y += self._move_acceleration * self._delta_time

    def _move_downward(self) -> None:
        self._speed_y -= self._move_acceleration * self._delta_time

    def _dash_forward(self) -> None:
        self._speed_x += self._dash_acceleration * self._vector_x * self._delta_time
        self._speed_z += self._dash_acceleration * self._vector_z * self._delta_time

    def _dash_backward(self) -> None:
        self._speed_x -= self._dash_acceleration * self._vector_x * self._delta_time
        self._speed_z -= self._dash_acceleration * self._vector_z * self._delta_time

    def _dash_leftward(self) -> None:
        self._speed_x -= self._dash_acceleration * self._vector_z * self._delta_time
        self._speed_z += self._dash_acceleration * self._vector_x * self._delta_time

    def _dash_rightward(self) -> None:
        self._speed_x += self._dash_acceleration * self._vector_z * self._delta_time
        self._speed_z -= self._dash_acceleration * self._vector_x * self._delta_time

    def _dash_upward(self) -> None:
        self._speed_y += self._dash_acceleration * self._delta_time

    def _dash_downward(self) -> None:
        self._speed_y -= self._dash_acceleration * self._delta_time

    # Update methods
    def update(self, delta_time: float = 0.0) -> None:
        super(SmoothCamera, self).update(delta_time)
        self._x += self._speed_x * self._delta_time
        self._y += self._speed_y * self._delta_time
        self._z += self._speed_z * self._delta_time
        self._speed_x *= self._inertia_ratio**self._delta_time
        self._speed_y *= self._inertia_ratio**self._delta_time
        self._speed_z *= self._inertia_ratio**self._delta_time

    # Reset methods
    def _reset(self) -> None:
        super(SmoothCamera, self)._reset()
        self._speed_x = 0.0
        self._speed_y = 0.0
        self._speed_z = 0.0
