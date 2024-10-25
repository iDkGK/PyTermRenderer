import math

from hinting import (
    FrustumBorderType,
    ImageType,
    ScreenPixelDataType,
    ScreenPoint2DType,
    Vertex3DTexture3DNormal3DType,
    Vertex3DType,
)

# Constant variables
FILLED_PIXEL = ord("█")
EMPTY_PIXEL = ord(" ")
DEFAULT_PIXEL_DATA = (math.inf, 255, 255, 255, 255, FILLED_PIXEL)


# Render helper functions
def get_untextured_line_bresenham(
    vertex1: Vertex3DType,
    vertex2: Vertex3DType,
    frustum_border: FrustumBorderType,
    pixel_buffer: dict[ScreenPoint2DType, ScreenPixelDataType],
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
            z_buffered, *_ = pixel_buffer.get(interpolation, DEFAULT_PIXEL_DATA)
            pixel_data = (z1, 255, 255, 255, 255, FILLED_PIXEL)
            if z_buffered > z1:
                pixel_buffer[interpolation] = line[interpolation] = pixel_data
            else:
                line[interpolation] = pixel_data
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


def get_untextured_line_sweepline(
    xzsony_long: dict[int, list[tuple[int, float]]],
    xzsony_short: dict[int, list[tuple[int, float]]],
    frustum_border: FrustumBorderType,
    pixel_buffer: dict[ScreenPoint2DType, ScreenPixelDataType],
) -> None:
    for y_shared, xzs_short in xzsony_short.items():
        if y_shared not in xzsony_long:
            continue
        xzs_long_sorted = sorted(xzsony_long[y_shared], key=lambda xzs: xzs[0])
        (x_long_min, z_long_min), (x_long_max, z_long_max) = (
            xzs_long_sorted[0],
            xzs_long_sorted[-1],
        )
        xzs_short_sorted = sorted(xzs_short, key=lambda xzs: xzs[0])
        (x_short_min, z_short_min), (x_short_max, z_short_max) = (
            xzs_short_sorted[0],
            xzs_short_sorted[-1],
        )
        if abs(x_long_min - x_short_max) < abs(x_long_max - x_short_min):
            get_untextured_line_bresenham(
                (x_long_min, y_shared, z_long_min),
                (x_short_max, y_shared, z_short_max),
                frustum_border,
                pixel_buffer,
            )
        else:
            get_untextured_line_bresenham(
                (x_long_max, y_shared, z_long_max),
                (x_short_min, y_shared, z_short_min),
                frustum_border,
                pixel_buffer,
            )


def get_textured_line_bresenham_margin(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType,
    texture_size: tuple[int, int],
    pixel_buffer: dict[ScreenPoint2DType, ScreenPixelDataType],
) -> dict[ScreenPoint2DType, ScreenPixelDataType]:
    # Bresenham line algorithm
    line: dict[ScreenPoint2DType, ScreenPixelDataType] = {}
    left, right, top, bottom, near, far = frustum_border
    x_a, y_a, z_a, u_a, v_a, *_ = vertex_texture_normal_a
    x_b, y_b, z_b, u_b, v_b, *_ = vertex_texture_normal_b
    x_c, y_c, z_c, u_c, v_c, *_ = vertex_texture_normal_c
    w_a = 1 / z_a if z_a != 0 else 0
    w_b = 1 / z_b if z_b != 0 else 0
    w_c = 1 / z_c if z_c != 0 else 0
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
    denominator = (y_b - y_c) * (x_a - x_c) + (x_c - x_b) * (y_a - y_c)
    reciprocal_denominator = 1 / denominator if denominator != 0.0 else math.inf
    while True:
        interpolation = (x, y)
        if left < x < right and bottom < y < top and near < z < far:
            alpha, beta, gamma, u, v, w = 0.5, 0.5, 0.0, 1.0, 0.0, 0.0
            if denominator != 0.0:
                alpha = (
                    (y_b - y_c) * (x - x_c) + (x_c - x_b) * (y - y_c)
                ) * reciprocal_denominator
                beta = (
                    (y_c - y_a) * (x - x_c) + (x_a - x_c) * (y - y_c)
                ) * reciprocal_denominator
                gamma = 1 - alpha - beta
                w = alpha * w_a + beta * w_b + gamma * w_c
                u, v = (
                    (alpha * u_a * w_a + beta * u_b * w_b + gamma * u_c * w_c) / w,
                    1 - (alpha * v_a * w_a + beta * v_b * w_b + gamma * v_c * w_c) / w,
                )
            texture_x = min(max(0, round(u * texture_width)), texture_width)
            texture_y = min(max(0, round(v * texture_height)), texture_height)
            z_buffered, *_ = pixel_buffer.get(interpolation, DEFAULT_PIXEL_DATA)
            r, g, b, a = texture_image[texture_y][texture_x]
            pixel_data = (z, r, g, b, a, FILLED_PIXEL)
            if z_buffered > z:
                pixel_buffer[interpolation] = line[interpolation] = pixel_data
            else:
                line[interpolation] = pixel_data
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


def get_textured_line_bresenham_padding(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    vertex1: Vertex3DType,
    vertex2: Vertex3DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType,
    texture_size: tuple[int, int],
    pixel_buffer: dict[ScreenPoint2DType, ScreenPixelDataType],
) -> None:
    # Bresenham line algorithm
    line: dict[ScreenPoint2DType, ScreenPixelDataType] = {}
    left, right, top, bottom, near, far = frustum_border
    x_a, y_a, z_a, u_a, v_a, *_ = vertex_texture_normal_a
    x_b, y_b, z_b, u_b, v_b, *_ = vertex_texture_normal_b
    x_c, y_c, z_c, u_c, v_c, *_ = vertex_texture_normal_c
    w_a = 1 / z_a if z_a != 0 else 0
    w_b = 1 / z_b if z_b != 0 else 0
    w_c = 1 / z_c if z_c != 0 else 0
    x1, y1, z1 = vertex1
    x2, y2, z2 = vertex2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x, y, z = x1, y1, z1
    texture_width, texture_height = texture_size
    destination = (x2, y2)
    delta_x = abs(x2 - x1)
    delta_y = abs(y2 - y1)
    delta_z = abs(z2 - z1)
    steps = max(delta_x, delta_y)
    step_x = 1 if x1 < x2 else -1
    step_y = 1 if y1 < y2 else -1
    step_z = 0.0 if steps == 0.0 else (delta_z / steps if z1 < z2 else -delta_z / steps)
    error = delta_x - delta_y
    denominator = (y_b - y_c) * (x_a - x_c) + (x_c - x_b) * (y_a - y_c)
    reciprocal_denominator = 1 / denominator if denominator != 0.0 else math.inf
    while True:
        interpolation = (x, y)
        if left < x < right and bottom < y < top and near < z < far:
            alpha, beta, gamma, u, v, w = 0.5, 0.5, 0.0, 1.0, 0.0, 0.0
            if denominator != 0.0:
                alpha = (
                    (y_b - y_c) * (x - x_c) + (x_c - x_b) * (y - y_c)
                ) * reciprocal_denominator
                beta = (
                    (y_c - y_a) * (x - x_c) + (x_a - x_c) * (y - y_c)
                ) * reciprocal_denominator
                gamma = 1 - alpha - beta
                w = alpha * w_a + beta * w_b + gamma * w_c
                u, v = (
                    (alpha * u_a * w_a + beta * u_b * w_b + gamma * u_c * w_c) / w,
                    1 - (alpha * v_a * w_a + beta * v_b * w_b + gamma * v_c * w_c) / w,
                )
            texture_x = min(max(0, round(u * texture_width)), texture_width)
            texture_y = min(max(0, round(v * texture_height)), texture_height)
            z_buffered, *_ = pixel_buffer.get(interpolation, DEFAULT_PIXEL_DATA)
            r, g, b, a = texture_image[texture_y][texture_x]
            pixel_data = (z, r, g, b, a, FILLED_PIXEL)
            if z_buffered > z:
                pixel_buffer[interpolation] = line[interpolation] = pixel_data
            else:
                line[interpolation] = pixel_data
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


def get_textured_line_sweepline(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    xzsony_long: dict[int, list[tuple[int, float]]],
    xzsony_short: dict[int, list[tuple[int, float]]],
    frustum_border: FrustumBorderType,
    texture_image: ImageType,
    texture_size: tuple[int, int],
    pixel_buffer: dict[ScreenPoint2DType, ScreenPixelDataType],
) -> None:
    for y_shared, xzs_short in xzsony_short.items():
        if y_shared not in xzsony_long:
            continue
        xzs_long_sorted = sorted(xzsony_long[y_shared], key=lambda xzs: xzs[0])
        (x_long_min, z_long_min), (x_long_max, z_long_max) = (
            xzs_long_sorted[0],
            xzs_long_sorted[-1],
        )
        xzs_short_sorted = sorted(xzs_short, key=lambda xzs: xzs[0])
        (x_short_min, z_short_min), (x_short_max, z_short_max) = (
            xzs_short_sorted[0],
            xzs_short_sorted[-1],
        )
        if abs(x_long_min - x_short_max) < abs(x_long_max - x_short_min):
            get_textured_line_bresenham_padding(
                vertex_texture_normal_a,
                vertex_texture_normal_b,
                vertex_texture_normal_c,
                (x_long_min, y_shared, z_long_min),
                (x_short_max, y_shared, z_short_max),
                frustum_border,
                texture_image,
                texture_size,
                pixel_buffer,
            )
        else:
            get_textured_line_bresenham_padding(
                vertex_texture_normal_a,
                vertex_texture_normal_b,
                vertex_texture_normal_c,
                (x_long_max, y_shared, z_long_max),
                (x_short_min, y_shared, z_short_min),
                frustum_border,
                texture_image,
                texture_size,
                pixel_buffer,
            )


# Render functions
def render_mesh_line_no_culling(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType | None,
    texture_size: tuple[int, int] | None,
    pixel_buffer: dict[ScreenPoint2DType, ScreenPixelDataType],
) -> None:
    x_a, y_a, z_a, *_ = vertex_texture_normal_a
    x_b, y_b, z_b, *_ = vertex_texture_normal_b
    x_c, y_c, z_c, *_ = vertex_texture_normal_c
    vertex_a = (x_a, y_a, z_a)
    vertex_b = (x_b, y_b, z_b)
    vertex_c = (x_c, y_c, z_c)
    get_untextured_line_bresenham(vertex_a, vertex_b, frustum_border, pixel_buffer)
    get_untextured_line_bresenham(vertex_b, vertex_c, frustum_border, pixel_buffer)
    get_untextured_line_bresenham(vertex_c, vertex_a, frustum_border, pixel_buffer)


def render_mesh_line_backface_culling(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType | None,
    texture_size: tuple[int, int] | None,
    pixel_buffer: dict[ScreenPoint2DType, ScreenPixelDataType],
) -> None:
    x_a, y_a, z_a, *_ = vertex_texture_normal_a
    x_b, y_b, z_b, *_ = vertex_texture_normal_b
    x_c, y_c, z_c, *_ = vertex_texture_normal_c
    v_ab_x, v_ab_y = x_b - x_a, y_b - y_a
    v_bc_x, v_bc_y = x_c - x_b, y_c - y_b
    vertex_a = (x_a, y_a, z_a)
    vertex_b = (x_b, y_b, z_b)
    vertex_c = (x_c, y_c, z_c)
    # Back-face culling
    if v_ab_y * v_bc_x - v_ab_x * v_bc_y < 0:
        return
    get_untextured_line_bresenham(vertex_a, vertex_b, frustum_border, pixel_buffer)
    get_untextured_line_bresenham(vertex_b, vertex_c, frustum_border, pixel_buffer)
    get_untextured_line_bresenham(vertex_c, vertex_a, frustum_border, pixel_buffer)


def render_untextured_model(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType | None,
    texture_size: tuple[int, int] | None,
    pixel_buffer: dict[ScreenPoint2DType, ScreenPixelDataType],
) -> None:
    x_a, y_a, z_a, *_ = vertex_texture_normal_a
    x_b, y_b, z_b, *_ = vertex_texture_normal_b
    x_c, y_c, z_c, *_ = vertex_texture_normal_c
    v_ab_x, v_ab_y = x_b - x_a, y_b - y_a
    v_bc_x, v_bc_y = x_c - x_b, y_c - y_b
    # Back-face culling
    if v_ab_y * v_bc_x - v_ab_x * v_bc_y < 0:
        return
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
    line_ac = get_untextured_line_bresenham(
        vertex_a, vertex_c, frustum_border, pixel_buffer
    )
    # Other lines
    line_ab = get_untextured_line_bresenham(
        vertex_a, vertex_b, frustum_border, pixel_buffer
    )
    line_bc = get_untextured_line_bresenham(
        vertex_b, vertex_c, frustum_border, pixel_buffer
    )
    # Longest line xs and zs on y
    xzsony_ac: dict[int, list[tuple[int, float]]] = {}
    for (x, y), (z, *_) in line_ac.items():
        xzsony_ac.setdefault(y, [])
        xzsony_ac[y].append((x, z))
    # Other lines xs and zs on y
    xzsony_ab: dict[int, list[tuple[int, float]]] = {}
    for (x, y), (z, *_) in line_ab.items():
        xzsony_ab.setdefault(y, [])
        xzsony_ab[y].append((x, z))
    xzsony_bc: dict[int, list[tuple[int, float]]] = {}
    for (x, y), (z, *_) in line_bc.items():
        xzsony_bc.setdefault(y, [])
        xzsony_bc[y].append((x, z))
    # Sweep line algorithm
    get_untextured_line_sweepline(xzsony_ac, xzsony_ab, frustum_border, pixel_buffer)
    get_untextured_line_sweepline(xzsony_ac, xzsony_bc, frustum_border, pixel_buffer)


def render_textured_model(
    vertex_texture_normal_a: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_b: Vertex3DTexture3DNormal3DType,
    vertex_texture_normal_c: Vertex3DTexture3DNormal3DType,
    frustum_border: FrustumBorderType,
    texture_image: ImageType | None,
    texture_size: tuple[int, int] | None,
    pixel_buffer: dict[ScreenPoint2DType, ScreenPixelDataType],
) -> None:
    if texture_image is None or texture_size is None:
        render_untextured_model(
            vertex_texture_normal_a,
            vertex_texture_normal_b,
            vertex_texture_normal_c,
            frustum_border,
            texture_image,
            texture_size,
            pixel_buffer,
        )
        return
    x_a, y_a, *_ = vertex_texture_normal_a
    x_b, y_b, *_ = vertex_texture_normal_b
    x_c, y_c, *_ = vertex_texture_normal_c
    v_ab_x, v_ab_y = x_b - x_a, y_b - y_a
    v_bc_x, v_bc_y = x_c - x_b, y_c - y_b
    # Back-face culling
    if v_ab_y * v_bc_x - v_ab_x * v_bc_y < 0:
        return
    # Sort by y coordinate
    vertex_texture_normal_a, vertex_texture_normal_b, vertex_texture_normal_c = sorted(
        (vertex_texture_normal_a, vertex_texture_normal_b, vertex_texture_normal_c),
        key=lambda _: _[1],
        reverse=True,
    )
    # Longest line
    line_ac = get_textured_line_bresenham_margin(
        vertex_texture_normal_a,
        vertex_texture_normal_c,
        vertex_texture_normal_b,
        frustum_border,
        texture_image,
        texture_size,
        pixel_buffer,
    )
    # Other lines
    line_ab = get_textured_line_bresenham_margin(
        vertex_texture_normal_a,
        vertex_texture_normal_b,
        vertex_texture_normal_c,
        frustum_border,
        texture_image,
        texture_size,
        pixel_buffer,
    )
    line_bc = get_textured_line_bresenham_margin(
        vertex_texture_normal_b,
        vertex_texture_normal_c,
        vertex_texture_normal_a,
        frustum_border,
        texture_image,
        texture_size,
        pixel_buffer,
    )
    # Longest line xs and zs on y
    xzsony_ac: dict[int, list[tuple[int, float]]] = {}
    for (x, y), (z, *_) in line_ac.items():
        xzsony_ac.setdefault(y, [])
        xzsony_ac[y].append((x, z))
    # Other lines xs and zs on y
    xzsony_ab: dict[int, list[tuple[int, float]]] = {}
    for (x, y), (z, *_) in line_ab.items():
        xzsony_ab.setdefault(y, [])
        xzsony_ab[y].append((x, z))
    xzsony_bc: dict[int, list[tuple[int, float]]] = {}
    for (x, y), (z, *_) in line_bc.items():
        xzsony_bc.setdefault(y, [])
        xzsony_bc[y].append((x, z))
    # Sweep line algorithm
    get_textured_line_sweepline(
        vertex_texture_normal_a,
        vertex_texture_normal_b,
        vertex_texture_normal_c,
        xzsony_ac,
        xzsony_ab,
        frustum_border,
        texture_image,
        texture_size,
        pixel_buffer,
    )
    get_textured_line_sweepline(
        vertex_texture_normal_a,
        vertex_texture_normal_b,
        vertex_texture_normal_c,
        xzsony_ac,
        xzsony_bc,
        frustum_border,
        texture_image,
        texture_size,
        pixel_buffer,
    )
