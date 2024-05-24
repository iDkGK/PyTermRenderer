import math
from pathlib import Path
from hintings import MatrixType


def matrix_transpose(matrix: MatrixType) -> MatrixType:
    return list(map(list, zip(*matrix)))


def matrix_add(*matrices: MatrixType) -> MatrixType:
    matrices_iterator = iter(matrices)
    first_matrix = next(matrices_iterator)
    # deep copy to avoid modification of first matrix
    result_matrix: MatrixType = [
        first_matrix_row[:] for first_matrix_row in first_matrix
    ]
    # continuously addition
    for other_matrix in matrices_iterator:
        for row_index, (result_matrix_row, other_matrix_row) in enumerate(
            zip(result_matrix, other_matrix)
        ):
            for element_index in range(0, len(result_matrix_row)):
                result_matrix[row_index][element_index] = (  # type: ignore
                    result_matrix_row[element_index] + other_matrix_row[element_index]
                )
    return result_matrix


def matrix_subtract(*matrices: MatrixType) -> MatrixType:
    matrices_iterator = iter(matrices)
    first_matrix = next(matrices_iterator)
    # deep copy to avoid modification of first matrix
    result_matrix: MatrixType = [
        first_matrix_row[:] for first_matrix_row in first_matrix
    ]
    # continuously subtraction
    for other_matrix in matrices_iterator:
        for row_index, (result_matrix_row, other_matrix_row) in enumerate(
            zip(result_matrix, other_matrix)
        ):
            for element_index in range(0, len(result_matrix_row)):
                result_matrix[row_index][element_index] = (  # type: ignore
                    result_matrix_row[element_index] - other_matrix_row[element_index]
                )
    return result_matrix


def matrix_multiply(*matrices: MatrixType) -> MatrixType:
    matrices_iterator = iter(matrices)
    first_matrix = next(matrices_iterator)
    # deep copy to avoid modification of first matrix
    result_matrix: MatrixType = [
        first_matrix_row[:] for first_matrix_row in first_matrix
    ]
    # continuously multiplication
    for other_matrix in matrices_iterator:
        other_matrix = matrix_transpose(other_matrix)
        result_matrix = [
            [
                sum(
                    result_matrix_element * other_matrix_element
                    for result_matrix_element, other_matrix_element in zip(row, column)
                )
                for column in other_matrix
            ]
            for row in result_matrix
        ]
    return result_matrix


class Vector(object):
    def __init__(self, x: float, y: float) -> None:
        self._x = x
        self._y = y

    def __matmul__(self, other: "Vector") -> float:
        return self._x * other._x + self._y * other._y

    def __rmatmul__(self, other: "Vector") -> float:
        return other._x * self._x + other._y * self._y


class Triangle(object):
    def __init__(
        self,
        null: bool = False,
        vertex_a: tuple[float, float] = (0.0, 0.0),
        vertex_b: tuple[float, float] = (0.0, 0.0),
        vertex_c: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self._null = null
        self._x_a, self._y_a = vertex_a
        x_b, y_b = vertex_b
        x_c, y_c = vertex_c
        self._v_ab = Vector(x_b - self._x_a, y_b - self._y_a)
        self._v_ac = Vector(x_c - self._x_a, y_c - self._y_a)
        self._p_ab_ab = self._v_ab @ self._v_ab
        self._p_ac_ac = self._v_ac @ self._v_ac
        self._p_ab_ac = self._v_ab @ self._v_ac
        self._p_ac_ab = self._v_ac @ self._v_ab
        self._d_ab_ab_ac_ac_ac_ab_ac_ab = (
            self._p_ab_ab * self._p_ac_ac - self._p_ac_ab * self._p_ac_ab
        )

    def __contains__(self, point: tuple[float, float]) -> bool:
        if self._null:
            return False
        x_p, y_p = point
        v_ap = Vector(x_p - self._x_a, y_p - self._y_a)
        p_ap_ab, p_ap_ac = v_ap @ self._v_ab, v_ap @ self._v_ac
        d_ap_ab_ac_ac_ap_ac_ab_ac = p_ap_ab * self._p_ac_ac - p_ap_ac * self._p_ab_ac
        if d_ap_ab_ac_ac_ap_ac_ab_ac < 0.0:
            return False
        d_ap_ac_ab_ab_ap_ab_ac_ab = p_ap_ac * self._p_ab_ab - p_ap_ab * self._p_ac_ab
        if d_ap_ac_ab_ab_ap_ab_ac_ab < 0.0:
            return False
        if (
            d_ap_ab_ac_ac_ap_ac_ab_ac
            + d_ap_ac_ab_ab_ap_ab_ac_ab
            - self._d_ab_ab_ac_ac_ac_ab_ac_ab
            > 0.0
        ):
            return False
        return True


class Object(object):
    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)
        self._name = ""
        self._triangle_vertices: set[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ]
        ] = set()
        # Parse file data and retrieve triangles vertices
        vertices: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        textures: list[tuple[float, ...]] = []
        faces: list[tuple[int, ...]] = []
        for line in self._filepath.read_text().strip().splitlines():
            data_type, *data = line.strip().split()
            if data_type == "o":
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
                    textures.append((u, v, w))
                elif len(vw) == 1:
                    (v,) = vw
                    textures.append((u, v))
                elif len(vw) == 0:
                    textures.append((u,))
            elif data_type == "s":
                (data,) = data
                if data == "off":
                    group_number = None  # type: ignore
                else:
                    group_number = int(data)  # type: ignore
            elif data_type == "f":
                face: list[int] | tuple[int, ...] = []
                for part in data:
                    v_index, vt_index, vn_index = part.split("/")  # type: ignore
                    face.append(int(v_index) - 1)
                faces.append(tuple(face))
        for face in faces:
            index_a, index_b, index_c, *_ = face
            self._triangle_vertices.add(
                (
                    vertices[index_a],
                    vertices[index_b],
                    vertices[index_c],
                )
            )

    # Properties
    @property
    def name(self) -> str:
        return self._name

    @property
    def triangle_vertices(self) -> set[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ]:
        return self._triangle_vertices

    # Update methods
    def update(self, delta_time: float = 0.0) -> None:
        pass

    # Camera-related methods
    def show_to(self, camera: "Camera") -> None:
        camera.show_object(self)

    def hide_from(self, camera: "Camera") -> None:
        camera.hide_object(self)


class ScreenTooSmallError(Exception):
    pass


class Camera(object):
    def __init__(
        self,
        *,
        fov: float,
        view: tuple[int, int],
        coordinate: tuple[float, float, float],
        rotation: tuple[float, float, float],
        move_speed: float = 0.5,
        dash_speed: float = 1.0,
    ) -> None:
        self._fov = fov
        self._screen_width, self._screen_height = view
        self._screen_width = (self._screen_width // 2) * 2 or 2
        self._half_width, self._half_height = (
            self._screen_width / 2,
            self._screen_height / 2,
        )
        if self._screen_height < 0:
            raise ScreenTooSmallError("screen is too small to render objects.")
        self._focal = (
            max(self._screen_width, self._screen_height)
            / math.tan(math.radians(fov / 2.0))
            / 2
        )
        self._x, self._y, self._z = coordinate
        self._original_coordinate = coordinate
        self._yaw, self._pitch, self._roll = rotation
        self._original_rotation = rotation
        self._move_speed = abs(move_speed)
        self._dash_speed = abs(dash_speed)
        self._objects: set[Object] = set()
        self._triangles: list[Triangle] = []
        self._information: list[str] = []

    # Move methods
    def move_forward(self) -> None:
        self._x += self._move_speed * self._vector_x
        self._z += self._move_speed * self._vector_z

    def move_backward(self) -> None:
        self._x -= self._move_speed * self._vector_x
        self._z -= self._move_speed * self._vector_z

    def move_leftward(self) -> None:
        self._x -= self._move_speed * self._vector_z
        self._z += self._move_speed * self._vector_x

    def move_rightward(self) -> None:
        self._x += self._move_speed * self._vector_z
        self._z -= self._move_speed * self._vector_x

    def move_upward(self) -> None:
        self._y += self._move_speed

    def move_downward(self) -> None:
        self._y -= self._move_speed

    def dash_forward(self) -> None:
        self._x += self._dash_speed * self._vector_x
        self._z += self._dash_speed * self._vector_z

    def dash_backward(self) -> None:
        self._x -= self._dash_speed * self._vector_x
        self._z -= self._dash_speed * self._vector_z

    def dash_leftward(self) -> None:
        self._x -= self._dash_speed * self._vector_z
        self._z += self._dash_speed * self._vector_x

    def dash_rightward(self) -> None:
        self._x += self._dash_speed * self._vector_z
        self._z -= self._dash_speed * self._vector_x

    def dash_upward(self) -> None:
        self._y += self._dash_speed

    def dash_downward(self) -> None:
        self._y -= self._dash_speed

    # Rotate methods
    def rotate(
        self,
        *,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        self._yaw += yaw
        self._pitch += pitch
        self._roll += roll
        self._yaw = max(min(self._yaw, 90.0), -90.0)
        self._pitch %= 360.0
        self._roll %= 360.0

    # Reset methods
    def reset(self) -> None:
        self._x, self._y, self._z = self._original_coordinate
        self._yaw, self._pitch, self._roll = self._original_rotation

    # Update methods
    def update(self, delta_time: float = 0.0) -> None:
        self._update_trigonometrics()
        self._update_vector()
        self._update_objects()
        self._update_infomation()

    def _update_trigonometrics(self) -> None:
        yaw_radians = math.radians(-self._yaw)
        pitch_radians = math.radians(self._pitch)
        roll_radians = math.radians(self._roll)
        self._sin_yaw = math.sin(yaw_radians)
        self._cos_yaw = math.cos(yaw_radians)
        self._sin_pitch = math.sin(pitch_radians)
        self._cos_pitch = math.cos(pitch_radians)
        self._sin_roll = math.sin(roll_radians)
        self._cos_roll = math.cos(roll_radians)

    def _update_vector(self) -> None:
        self._vector_x = self._sin_pitch  # X component
        self._vector_y = 0.0  # Y component
        self._vector_z = self._cos_pitch  # Z component

    def _update_objects(self) -> None:
        # Triangles
        self._triangles.clear()
        # Iteration over all triangles
        # Assuming that every triangle is ▲abc
        for obj in self._objects:
            for triangle_vertices in obj.triangle_vertices:
                (
                    (triangle_a_x, triangle_a_y, triangle_a_z),
                    (triangle_b_x, triangle_b_y, triangle_b_z),
                    (triangle_c_x, triangle_c_y, triangle_c_z),
                ) = triangle_vertices
                # Position
                # Using vector for relative position
                (
                    distance_camera_triangle_a_x,
                    distance_camera_triangle_a_y,
                    distance_camera_triangle_a_z,
                ) = (
                    triangle_a_x - self._x,
                    triangle_a_y - self._y,
                    triangle_a_z - self._z,
                )
                (
                    distance_camera_triangle_b_x,
                    distance_camera_triangle_b_y,
                    distance_camera_triangle_b_z,
                ) = (
                    triangle_b_x - self._x,
                    triangle_b_y - self._y,
                    triangle_b_z - self._z,
                )
                (
                    distance_camera_triangle_c_x,
                    distance_camera_triangle_c_y,
                    distance_camera_triangle_c_z,
                ) = (
                    triangle_c_x - self._x,
                    triangle_c_y - self._y,
                    triangle_c_z - self._z,
                )
                # Rotation
                # Z-axis rotation that affects X/Y coordinates
                distance_camera_triangle_a_x, distance_camera_triangle_a_y = (
                    distance_camera_triangle_a_x * self._cos_roll
                    + distance_camera_triangle_a_y * self._sin_roll,
                    -distance_camera_triangle_a_x * self._sin_roll
                    + distance_camera_triangle_a_y * self._cos_roll,
                )
                distance_camera_triangle_b_x, distance_camera_triangle_b_y = (
                    distance_camera_triangle_b_x * self._cos_roll
                    + distance_camera_triangle_b_y * self._sin_roll,
                    -distance_camera_triangle_b_x * self._sin_roll
                    + distance_camera_triangle_b_y * self._cos_roll,
                )
                distance_camera_triangle_c_x, distance_camera_triangle_c_y = (
                    distance_camera_triangle_c_x * self._cos_roll
                    + distance_camera_triangle_c_y * self._sin_roll,
                    -distance_camera_triangle_c_x * self._sin_roll
                    + distance_camera_triangle_c_y * self._cos_roll,
                )
                # Y-axis rotation that affects X/Z coordinates
                distance_camera_triangle_a_x, distance_camera_triangle_a_z = (
                    distance_camera_triangle_a_x * self._cos_pitch
                    - distance_camera_triangle_a_z * self._sin_pitch,
                    distance_camera_triangle_a_x * self._sin_pitch
                    + distance_camera_triangle_a_z * self._cos_pitch,
                )
                distance_camera_triangle_b_x, distance_camera_triangle_b_z = (
                    distance_camera_triangle_b_x * self._cos_pitch
                    - distance_camera_triangle_b_z * self._sin_pitch,
                    distance_camera_triangle_b_x * self._sin_pitch
                    + distance_camera_triangle_b_z * self._cos_pitch,
                )
                distance_camera_triangle_c_x, distance_camera_triangle_c_z = (
                    distance_camera_triangle_c_x * self._cos_pitch
                    - distance_camera_triangle_c_z * self._sin_pitch,
                    distance_camera_triangle_c_x * self._sin_pitch
                    + distance_camera_triangle_c_z * self._cos_pitch,
                )
                # X-axis rotation that affects Y/Z coordinates
                distance_camera_triangle_a_y, distance_camera_triangle_a_z = (
                    distance_camera_triangle_a_y * self._cos_yaw
                    + distance_camera_triangle_a_z * self._sin_yaw,
                    -distance_camera_triangle_a_y * self._sin_yaw
                    + distance_camera_triangle_a_z * self._cos_yaw,
                )
                distance_camera_triangle_b_y, distance_camera_triangle_b_z = (
                    distance_camera_triangle_b_y * self._cos_yaw
                    + distance_camera_triangle_b_z * self._sin_yaw,
                    -distance_camera_triangle_b_y * self._sin_yaw
                    + distance_camera_triangle_b_z * self._cos_yaw,
                )
                distance_camera_triangle_c_y, distance_camera_triangle_c_z = (
                    distance_camera_triangle_c_y * self._cos_yaw
                    + distance_camera_triangle_c_z * self._sin_yaw,
                    -distance_camera_triangle_c_y * self._sin_yaw
                    + distance_camera_triangle_c_z * self._cos_yaw,
                )
                # Simple culling. TODO: advanced culling mechanism.
                if (
                    distance_camera_triangle_a_z > 0.0
                    and distance_camera_triangle_b_z > 0.0
                    and distance_camera_triangle_c_z > 0.0
                ):
                    self._triangles.append(
                        Triangle(
                            vertex_a=(
                                self._focal
                                * distance_camera_triangle_a_x
                                / distance_camera_triangle_a_z,
                                self._focal
                                * distance_camera_triangle_a_y
                                / distance_camera_triangle_a_z,
                            ),
                            vertex_b=(
                                self._focal
                                * distance_camera_triangle_b_x
                                / distance_camera_triangle_b_z,
                                self._focal
                                * distance_camera_triangle_b_y
                                / distance_camera_triangle_b_z,
                            ),
                            vertex_c=(
                                self._focal
                                * distance_camera_triangle_c_x
                                / distance_camera_triangle_c_z,
                                self._focal
                                * distance_camera_triangle_c_y
                                / distance_camera_triangle_c_z,
                            ),
                        )
                    )

    def _update_infomation(self) -> None:
        self._information = [
            ("FOV: %f" % self._fov).ljust(self._screen_width),
            ("Coordinate (X, Y, Z): (%f, %f, %f)" % (self._x, self._y, self._z)).ljust(
                self._screen_width
            ),
            (
                "Rotation (Yaw, Pitch, Roll): (%f, %f, %f)"
                % (self._yaw, self._pitch, self._roll)
            ).ljust(self._screen_width),
            (
                "Direction vector (X, Y, Z): (%f, %f, %f)"
                % (self._vector_x, self._vector_y, self._vector_z)
            ).ljust(self._screen_width),
        ]
        self._information.reverse()

    # Draw methods
    def get_pixel(self, x: int, y: int) -> tuple[int, ...]:
        if (
            self._information[y:]
            and self._information[y:][0][x:]
            and self._information[y:][0][x:][0] != " "
        ):
            return (255, 255, 255, 255, ord(self._information[y:][0][x:][0]))
        coordinate = ((x - self._half_width) / 2, y - self._half_height)
        for triangle in self._triangles:
            if coordinate in triangle:
                return (255, 255, 255, 255, ord("█"))
        return (255, 255, 255, 255, ord(" "))

    # Objects-related methods
    def show_object(self, obj: Object) -> None:
        self._objects.add(obj)

    def hide_object(self, obj: Object) -> None:
        self._objects.remove(obj)


class SmoothCamera(Camera):
    def __init__(
        self,
        *,
        fov: float,
        view: tuple[int, int],
        coordinate: tuple[float, float, float],
        rotation: tuple[float, float, float],
        move_speed: float = 0.5,
        dash_speed: float = 1.0,
        inertia_ratio: float = 0.5,
    ) -> None:
        self._acceleration_x = 0.0
        self._acceleration_y = 0.0
        self._acceleration_z = 0.0
        self._move_acceleration = abs(move_speed / 2.0)
        self._dash_acceleration = abs(dash_speed / 2.0)
        self._inertia_ratio = 1.0 - max(min(inertia_ratio, 1.0), 0.0)
        super(SmoothCamera, self).__init__(
            fov=fov,
            view=view,
            coordinate=coordinate,
            rotation=rotation,
            move_speed=move_speed,
            dash_speed=dash_speed,
        )

    # Move methods
    def move_forward(self) -> None:
        self._acceleration_x += self._move_acceleration * self._vector_x
        self._acceleration_z += self._move_acceleration * self._vector_z
        self._acceleration_x = max(
            min(self._acceleration_x, self._move_speed), -self._move_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._move_speed), -self._move_speed
        )

    def move_backward(self) -> None:
        self._acceleration_x -= self._move_acceleration * self._vector_x
        self._acceleration_z -= self._move_acceleration * self._vector_z
        self._acceleration_x = max(
            min(self._acceleration_x, self._move_speed), -self._move_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._move_speed), -self._move_speed
        )

    def move_leftward(self) -> None:
        self._acceleration_x -= self._move_acceleration * self._vector_z
        self._acceleration_z += self._move_acceleration * self._vector_x
        self._acceleration_x = max(
            min(self._acceleration_x, self._move_speed), -self._move_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._move_speed), -self._move_speed
        )

    def move_rightward(self) -> None:
        self._acceleration_x += self._move_acceleration * self._vector_z
        self._acceleration_z -= self._move_acceleration * self._vector_x
        self._acceleration_x = max(
            min(self._acceleration_x, self._move_speed), -self._move_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._move_speed), -self._move_speed
        )

    def move_upward(self) -> None:
        self._acceleration_y += self._move_acceleration
        self._acceleration_y = max(
            min(self._acceleration_y, self._move_speed), -self._move_speed
        )

    def move_downward(self) -> None:
        self._acceleration_y -= self._move_acceleration
        self._acceleration_y = max(
            min(self._acceleration_y, self._move_speed), -self._move_speed
        )

    def dash_forward(self) -> None:
        self._acceleration_x += self._dash_acceleration * self._vector_x
        self._acceleration_z += self._dash_acceleration * self._vector_z
        self._acceleration_x = max(
            min(self._acceleration_x, self._dash_speed), -self._dash_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._dash_speed), -self._dash_speed
        )

    def dash_backward(self) -> None:
        self._acceleration_x -= self._dash_acceleration * self._vector_x
        self._acceleration_z -= self._dash_acceleration * self._vector_z
        self._acceleration_x = max(
            min(self._acceleration_x, self._dash_speed), -self._dash_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._dash_speed), -self._dash_speed
        )

    def dash_leftward(self) -> None:
        self._acceleration_x -= self._dash_acceleration * self._vector_z
        self._acceleration_z += self._dash_acceleration * self._vector_x
        self._acceleration_x = max(
            min(self._acceleration_x, self._dash_speed), -self._dash_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._dash_speed), -self._dash_speed
        )

    def dash_rightward(self) -> None:
        self._acceleration_x += self._dash_acceleration * self._vector_z
        self._acceleration_z -= self._dash_acceleration * self._vector_x
        self._acceleration_x = max(
            min(self._acceleration_x, self._dash_speed), -self._dash_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._dash_speed), -self._dash_speed
        )

    def dash_upward(self) -> None:
        self._acceleration_y += self._dash_acceleration
        self._acceleration_y = max(
            min(self._acceleration_y, self._dash_speed), -self._dash_speed
        )

    def dash_downward(self) -> None:
        self._acceleration_y -= self._dash_acceleration
        self._acceleration_y = max(
            min(self._acceleration_y, self._dash_speed), -self._dash_speed
        )

    # Update methods
    def update(self, delta_time: float = 0.0) -> None:
        self._x += self._acceleration_x
        self._y += self._acceleration_y
        self._z += self._acceleration_z
        self._acceleration_x -= self._acceleration_x * self._inertia_ratio
        self._acceleration_y -= self._acceleration_y * self._inertia_ratio
        self._acceleration_z -= self._acceleration_z * self._inertia_ratio
        super(SmoothCamera, self).update(delta_time)

    # Reset methods
    def reset(self) -> None:
        self._acceleration_x = 0.0
        self._acceleration_y = 0.0
        self._acceleration_z = 0.0
        super(SmoothCamera, self).reset()


GRAVITY = 0.1


class PlayerCamera(SmoothCamera):
    def __init__(
        self,
        *,
        fov: float,
        view: tuple[int, int],
        coordinate: tuple[float, float, float],
        rotation: tuple[float, float, float],
        move_speed: float = 0.5,
        dash_speed: float = 1.0,
        inertia_ratio: float = 0.5,
        jump_strength: float = 0.5,
    ) -> None:
        self._jump_height = abs(jump_strength)
        self._gravity = abs(GRAVITY)
        super(PlayerCamera, self).__init__(
            fov=fov,
            view=view,
            coordinate=coordinate,
            rotation=rotation,
            move_speed=move_speed,
            dash_speed=dash_speed,
            inertia_ratio=inertia_ratio,
        )

    # Move methods
    def move_upward(self) -> None:
        if self._y <= 0.0:
            self._acceleration_y = self._jump_height

    def move_downward(self) -> None:
        pass

    def dash_upward(self) -> None:
        if self._y <= 0.0:
            self._acceleration_y = self._jump_height

    def dash_downward(self) -> None:
        pass

    def jump(self) -> None:
        if self._y <= 0.0:
            self._acceleration_y = self._jump_height

    def crouch(self) -> None:
        pass

    # Update methods
    def update(self, delta_time: float = 0.0) -> None:
        self._x += self._acceleration_x
        self._z += self._acceleration_z
        self._acceleration_x -= self._acceleration_x * self._inertia_ratio
        self._acceleration_z -= self._acceleration_z * self._inertia_ratio

        self._y += self._acceleration_y
        self._y = max(self._y, 0.0)
        if self._y > 0.0:
            self._acceleration_y -= self._gravity * self._inertia_ratio
        else:
            self._y = 0.0
        super(SmoothCamera, self).update(delta_time)
