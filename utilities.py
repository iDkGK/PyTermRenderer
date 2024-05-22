import math
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


class Camera(object):
    def __init__(
        self,
        *,
        fov: float,
        coordinate: tuple[float, float, float],
        rotation: tuple[float, float, float],
        move_speed: float = 1.0,
        dash_speed: float = 2.0,
    ) -> None:
        self._fov = fov
        self._x, self._y, self._z = coordinate
        self._original_coordinate = coordinate
        self._yaw, self._pitch, self._roll = rotation
        self._original_rotation = rotation
        self._move_speed = abs(move_speed)
        self._dash_speed = abs(dash_speed)
        self._update_trigonometrics()
        self._update_vector()

    # Properties
    @property
    def info(self) -> tuple[str, ...]:
        return (
            "Camera FOV: %f" % self.fov,
            "Camera coordinate (X, Y, Z): (%f, %f, %f)" % self.coordinate,
            "Camera rotation (Yaw, Pitch, Roll): (%f, %f, %f)" % self.rotation,
            "Camera direction vector (X, Y, Z): (%f, %f, %f)" % self.vector,
        )

    @property
    def fov(self) -> float:
        return self._fov

    @property
    def coordinate(self) -> tuple[float, float, float]:
        return (self._x, self._y, self._z)

    @property
    def rotation(self) -> tuple[float, float, float]:
        return (self._yaw, self._pitch, self._roll)

    @property
    def trigonometrics(self) -> tuple[float, float, float, float, float, float]:
        return (
            self._sin_yaw,
            self._cos_yaw,
            self._sin_pitch,
            self._cos_pitch,
            self._sin_roll,
            self._cos_roll,
        )

    @property
    def vector(self) -> tuple[float, float, float]:
        return (self._vector_x, self._vector_y, self._vector_z)

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
        self, *, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0
    ) -> None:
        self._yaw += yaw
        self._pitch += pitch
        self._roll += roll
        self._yaw = max(min(self._yaw, 90.0), -90.0)
        self._pitch %= 360.0
        self._roll %= 360.0
        self._update_trigonometrics()
        self._update_vector()

    def reset(self) -> None:
        self._x, self._y, self._z = self._original_coordinate
        self._yaw, self._pitch, self._roll = self._original_rotation
        self._update_trigonometrics()
        self._update_vector()

    # Update methods
    def update(self) -> None:
        pass

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


class SmoothCamera(Camera):
    def __init__(
        self,
        *,
        fov: float,
        coordinate: tuple[float, float, float],
        rotation: tuple[float, float, float],
        move_speed: float = 1.0,
        dash_speed: float = 2.0,
        deacceleration_rate: float = 0.5,
    ) -> None:
        self._acceleration_x = 0.0
        self._acceleration_y = 0.0
        self._acceleration_z = 0.0
        self._move_acceleration = abs(move_speed / 2.0)
        self._dash_acceleration = abs(dash_speed / 2.0)
        self._deacceleration_rate = max(min(deacceleration_rate, 1.0), 0.0)
        super().__init__(
            fov=fov,
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
    def update(self) -> None:
        self._x += self._acceleration_x
        self._y += self._acceleration_y
        self._z += self._acceleration_z
        self._acceleration_x -= self._acceleration_x * self._deacceleration_rate
        self._acceleration_y -= self._acceleration_y * self._deacceleration_rate
        self._acceleration_z -= self._acceleration_z * self._deacceleration_rate

    def reset(self) -> None:
        self._acceleration_x = 0.0
        self._acceleration_y = 0.0
        self._acceleration_z = 0.0
        super().reset()


class GravitySmoothCamera(SmoothCamera):
    def __init__(
        self,
        *,
        fov: float,
        coordinate: tuple[float, float, float],
        rotation: tuple[float, float, float],
        gravity: float = 1.0,
        jump_height: float = 5.0,
        move_speed: float = 1.0,
        dash_speed: float = 2.0,
        deacceleration_rate: float = 0.5,
    ) -> None:
        self._gravity = abs(gravity)
        self._jump_height = abs(jump_height)
        super().__init__(
            fov=fov,
            coordinate=coordinate,
            rotation=rotation,
            move_speed=move_speed,
            dash_speed=dash_speed,
            deacceleration_rate=deacceleration_rate,
        )

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

    def update(self) -> None:
        self._x += self._acceleration_x
        self._z += self._acceleration_z
        self._acceleration_x -= self._acceleration_x * self._deacceleration_rate
        self._acceleration_z -= self._acceleration_z * self._deacceleration_rate

        self._y += self._acceleration_y
        self._y = max(self._y, 0.0)
        if self._y > 0.0:
            self._acceleration_y -= self._gravity * self._deacceleration_rate
        else:
            self._y = 0.0
