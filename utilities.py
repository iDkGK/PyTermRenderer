import math
import sys
import warnings
from pathlib import Path
from threading import Lock

from hintings import FrameType, Point3DType, RotationType, Vertex3DType, Triangle3DType


def sort_counterclockwisely(
    vertex_a: tuple[float, float, float],
    vertex_b: tuple[float, float, float],
    vertex_c: tuple[float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    # Sort the vertices counterclockwisely
    vertex_a, vertex_b, vertex_c = sorted(
        (vertex_a, vertex_b, vertex_c),
        key=lambda tuple_xy: tuple_xy[1],
        reverse=True,
    )
    if vertex_b[1] == vertex_a[1] and vertex_b[0] > vertex_a[0]:
        vertex_a, vertex_b = vertex_b, vertex_a
    else:
        vertex_b, vertex_c = sorted(
            (vertex_b, vertex_c),
            key=lambda tuple_xy: tuple_xy[0],
        )
    return vertex_a, vertex_b, vertex_c


def bresenham_line(
    vertex1: tuple[float, float, float],
    vertex2: tuple[float, float, float],
    border: tuple[int, int, int, int],
) -> set[tuple[int, int]]:
    # Bresenham line algorithm
    line: set[tuple[int, int]] = set()
    zbuffer: dict[tuple[int, int], float] = {}
    left, right, top, bottom = border
    (x1, y1, z1), (x2, y2, _) = vertex1, vertex2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x1 < left and y1 < bottom and x2 < left and y2 < bottom:
        return line
    if x1 > right and y1 > top and x2 > right and y2 > top:
        return line
    end_coordinate = (x2, y2)
    delta_x = abs(x2 - x1)
    delta_y = abs(y2 - y1)
    step_x = 1 if x1 < x2 else -1
    step_y = 1 if y1 < y2 else -1
    error = delta_x - delta_y
    while True:
        middle_coordinate = (x1, y1)
        if (
            zbuffer.get(middle_coordinate, 1.0) > z1
            and left <= x1 <= right
            and bottom <= y1 <= top
        ):
            line.add(middle_coordinate)
            zbuffer[middle_coordinate] = z1
        if middle_coordinate == end_coordinate:
            break
        double_error = 2 * error
        if double_error > -delta_y:
            error -= delta_y
            x1 += step_x
        if double_error < delta_x:
            error += delta_x
            y1 += step_y
    return line


def get_mesh_line(
    vertices: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    border: tuple[int, int, int, int],
) -> set[tuple[int, int]]:
    vertex_a, vertex_b, vertex_c = vertices
    # There is a risk that the k of the line goes infinite or 0,
    # leading to poor performance of bresenham line algorithm.
    # TODO: fixme.
    return (
        bresenham_line(vertex_a, vertex_b, border)
        | bresenham_line(vertex_b, vertex_c, border)
        | bresenham_line(vertex_c, vertex_a, border)
    )


class Object(object):
    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)
        self._name = ""
        self._triangle_vertices: set[Triangle3DType] = set()
        # Parse file data and retrieve triangles vertices
        vertices: list[Vertex3DType] = []
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
    def triangle_vertices(self) -> set[Triangle3DType]:
        return self._triangle_vertices

    # Update methods
    def update(self, delta_time: float = 0.0) -> None:
        pass

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
        self._screen_border = (
            -self._half_width,
            self._screen_width - self._half_width,
            self._screen_height - self._half_height,
            -self._half_height,
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
        self._near_plane = near_plane
        self._far_plane = far_plane
        self._screen_depth = far_plane - near_plane
        self._x, self._y, self._z = coordinate
        self._original_coordinate = coordinate
        self._yaw, self._pitch = rotation
        self._original_rotation = rotation
        self._move_speed = abs(move_speed)
        self._dash_speed = abs(dash_speed)
        self._controllable = controllable
        self._objects: set[Object] = set()
        self._lines: set[tuple[int, int]] = set()
        self._information: list[str] = []
        self._delta_time = 0.0
        # Register controller
        if not self._controllable:
            return
        try:
            import keyboard

        except ImportError:
            keyboard = None
            warnings.warn(
                "no third-party support for keyboard. Using custom KeyboardListener."
            )
        try:
            import mouse  # type: ignore

        except ImportError:
            mouse = None
            warnings.warn(
                "no third-party support for mouse. Using custom KeyboardListener."
            )
        # keyboard = mouse = None
        if keyboard is None or mouse is None:
            from controller import KeyboardListener

            keyboard_listener = KeyboardListener()
        else:
            keyboard_listener = None

        # Quit and reset state
        self._quit_state = False
        self._reset_state = False

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
            from keyboard import KeyboardEvent, KEY_DOWN, KEY_UP

            active_states = {KEY_DOWN: True, KEY_UP: False, None: False}

            def quit(event: KeyboardEvent) -> None:
                self._quit_state = active_states.get(event.event_type, False)

            def reset(event: KeyboardEvent) -> None:
                self._reset_state = active_states.get(event.event_type, False)

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

            # Quiting
            keyboard.hook_key("escape", quit)
            # Position
            keyboard.hook_key("r", reset)
            keyboard.hook_key("shift", dash)
            keyboard.hook_key("w", move_forward)
            keyboard.hook_key("s", move_backward)
            keyboard.hook_key("a", move_leftward)
            keyboard.hook_key("d", move_rightward)
            keyboard.hook_key("space", move_upward)
            keyboard.hook_key("ctrl", move_downward)
        # Mouse
        if mouse is not None:
            from mouse import ButtonEvent, WheelEvent, MoveEvent  # type: ignore

            rotate_lock = Lock()

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

            def stop_and_quit() -> None:
                keyboard_listener.stop()
                self._quit_state = True

            # Quiting
            keyboard_listener.register("\x1b", stop_and_quit)
            # Position
            keyboard_listener.register("r", self._reset)
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
        self._lines: set[tuple[int, int]] = set()
        for obj in self._objects:
            for triangle_vertices in obj.triangle_vertices:
                (
                    (triangle_a_x, triangle_a_y, triangle_a_z),
                    (triangle_b_x, triangle_b_y, triangle_b_z),
                    (triangle_c_x, triangle_c_y, triangle_c_z),
                ) = triangle_vertices
                # Position
                # Using vector for relative position
                (triangle_a_x, triangle_a_y, triangle_a_z) = (
                    triangle_a_x - self._x,
                    triangle_a_y - self._y,
                    triangle_a_z - self._z,
                )
                (triangle_b_x, triangle_b_y, triangle_b_z) = (
                    triangle_b_x - self._x,
                    triangle_b_y - self._y,
                    triangle_b_z - self._z,
                )
                (triangle_c_x, triangle_c_y, triangle_c_z) = (
                    triangle_c_x - self._x,
                    triangle_c_y - self._y,
                    triangle_c_z - self._z,
                )
                # Rotation
                # Y-axis rotation that affects X/Z coordinates
                triangle_a_x, triangle_a_z = (
                    triangle_a_x * self._cos_pitch - triangle_a_z * self._sin_pitch,
                    triangle_a_x * self._sin_pitch + triangle_a_z * self._cos_pitch,
                )
                triangle_b_x, triangle_b_z = (
                    triangle_b_x * self._cos_pitch - triangle_b_z * self._sin_pitch,
                    triangle_b_x * self._sin_pitch + triangle_b_z * self._cos_pitch,
                )
                triangle_c_x, triangle_c_z = (
                    triangle_c_x * self._cos_pitch - triangle_c_z * self._sin_pitch,
                    triangle_c_x * self._sin_pitch + triangle_c_z * self._cos_pitch,
                )
                # X-axis rotation that affects Y/Z coordinates
                triangle_a_y, triangle_a_z = (
                    triangle_a_y * self._cos_yaw + triangle_a_z * self._sin_yaw,
                    -triangle_a_y * self._sin_yaw + triangle_a_z * self._cos_yaw,
                )
                triangle_b_y, triangle_b_z = (
                    triangle_b_y * self._cos_yaw + triangle_b_z * self._sin_yaw,
                    -triangle_b_y * self._sin_yaw + triangle_b_z * self._cos_yaw,
                )
                triangle_c_y, triangle_c_z = (
                    triangle_c_y * self._cos_yaw + triangle_c_z * self._sin_yaw,
                    -triangle_c_y * self._sin_yaw + triangle_c_z * self._cos_yaw,
                )
                # Simple near/far plane culling
                # TODO: implement advanced near/far plane culling
                if not (
                    self._near_plane < triangle_a_z < self._far_plane
                    and self._near_plane < triangle_b_z < self._far_plane
                    and self._near_plane < triangle_c_z < self._far_plane
                ):
                    continue
                # Triangle vertices on camera screen
                camera_triangle_vertex_a = (
                    self._focal * triangle_a_x / triangle_a_z,
                    self._focal * triangle_a_y / triangle_a_z,
                    triangle_a_z / self._screen_depth,
                )
                camera_triangle_vertex_b = (
                    self._focal * triangle_b_x / triangle_b_z,
                    self._focal * triangle_b_y / triangle_b_z,
                    triangle_b_z / self._screen_depth,
                )
                camera_triangle_vertex_c = (
                    self._focal * triangle_c_x / triangle_c_z,
                    self._focal * triangle_c_y / triangle_c_z,
                    triangle_c_z / self._screen_depth,
                )
                # Triangle on camera screen
                triangle_vertices = sort_counterclockwisely(
                    vertex_a=camera_triangle_vertex_a,
                    vertex_b=camera_triangle_vertex_b,
                    vertex_c=camera_triangle_vertex_c,
                )
                self._lines |= get_mesh_line(triangle_vertices, self._screen_border)

    def _update_infomation(self, delta_time: float) -> None:
        self._information = [
            *(("".ljust(self._screen_width),) * self._screen_height),
            ("FOV: %f" % self._field_of_view).ljust(self._screen_width),
            ("Coordinate (X, Y, Z): (%f, %f, %f)" % (self._x, self._y, self._z)).ljust(
                self._screen_width
            ),
            ("Rotation (Yaw, Pitch): (%f, %f)" % (self._yaw, self._pitch)).ljust(
                self._screen_width
            ),
            (
                "Direction vector (X, Z): (%f, %f, %f)"
                % (self._vector_x, self._vector_y, self._vector_z)
            ).ljust(self._screen_width),
        ]
        self._information.reverse()

    def update(self, delta_time: float = 0.0) -> None:
        self._update_position(delta_time)
        self._update_trigonometrics(delta_time)
        self._update_vector(delta_time)
        self._update_objects(delta_time)
        self._update_infomation(delta_time)

    # Draw methods
    def get_frame(self) -> FrameType:
        frame: FrameType = [
            [
                (
                    (255, 255, 255, 255, ord(self._information[y][x]))
                    if self._information[y][x] != " "
                    else (
                        (255, 255, 255, 255, 9608)
                        if ((x - self._half_width) // 2, y - self._half_height)
                        in self._lines
                        else (255, 255, 255, 255, 32)
                    )
                )
                for x in range(0, self._screen_width)
            ]
            for y in range(0, self._screen_height)
        ]
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
