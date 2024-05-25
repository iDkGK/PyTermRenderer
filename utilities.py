import math
import sys
import warnings
from pathlib import Path

from hintings import Point3DType, RotationType, VertexType, TriangleType


class Triangle(object):
    def __init__(
        self,
        vertex_a: tuple[int, int],
        vertex_b: tuple[int, int],
        vertex_c: tuple[int, int],
    ) -> None:
        # Counterclock reorder
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
        # Bresenham line algorithm
        self._coordinates: set[tuple[int, int]] = set()
        for (x0, y0), (x1, y1) in (
            (vertex_a, vertex_b),
            (vertex_b, vertex_c),
            (vertex_c, vertex_a),
        ):
            delta_x = abs(x1 - x0)
            delta_y = abs(y1 - y0)
            step_x = 1 if x0 < x1 else -1
            step_y = 1 if y0 < y1 else -1
            error = delta_x - delta_y
            while True:
                self._coordinates.add((x0, y0))
                if x0 == x1 and y0 == y1:
                    break
                double_error = 2 * error
                if double_error > -delta_y:
                    error -= delta_y
                    x0 += step_x
                if double_error < delta_x:
                    error += delta_x
                    y0 += step_y

    def __contains__(self, coordinate: tuple[int, int]) -> bool:
        return coordinate in self._coordinates

    def get_pixel(self, x: int, y: int) -> tuple[int, ...]:
        return (255, 255, 255, 255, 9608)


class Object(object):
    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)
        self._name = ""
        self._triangle_vertices: set[TriangleType] = set()
        # Parse file data and retrieve triangles vertices
        vertices: list[VertexType] = []
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
    def triangle_vertices(self) -> set[TriangleType]:
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
        move_speed: float = 0.5,
        dash_speed: float = 1.0,
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
        self._near_plane = near_plane
        self._far_plane = far_plane
        self._x, self._y, self._z = coordinate
        self._original_coordinate = coordinate
        self._yaw, self._pitch, self._roll = rotation
        self._original_rotation = rotation
        self._move_speed = abs(move_speed)
        self._dash_speed = abs(dash_speed)
        self._objects: set[Object] = set()
        self._triangles: set[Triangle] = set()
        self._information: list[str] = []
        self._controllable = controllable
        self._register_controller()

    def _register_controller(self) -> None:
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

        # Quit state
        self._quit_state = False

        # With third-party modules
        # Move state
        self._dash_state = False
        self._forward_state = False
        self._backward_state = False
        self._leftward_state = False
        self._rightward_state = False
        self._upward_state = False
        self._downward_state = False
        self._reset_state = False
        # Position
        if keyboard is not None:
            from keyboard import KeyboardEvent, KEY_DOWN, KEY_UP

            active_states = {KEY_DOWN: True, KEY_UP: False, None: False}

            def quit(event: KeyboardEvent) -> None:
                self._quit_state = active_states.get(event.event_type, False)

            def dash(event: KeyboardEvent) -> None:
                self._dash_state = active_states.get(event.event_type, False)

            def move_forward(event: KeyboardEvent) -> None:
                self._forward_state = active_states.get(event.event_type, False)

            def move_backward(event: KeyboardEvent) -> None:
                self._backward_state = active_states.get(event.event_type, False)

            def move_leftward(event: KeyboardEvent) -> None:
                self._leftward_state = active_states.get(event.event_type, False)

            def move_rightward(event: KeyboardEvent) -> None:
                self._rightward_state = active_states.get(event.event_type, False)

            def move_upward(event: KeyboardEvent) -> None:
                self._upward_state = active_states.get(event.event_type, False)

            def move_downward(event: KeyboardEvent) -> None:
                self._downward_state = active_states.get(event.event_type, False)

            def reset(event: KeyboardEvent) -> None:
                self._reset_state = active_states.get(event.event_type, False)

            keyboard.hook_key("escape", quit)
            keyboard.hook_key("shift", dash)
            keyboard.hook_key("w", move_forward)
            keyboard.hook_key("s", move_backward)
            keyboard.hook_key("a", move_leftward)
            keyboard.hook_key("d", move_rightward)
            keyboard.hook_key("space", move_upward)
            keyboard.hook_key("ctrl", move_downward)
            keyboard.hook_key("r", reset)
        # Rotation
        if mouse is not None:
            from mouse import ButtonEvent, WheelEvent, MoveEvent  # type: ignore

            mouse_position = (self._screen_width, self._screen_height)
            mouse.move(*mouse_position)  # type: ignore

            def rotate(event: ButtonEvent | WheelEvent | MoveEvent):
                nonlocal mouse_position
                if type(event) == MoveEvent:
                    mouse_x, mouse_y = mouse_position
                    mouse.move(mouse_x, mouse_y)  # type: ignore
                    self._rotate(
                        yaw=-(event.y - mouse_y) / 18,  # type: ignore
                        pitch=+(event.x - mouse_x) / 18,  # type: ignore
                    )

            mouse.hook(rotate)  # type: ignore
        # With custom `KeyboardListener` as fallback
        if keyboard_listener is not None:

            def stop_keyboard_listener() -> None:
                keyboard_listener.stop()
                self._quit_state = True

            # Position
            keyboard_listener.register("\x1b", stop_keyboard_listener)
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
            keyboard_listener.register("r", self._reset)
            # Rotation
            keyboard_listener.register("8", self._rotate_yaw_forward)
            keyboard_listener.register("2", self._rotate_yaw_reverse)
            keyboard_listener.register("6", self._rotate_pitch_forward)
            keyboard_listener.register("4", self._rotate_pitch_reverse)
            keyboard_listener.register("e", self._rotate_roll_forward)
            keyboard_listener.register("q", self._rotate_roll_reverse)

    # Move methods
    def _move_forward(self) -> None:
        self._x += self._move_speed * self._vector_x
        self._z += self._move_speed * self._vector_z

    def _move_backward(self) -> None:
        self._x -= self._move_speed * self._vector_x
        self._z -= self._move_speed * self._vector_z

    def _move_leftward(self) -> None:
        self._x -= self._move_speed * self._vector_z
        self._z += self._move_speed * self._vector_x

    def _move_rightward(self) -> None:
        self._x += self._move_speed * self._vector_z
        self._z -= self._move_speed * self._vector_x

    def _move_upward(self) -> None:
        self._y += self._move_speed

    def _move_downward(self) -> None:
        self._y -= self._move_speed

    def _dash_forward(self) -> None:
        self._x += self._dash_speed * self._vector_x
        self._z += self._dash_speed * self._vector_z

    def _dash_backward(self) -> None:
        self._x -= self._dash_speed * self._vector_x
        self._z -= self._dash_speed * self._vector_z

    def _dash_leftward(self) -> None:
        self._x -= self._dash_speed * self._vector_z
        self._z += self._dash_speed * self._vector_x

    def _dash_rightward(self) -> None:
        self._x += self._dash_speed * self._vector_z
        self._z -= self._dash_speed * self._vector_x

    def _dash_upward(self) -> None:
        self._y += self._dash_speed

    def _dash_downward(self) -> None:
        self._y -= self._dash_speed

    # Rotate methods
    def _rotate_yaw_forward(self) -> None:
        self._yaw += 1.0
        self._yaw = max(min(self._yaw, 90.0), -90.0)

    def _rotate_yaw_reverse(self) -> None:
        self._yaw -= 1.0
        self._yaw = max(min(self._yaw, 90.0), -90.0)

    def _rotate_pitch_forward(self) -> None:
        self._pitch += 1.0
        self._pitch %= 360.0

    def _rotate_pitch_reverse(self) -> None:
        self._pitch -= 1.0
        self._pitch %= 360.0

    def _rotate_roll_forward(self) -> None:
        self._roll += 1.0
        self._roll %= 360.0

    def _rotate_roll_reverse(self) -> None:
        self._roll -= 1.0
        self._roll %= 360.0

    def _rotate(
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
    def _reset(self) -> None:
        self._x, self._y, self._z = self._original_coordinate
        self._yaw, self._pitch, self._roll = self._original_rotation

    # Update methods
    def _update_position(self) -> None:
        if not self._controllable:
            return
        if self._forward_state:
            if self._dash_state:
                self._dash_forward()
            else:
                self._move_forward()
        if self._backward_state:
            if self._dash_state:
                self._dash_backward()
            else:
                self._move_backward()
        if self._leftward_state:
            if self._dash_state:
                self._dash_leftward()
            else:
                self._move_leftward()
        if self._rightward_state:
            if self._dash_state:
                self._dash_rightward()
            else:
                self._move_rightward()
        if self._upward_state:
            if self._dash_state:
                self._dash_upward()
            else:
                self._move_upward()
        if self._downward_state:
            if self._dash_state:
                self._dash_downward()
            else:
                self._move_downward()
        if self._reset_state:
            self._reset()
        if self._quit_state:
            sys.exit(0)

    def _update_trigonometrics(self) -> None:
        yaw_radians = math.radians(-self._yaw)
        pitch_radians = math.radians(self._pitch)
        roll_radians = -math.radians(self._roll)
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
                # Z-axis rotation that affects X/Y coordinates
                triangle_a_x, triangle_a_y = (
                    triangle_a_x * self._cos_roll + triangle_a_y * self._sin_roll,
                    -triangle_a_x * self._sin_roll + triangle_a_y * self._cos_roll,
                )
                triangle_b_x, triangle_b_y = (
                    triangle_b_x * self._cos_roll + triangle_b_y * self._sin_roll,
                    -triangle_b_x * self._sin_roll + triangle_b_y * self._cos_roll,
                )
                triangle_c_x, triangle_c_y = (
                    triangle_c_x * self._cos_roll + triangle_c_y * self._sin_roll,
                    -triangle_c_x * self._sin_roll + triangle_c_y * self._cos_roll,
                )
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
                # Near/far plane culling
                if (
                    self._near_plane < triangle_a_z < self._far_plane
                    and self._near_plane < triangle_b_z < self._far_plane
                    and self._near_plane < triangle_c_z < self._far_plane
                ):
                    # Triangle vertices on camera screen
                    camera_triangle_vertex_a = (
                        int(self._focal * triangle_a_x / triangle_a_z),
                        int(self._focal * triangle_a_y / triangle_a_z),
                    )
                    camera_triangle_vertex_b = (
                        int(self._focal * triangle_b_x / triangle_b_z),
                        int(self._focal * triangle_b_y / triangle_b_z),
                    )
                    camera_triangle_vertex_c = (
                        int(self._focal * triangle_c_x / triangle_c_z),
                        int(self._focal * triangle_c_y / triangle_c_z),
                    )
                    # Triangle on camera screen
                    self._triangles.add(
                        Triangle(
                            vertex_a=camera_triangle_vertex_a,
                            vertex_b=camera_triangle_vertex_b,
                            vertex_c=camera_triangle_vertex_c,
                        )
                    )

    def _update_infomation(self) -> None:
        self._information = [
            ("FOV: %f" % self._field_of_view).ljust(self._screen_width),
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

    def update(self, delta_time: float = 0.0) -> None:
        self._update_position()
        self._update_trigonometrics()
        self._update_vector()
        self._update_objects()
        self._update_infomation()

    # Draw methods
    def get_pixel(self, x: int, y: int) -> tuple[int, ...]:
        if (
            self._information[y:]
            and self._information[y:][0][x:]
            and self._information[y:][0][x:][0] != " "
        ):
            return (255, 255, 255, 255, ord(self._information[y:][0][x:][0]))
        camera_x, camera_y = (x - self._half_width) // 2, y - self._half_height
        for triangle in self._triangles:
            if (camera_x, camera_y) in triangle:
                return triangle.get_pixel(camera_x, camera_y)
        return (255, 255, 255, 255, 32)

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
        move_speed: float = 0.5,
        dash_speed: float = 1.0,
        inertia_ratio: float = 0.5,
        controllable: bool = True,
    ) -> None:
        self._acceleration_x = 0.0
        self._acceleration_y = 0.0
        self._acceleration_z = 0.0
        self._move_acceleration = abs(move_speed / 2.0)
        self._dash_acceleration = abs(dash_speed / 2.0)
        self._inertia_ratio = 1.0 - max(min(inertia_ratio, 1.0), 0.0)
        super(SmoothCamera, self).__init__(
            screen_size=screen_size,
            field_of_view=field_of_view,
            near_plane=near_plane,
            far_plane=far_plane,
            coordinate=coordinate,
            rotation=rotation,
            move_speed=move_speed,
            dash_speed=dash_speed,
            controllable=controllable,
        )

    # Move methods
    def _move_forward(self) -> None:
        self._acceleration_x += self._move_acceleration * self._vector_x
        self._acceleration_z += self._move_acceleration * self._vector_z
        self._acceleration_x = max(
            min(self._acceleration_x, self._move_speed), -self._move_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._move_speed), -self._move_speed
        )

    def _move_backward(self) -> None:
        self._acceleration_x -= self._move_acceleration * self._vector_x
        self._acceleration_z -= self._move_acceleration * self._vector_z
        self._acceleration_x = max(
            min(self._acceleration_x, self._move_speed), -self._move_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._move_speed), -self._move_speed
        )

    def _move_leftward(self) -> None:
        self._acceleration_x -= self._move_acceleration * self._vector_z
        self._acceleration_z += self._move_acceleration * self._vector_x
        self._acceleration_x = max(
            min(self._acceleration_x, self._move_speed), -self._move_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._move_speed), -self._move_speed
        )

    def _move_rightward(self) -> None:
        self._acceleration_x += self._move_acceleration * self._vector_z
        self._acceleration_z -= self._move_acceleration * self._vector_x
        self._acceleration_x = max(
            min(self._acceleration_x, self._move_speed), -self._move_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._move_speed), -self._move_speed
        )

    def _move_upward(self) -> None:
        self._acceleration_y += self._move_acceleration
        self._acceleration_y = max(
            min(self._acceleration_y, self._move_speed), -self._move_speed
        )

    def _move_downward(self) -> None:
        self._acceleration_y -= self._move_acceleration
        self._acceleration_y = max(
            min(self._acceleration_y, self._move_speed), -self._move_speed
        )

    def _dash_forward(self) -> None:
        self._acceleration_x += self._dash_acceleration * self._vector_x
        self._acceleration_z += self._dash_acceleration * self._vector_z
        self._acceleration_x = max(
            min(self._acceleration_x, self._dash_speed), -self._dash_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._dash_speed), -self._dash_speed
        )

    def _dash_backward(self) -> None:
        self._acceleration_x -= self._dash_acceleration * self._vector_x
        self._acceleration_z -= self._dash_acceleration * self._vector_z
        self._acceleration_x = max(
            min(self._acceleration_x, self._dash_speed), -self._dash_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._dash_speed), -self._dash_speed
        )

    def _dash_leftward(self) -> None:
        self._acceleration_x -= self._dash_acceleration * self._vector_z
        self._acceleration_z += self._dash_acceleration * self._vector_x
        self._acceleration_x = max(
            min(self._acceleration_x, self._dash_speed), -self._dash_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._dash_speed), -self._dash_speed
        )

    def _dash_rightward(self) -> None:
        self._acceleration_x += self._dash_acceleration * self._vector_z
        self._acceleration_z -= self._dash_acceleration * self._vector_x
        self._acceleration_x = max(
            min(self._acceleration_x, self._dash_speed), -self._dash_speed
        )
        self._acceleration_z = max(
            min(self._acceleration_z, self._dash_speed), -self._dash_speed
        )

    def _dash_upward(self) -> None:
        self._acceleration_y += self._dash_acceleration
        self._acceleration_y = max(
            min(self._acceleration_y, self._dash_speed), -self._dash_speed
        )

    def _dash_downward(self) -> None:
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
    def _reset(self) -> None:
        self._acceleration_x = 0.0
        self._acceleration_y = 0.0
        self._acceleration_z = 0.0
        super(SmoothCamera, self)._reset()


GRAVITY = 0.1


class PlayerCamera(SmoothCamera):
    def __init__(
        self,
        *,
        screen_size: tuple[int, int],
        field_of_view: float,
        near_plane: float,
        far_plane: float,
        coordinate: Point3DType,
        rotation: RotationType,
        move_speed: float = 0.5,
        dash_speed: float = 1.0,
        inertia_ratio: float = 0.5,
        jump_strength: float = 0.5,
        controllable: bool = True,
    ) -> None:
        self._jump_strength = abs(jump_strength)
        self._gravity = abs(GRAVITY)
        super(PlayerCamera, self).__init__(
            screen_size=screen_size,
            field_of_view=field_of_view,
            near_plane=near_plane,
            far_plane=far_plane,
            coordinate=coordinate,
            rotation=rotation,
            move_speed=move_speed,
            dash_speed=dash_speed,
            inertia_ratio=inertia_ratio,
            controllable=controllable,
        )

    # Move methods
    def _move_upward(self) -> None:
        if self._y <= 0.0:
            self._acceleration_y = self._jump_strength

    def _move_downward(self) -> None:
        pass

    def _dash_upward(self) -> None:
        if self._y <= 0.0:
            self._acceleration_y = self._jump_strength

    def _dash_downward(self) -> None:
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
