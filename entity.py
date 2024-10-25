import math
import sys
import time
import warnings
from pathlib import Path
from threading import Lock

from decoder import PNG
from exception import CameraScreenTooSmallError
from hinting import (
    FrameType,
    Normal3DType,
    Point3DType,
    RotationType,
    RowType,
    ScreenPixelDataType,
    ScreenPoint2DType,
    Texture3DType,
    TriangleNormalsType,
    TriangleTexturesType,
    TriangleVerticesType,
    Vertex3DType,
)
from utility import (
    EMPTY_PIXEL,
    render_mesh_line_no_culling,
    render_mesh_line_backface_culling,
    render_untextured_model,
    render_textured_model,
)


class Object(object):
    def __init__(
        self,
        filepath: str,
        coordinate: Point3DType,
    ) -> None:
        self._filepath = filepath
        self._x, self._y, self._z = coordinate
        self._name = ""
        self._original_triangles: set[
            tuple[
                TriangleVerticesType,
                TriangleTexturesType,
                TriangleNormalsType,
            ]
        ] = set()
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
            vertex_a_x, vertex_a_y, vertex_a_z = vertices[face_vertex_a_index]
            vertex_b_x, vertex_b_y, vertex_b_z = vertices[face_vertex_b_index]
            vertex_c_x, vertex_c_y, vertex_c_z = vertices[face_vertex_c_index]
            texture_a_u, texture_a_v, texture_a_w = textures[face_texture_a_index]
            texture_b_u, texture_b_v, texture_b_w = textures[face_texture_b_index]
            texture_c_u, texture_c_v, texture_c_w = textures[face_texture_c_index]
            normal_a_x, normal_a_y, normal_a_z = normals[face_normal_a_index]
            normal_b_x, normal_b_y, normal_b_z = normals[face_normal_b_index]
            normal_c_x, normal_c_y, normal_c_z = normals[face_normal_c_index]
            self._original_triangles.add(
                (
                    (
                        (vertex_a_x, vertex_a_y, vertex_a_z),
                        (vertex_b_x, vertex_b_y, vertex_b_z),
                        (vertex_c_x, vertex_c_y, vertex_c_z),
                    ),
                    (
                        (texture_a_u, texture_a_v, texture_a_w),
                        (texture_b_u, texture_b_v, texture_b_w),
                        (texture_c_u, texture_c_v, texture_c_w),
                    ),
                    (
                        (normal_a_x, normal_a_y, normal_a_z),
                        (normal_b_x, normal_b_y, normal_b_z),
                        (normal_c_x, normal_c_y, normal_c_z),
                    ),
                )
            )
            self._triangles.add(
                (
                    (
                        (
                            vertex_a_x + self._x,
                            vertex_a_y + self._y,
                            vertex_a_z + self._z,
                        ),
                        (
                            vertex_b_x + self._x,
                            vertex_b_y + self._y,
                            vertex_b_z + self._z,
                        ),
                        (
                            vertex_c_x + self._x,
                            vertex_c_y + self._y,
                            vertex_c_z + self._z,
                        ),
                    ),
                    (
                        (texture_a_u, texture_a_v, texture_a_w),
                        (texture_b_u, texture_b_v, texture_b_w),
                        (texture_c_u, texture_c_v, texture_c_w),
                    ),
                    (
                        (normal_a_x, normal_a_y, normal_a_z),
                        (normal_b_x, normal_b_y, normal_b_z),
                        (normal_c_x, normal_c_y, normal_c_z),
                    ),
                )
            )

    # Properties
    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z

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
        pass

    # Camera-related methods
    def show_to(self, camera: "Camera") -> None:
        camera.show_object(self)

    def hide_from(self, camera: "Camera") -> None:
        camera.hide_object(self)


class RotatingObject(Object):
    def __init__(self, filepath: str, coordinate: Point3DType) -> None:
        super().__init__(filepath, coordinate)

    def update(self, delta_time: float = 0.0) -> None:
        self._triangles: set[
            tuple[
                TriangleVerticesType,
                TriangleTexturesType,
                TriangleNormalsType,
            ]
        ] = set()
        time_factor = time.perf_counter() * 360
        round_per_second = 0.75
        sin_θ = math.sin(math.radians(time_factor * round_per_second))
        cos_θ = math.cos(math.radians(time_factor * round_per_second))
        stroke_offset = math.sin(math.radians(time_factor)) * 2.0
        for vertices, textures, normals in self._original_triangles:
            # Unpacking
            (
                (vertex_a_x, vertex_a_y, vertex_a_z),
                (vertex_b_x, vertex_b_y, vertex_b_z),
                (vertex_c_x, vertex_c_y, vertex_c_z),
            ) = vertices
            # Repacking
            vertices = (
                (
                    vertex_a_x * cos_θ - vertex_a_z * sin_θ + self._x,
                    vertex_a_y + self._y + stroke_offset,
                    vertex_a_x * sin_θ + vertex_a_z * cos_θ + self._z,
                ),
                (
                    vertex_b_x * cos_θ - vertex_b_z * sin_θ + self._x,
                    vertex_b_y + self._y + stroke_offset,
                    vertex_b_x * sin_θ + vertex_b_z * cos_θ + self._z,
                ),
                (
                    vertex_c_x * cos_θ - vertex_c_z * sin_θ + self._x,
                    vertex_c_y + self._y + stroke_offset,
                    vertex_c_x * sin_θ + vertex_c_z * cos_θ + self._z,
                ),
            )
            self._triangles.add((vertices, textures, normals))


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
        self._field_of_view = min(max(60.0, field_of_view), 120.0)
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
                "Then run the `main.py` as root. "
            )
            warnings.warn("using custom KeyboardListener for camera controlling.")
        try:
            import mouse  # type: ignore

            mouse.move(self._screen_width, self._screen_height)  # type: ignore
        except ImportError:
            mouse = None
            warnings.warn(
                "no support for third-party mouse library. "
                "If it's on Linux, install it via `pip install mouse`. "
                "Then run the `main.py` as root. "
            )
            warnings.warn("using custom KeyboardListener for camera controlling.")
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
            time_counter = time.perf_counter()

            def quit(event: KeyboardEvent) -> None:
                self._quit_state = active_states.get(event.event_type, False)

            def reset(event: KeyboardEvent) -> None:
                self._reset_state = active_states.get(event.event_type, False)

            def switch_display_information(event: KeyboardEvent) -> None:
                nonlocal time_counter
                # Timeout using second
                if time.perf_counter() - time_counter > 0.2:
                    time_counter = time.perf_counter()
                    self._display_information_state = (
                        not self._display_information_state
                    )

            def change_render_mode(event: KeyboardEvent) -> None:
                nonlocal time_counter, render_mode_index
                # Timeout using second
                if time.perf_counter() - time_counter > 0.2:
                    time_counter = time.perf_counter()
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

            def rotate(event: ButtonEvent | WheelEvent | MoveEvent) -> None:  # type: ignore
                if not rotate_lock.acquire(blocking=False):
                    return
                while True:
                    if mouse.get_position() != (self._screen_width, self._screen_height):  # type: ignore
                        mouse.move(self._screen_width, self._screen_height)  # type: ignore
                    else:
                        break
                event_type = type(event)  # type: ignore
                if event_type == MoveEvent:
                    self._rotate_view(
                        yaw=-(event.y - self._screen_height) / 72,  # type: ignore
                        pitch=+(event.x - self._screen_width) / 72,  # type: ignore
                    )
                elif event_type == WheelEvent:
                    self._change_fov(delta_fov=event.delta)  # type: ignore
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
            keyboard_listener.register(("\x1b",), (stop_and_quit_legacy,))
            keyboard_listener.register(("r",), (self._reset,))
            # Miscellaneous
            keyboard_listener.register(("i",), (switch_display_information_legacy,))
            keyboard_listener.register(("p",), (change_render_mode_legacy,))
            # Position
            keyboard_listener.register(("W",), (self._dash_forward,))
            keyboard_listener.register(("w",), (self._move_forward,))
            keyboard_listener.register(("S",), (self._dash_backward,))
            keyboard_listener.register(("s",), (self._move_backward,))
            keyboard_listener.register(("A",), (self._dash_leftward,))
            keyboard_listener.register(("a",), (self._move_leftward,))
            keyboard_listener.register(("D",), (self._dash_rightward,))
            keyboard_listener.register(("d",), (self._move_rightward,))
            keyboard_listener.register((" ",), (self._move_upward,))
            keyboard_listener.register(("\r",), (self._move_downward,))
            # Rotation
            keyboard_listener.register(("8",), (self._yaw_forward,))
            keyboard_listener.register(("2",), (self._yaw_reverse,))
            keyboard_listener.register(("6",), (self._pitch_forward,))
            keyboard_listener.register(("4",), (self._pitch_reverse,))
            # View
            keyboard_listener.register(("+",), (self._fov_increase,))
            keyboard_listener.register(("-",), (self._fov_decrease,))

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

    def _rotate_view(
        self,
        *,
        yaw: float = 0.0,
        pitch: float = 0.0,
    ) -> None:
        self._yaw += yaw
        self._pitch += pitch
        self._yaw = max(min(self._yaw, 90.0), -90.0)
        self._pitch %= 360.0

    # View methods
    def _fov_increase(self) -> None:
        self._field_of_view += 1.0
        self._field_of_view = min(max(60.0, self._field_of_view), 120.0)
        self._focal = (
            max(self._screen_width, self._screen_height)
            / math.tan(math.radians(self._field_of_view / 2.0))
            / 2
        )

    def _fov_decrease(self) -> None:
        self._field_of_view -= 1.0
        self._field_of_view = min(max(60.0, self._field_of_view), 120.0)
        self._focal = (
            max(self._screen_width, self._screen_height)
            / math.tan(math.radians(self._field_of_view / 2.0))
            / 2
        )

    def _change_fov(self, *, delta_fov: float) -> None:
        self._field_of_view += delta_fov
        self._field_of_view = min(max(60.0, self._field_of_view), 120.0)
        self._focal = (
            max(self._screen_width, self._screen_height)
            / math.tan(math.radians(self._field_of_view / 2.0))
            / 2
        )

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
        for obj in sorted(
            self._objects,
            key=lambda obj: (
                (self._x - obj.x) ** 2 + (self._y - obj.y) ** 2 + (self._y - obj.y) ** 2
            )
            ** 0.5,
        ):
            if obj.texture is None:
                texture_image = None
                texture_size = None
            else:
                texture_image = obj.texture.image_data
                texture_width, texture_height = obj.texture.image_size
                texture_size = (texture_width - 1, texture_height - 1)
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
                self._selected_render_function(
                    vertex_texture_normal_a,
                    vertex_texture_normal_b,
                    vertex_texture_normal_c,
                    self._frustum_border,
                    texture_image,
                    texture_size,
                    self._pixels,
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
                    # Ignore the Z-depth value
                    _, *pixel = self._pixels.get(
                        ((x - self._half_width) // 2, y - self._half_height),
                        (0, 255, 255, 255, 255, EMPTY_PIXEL),
                    )
                    pixel = tuple(pixel)
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
