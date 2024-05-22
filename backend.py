import math
import os
import random
import string
import sys
import warnings
from datetime import datetime

from decoder import PNGSequence
from hintings import FrameType, FramesType, RowType, EffectModeType
from utilities import Triangle

ASCII_CHARACTERS = "".join((string.digits, string.ascii_letters, string.punctuation))
BINARY_CHARACTERS = "01" * 32
PIXEL_DENSITY_SHORT_CHARACTERS = "@%#*+=-:. "
PIXEL_DENSITY_STANDARD_CHARACTERS = (
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
)
PIXEL_DENSITY_LONG_CHARACTERS = "@MBHENR#KWXDFPQASUZbdehx*8Gm&04LOVYkpq5Tagns69owz$CIu23Jcfry%1v7l+it[]{}?j|()=~!-/<>\"^_';,:`."
ASCII_CHARACTERS_LENGTH = len(ASCII_CHARACTERS) - 1
BINARY_CHARACTERS_LENGTH = len(BINARY_CHARACTERS) - 1
PIXEL_DENSITY_SHORT_CHARACTERS_LENGTH = len(PIXEL_DENSITY_SHORT_CHARACTERS) - 1
PIXEL_DENSITY_STANDARD_CHARACTERS_LENGTH = len(PIXEL_DENSITY_STANDARD_CHARACTERS) - 1
PIXEL_DENSITY_LONG_CHARACTERS_LENGTH = len(PIXEL_DENSITY_LONG_CHARACTERS) - 1


class InvalidEffectModeError(Exception):
    pass


class Backend(object):
    def __init__(self) -> None:
        raise NotImplementedError

    @property
    def frames(self) -> FramesType:
        raise NotImplementedError


class TheMatrixCodeRain(Backend):
    RAIN_COLOR = (53, 255, 86, 255)
    SKY_COLOR = (12, 12, 12, 255)

    def __init__(self, mode: EffectModeType = "long") -> None:
        self._terminal_width = -1
        self._terminal_height = -1
        if mode == "ascii":
            self._code_rain_sequence = ASCII_CHARACTERS
            self._code_rain_sequence_length = ASCII_CHARACTERS_LENGTH
        elif mode == "binary":
            self._code_rain_sequence = BINARY_CHARACTERS
            self._code_rain_sequence_length = BINARY_CHARACTERS_LENGTH
        elif mode == "short":
            self._code_rain_sequence = PIXEL_DENSITY_SHORT_CHARACTERS
            self._code_rain_sequence_length = PIXEL_DENSITY_SHORT_CHARACTERS_LENGTH
        elif mode == "standard":
            self._code_rain_sequence = PIXEL_DENSITY_STANDARD_CHARACTERS
            self._code_rain_sequence_length = PIXEL_DENSITY_STANDARD_CHARACTERS_LENGTH
        elif mode == "long":
            self._code_rain_sequence = PIXEL_DENSITY_LONG_CHARACTERS
            self._code_rain_sequence_length = PIXEL_DENSITY_LONG_CHARACTERS_LENGTH
        else:
            raise InvalidEffectModeError('unknown code rain mode - "%s"')

    def _spawn_code_rain(self, terminal_height: int) -> tuple[int, int, int, str]:
        length = random.randint(terminal_height // 3, terminal_height)
        offset = random.randint(0, terminal_height - length)
        iterations = 0
        characters = "".join(
            self._code_rain_sequence[index * self._code_rain_sequence_length // length]
            for index in range(0, length)
        )
        return length, offset, iterations, characters

    def _step_code_rain(self, x_position: int) -> None:
        length, offset, iterations, characters = self._code_rains[x_position]
        self._code_rains[x_position] = (
            length,
            offset,
            iterations + 1,
            characters,
        )

    @property
    def frames(self) -> FramesType:
        while True:
            terminal_width, terminal_height = os.get_terminal_size()
            if (
                terminal_width != self._terminal_width
                or terminal_height != self._terminal_height
            ):
                self._terminal_width = terminal_width
                self._terminal_height = terminal_height
                self._code_rains = dict(
                    (
                        x_position,
                        self._spawn_code_rain(self._terminal_height),
                    )
                    for x_position in range(0, self._terminal_width)
                )
            frame_buffer: FrameType = []
            for y in range(0, terminal_height):
                row_buffer: RowType = []
                for x in range(0, terminal_width):
                    fr, fg, fb, fa = TheMatrixCodeRain.SKY_COLOR
                    c = ord(" ")
                    length, offset, iterations, characters = self._code_rains[x]
                    if offset <= y < offset + length:
                        step = offset + length - y
                        double_length = length * 2
                        progress = iterations // length
                        if 0 <= progress < length:
                            character = characters[-progress - 1 :].ljust(length)[
                                y - offset
                            ]
                            self._step_code_rain(x)
                        elif length <= progress < double_length:
                            character = characters[
                                : double_length - progress - 1
                            ].rjust(length)[y - offset]
                            self._step_code_rain(x)
                        else:
                            character = " "
                            self._code_rains[x] = self._spawn_code_rain(terminal_height)
                        fr, fg, fb, fa = TheMatrixCodeRain.RAIN_COLOR
                        fr = int(fr * ((step / length) ** (1 / 3)))
                        fg = int(fg * ((step / length) ** (1 / 3)))
                        fb = int(fb * ((step / length) ** (1 / 3)))
                        c = ord(character)
                    row_buffer.append((fr, fg, fb, fa, c))
                frame_buffer.append(row_buffer)
            yield frame_buffer


class DigitalTimeUnit(Backend):
    FG_COLOR = (255, 255, 255, 255)
    BG_COLOR = (12, 12, 12, 255)

    def __init__(self) -> None:
        self._images_size, self._images_data = zip(
            *(
                (image.image_size, image.image_data)
                for image in PNGSequence("resource/digits")
            )
        )

    @property
    def frames(self) -> FramesType:
        while True:
            terminal_width, terminal_height = os.get_terminal_size()
            current_time = datetime.now()
            time_string = "%02d:%02d:%02d" % (
                current_time.hour,
                current_time.minute,
                current_time.second,
            )
            time_string_length = len(time_string)
            canvas_width, canvas_height = (
                terminal_width * 80 // 100,
                terminal_height * 40 // 100,
            )
            canvas_left_border = (terminal_width - canvas_width) // 2
            canvas_right_border = terminal_width - canvas_left_border
            canvas_top_border = (terminal_height - canvas_height) // 2
            canvas_bottom_border = terminal_height - canvas_top_border
            unit_width, unit_height = (
                canvas_width // time_string_length,
                canvas_height,
            )
            indexed_unit_borders = dict(
                zip(
                    range(0, time_string_length),
                    (
                        (
                            canvas_left_border + i * unit_width,
                            canvas_left_border + (i + 1) * unit_width,
                            canvas_top_border,
                            canvas_top_border + unit_height,
                        )
                        for i in range(0, time_string_length)
                    ),
                ),
            )
            frame_buffer: FrameType = []
            for y in range(0, terminal_height):
                row_buffer: RowType = []
                for x in range(0, terminal_width):
                    fr, fg, fb, fa = DigitalTimeUnit.BG_COLOR
                    c = ord(" ")
                    if (
                        canvas_top_border <= y < canvas_bottom_border
                        and canvas_left_border <= x < canvas_right_border
                    ):
                        for index, (
                            left,
                            right,
                            top,
                            bottom,
                        ) in indexed_unit_borders.items():
                            if top <= y < bottom and left <= x < right:
                                character = time_string[index]
                                break
                        else:
                            index = left = right = -1
                            character = ""
                        if character:
                            if character.isdecimal():
                                character_index = int(character)
                            else:
                                character_index = 10
                            character_width, character_height = self._images_size[
                                character_index
                            ]
                            _, _, _, a = self._images_data[character_index][
                                (y - canvas_top_border)
                                * character_height
                                // unit_height
                            ][(x - left) * character_width // unit_width]
                            if a > 0:
                                fr, fg, fb, fa = DigitalTimeUnit.FG_COLOR
                                c = ord(character)
                    row_buffer.append((fr, fg, fb, fa, c))
                frame_buffer.append(row_buffer)
            yield frame_buffer


class Camera(object):
    def __init__(
        self,
        fov: float,
        coordinate: tuple[float, float, float],
        rotation: tuple[float, float, float],
    ) -> None:
        self._fov = fov
        self._x, self._y, self._z = coordinate
        self._original_coordinate = coordinate
        self._yaw, self._pitch, self._roll = rotation
        self._original_rotation = rotation
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
        self._x += self._vector_x
        self._z += self._vector_z

    def move_backward(self) -> None:
        self._x -= self._vector_x
        self._z -= self._vector_z

    def move_leftward(self) -> None:
        self._x -= self._vector_z
        self._z += self._vector_x

    def move_rightward(self) -> None:
        self._x += self._vector_z
        self._z -= self._vector_x

    def move_upward(self) -> None:
        self._y += 1.0

    def move_downward(self) -> None:
        self._y -= 1.0

    def dash_forward(self) -> None:
        self._x += 2.0 * self._vector_x
        self._z += 2.0 * self._vector_z

    def dash_backward(self) -> None:
        self._x -= 2.0 * self._vector_x
        self._z -= 2.0 * self._vector_z

    def dash_leftward(self) -> None:
        self._x -= 2.0 * self._vector_z
        self._z += 2.0 * self._vector_x

    def dash_rightward(self) -> None:
        self._x += 2.0 * self._vector_z
        self._z -= 2.0 * self._vector_x

    def dash_up(self) -> None:
        self._y += 2.0

    def dash_down(self) -> None:
        self._y -= 2.0

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

    # Trigonometrics and vector update methods
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


class Fake3DSceneGame(Backend):
    def __init__(self) -> None:
        try:
            import keyboard

            self._keyboard = keyboard
        except ImportError:
            self._keyboard = None
            warnings.warn(
                "no third-party support for keyboard. Using custom KeyboardListener."
            )
        try:
            import mouse  # type: ignore

            self._mouse = mouse
        except ImportError:
            self._mouse = None
            warnings.warn(
                "no third-party support for mouse. Using custom KeyboardListener."
            )
        if self._keyboard is None or self._mouse is None:
            from controller import KeyboardListener

            self._keyboard_listener = KeyboardListener()
        else:
            self._keyboard_listener = None

    def _update_triangles(self):
        # Triangles
        self._triangles: list[Triangle] = []
        # Camera properties
        camera_fov = self._camera.fov
        camera_x, camera_y, camera_z = self._camera.coordinate
        (
            camera_sin_yaw,
            camera_cos_yaw,
            camera_sin_pitch,
            camera_cos_pitch,
            camera_sin_roll,
            camera_cos_roll,
        ) = self._camera.trigonometrics
        # Iteration over all triangles
        # Assuming that every triangle is ▲abc
        for triangle_vertices in self._triangle_vertices:
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
                triangle_a_x - camera_x,
                triangle_a_y - camera_y,
                triangle_a_z - camera_z,
            )
            (
                distance_camera_triangle_b_x,
                distance_camera_triangle_b_y,
                distance_camera_triangle_b_z,
            ) = (
                triangle_b_x - camera_x,
                triangle_b_y - camera_y,
                triangle_b_z - camera_z,
            )
            (
                distance_camera_triangle_c_x,
                distance_camera_triangle_c_y,
                distance_camera_triangle_c_z,
            ) = (
                triangle_c_x - camera_x,
                triangle_c_y - camera_y,
                triangle_c_z - camera_z,
            )
            # Rotation
            # Z-axis rotation that affects X/Y coordinates
            distance_camera_triangle_a_x, distance_camera_triangle_a_y = (
                distance_camera_triangle_a_x * camera_cos_roll
                + distance_camera_triangle_a_y * camera_sin_roll,
                -distance_camera_triangle_a_x * camera_sin_roll
                + distance_camera_triangle_a_y * camera_cos_roll,
            )
            distance_camera_triangle_b_x, distance_camera_triangle_b_y = (
                distance_camera_triangle_b_x * camera_cos_roll
                + distance_camera_triangle_b_y * camera_sin_roll,
                -distance_camera_triangle_b_x * camera_sin_roll
                + distance_camera_triangle_b_y * camera_cos_roll,
            )
            distance_camera_triangle_c_x, distance_camera_triangle_c_y = (
                distance_camera_triangle_c_x * camera_cos_roll
                + distance_camera_triangle_c_y * camera_sin_roll,
                -distance_camera_triangle_c_x * camera_sin_roll
                + distance_camera_triangle_c_y * camera_cos_roll,
            )
            # Y-axis rotation that affects X/Z coordinates
            distance_camera_triangle_a_x, distance_camera_triangle_a_z = (
                distance_camera_triangle_a_x * camera_cos_pitch
                - distance_camera_triangle_a_z * camera_sin_pitch,
                distance_camera_triangle_a_x * camera_sin_pitch
                + distance_camera_triangle_a_z * camera_cos_pitch,
            )
            distance_camera_triangle_b_x, distance_camera_triangle_b_z = (
                distance_camera_triangle_b_x * camera_cos_pitch
                - distance_camera_triangle_b_z * camera_sin_pitch,
                distance_camera_triangle_b_x * camera_sin_pitch
                + distance_camera_triangle_b_z * camera_cos_pitch,
            )
            distance_camera_triangle_c_x, distance_camera_triangle_c_z = (
                distance_camera_triangle_c_x * camera_cos_pitch
                - distance_camera_triangle_c_z * camera_sin_pitch,
                distance_camera_triangle_c_x * camera_sin_pitch
                + distance_camera_triangle_c_z * camera_cos_pitch,
            )
            # X-axis rotation that affects Y/Z coordinates
            distance_camera_triangle_a_y, distance_camera_triangle_a_z = (
                distance_camera_triangle_a_y * camera_cos_yaw
                + distance_camera_triangle_a_z * camera_sin_yaw,
                -distance_camera_triangle_a_y * camera_sin_yaw
                + distance_camera_triangle_a_z * camera_cos_yaw,
            )
            distance_camera_triangle_b_y, distance_camera_triangle_b_z = (
                distance_camera_triangle_b_y * camera_cos_yaw
                + distance_camera_triangle_b_z * camera_sin_yaw,
                -distance_camera_triangle_b_y * camera_sin_yaw
                + distance_camera_triangle_b_z * camera_cos_yaw,
            )
            distance_camera_triangle_c_y, distance_camera_triangle_c_z = (
                distance_camera_triangle_c_y * camera_cos_yaw
                + distance_camera_triangle_c_z * camera_sin_yaw,
                -distance_camera_triangle_c_y * camera_sin_yaw
                + distance_camera_triangle_c_z * camera_cos_yaw,
            )
            # Simple culling. TODO: advanced culling mechanism.
            if (
                distance_camera_triangle_a_z <= 0.0
                or distance_camera_triangle_b_z <= 0.0
                or distance_camera_triangle_c_z <= 0.0
            ):
                self._triangles.append(Triangle(null=True))
            else:
                self._triangles.append(
                    Triangle(
                        vertex_a=(
                            (distance_camera_triangle_a_x)
                            * camera_fov
                            / (distance_camera_triangle_a_z),
                            (distance_camera_triangle_a_y)
                            * camera_fov
                            / (distance_camera_triangle_a_z),
                        ),
                        vertex_b=(
                            (distance_camera_triangle_b_x)
                            * camera_fov
                            / (distance_camera_triangle_b_z),
                            (distance_camera_triangle_b_y)
                            * camera_fov
                            / (distance_camera_triangle_b_z),
                        ),
                        vertex_c=(
                            (distance_camera_triangle_c_x)
                            * camera_fov
                            / (distance_camera_triangle_c_z),
                            (distance_camera_triangle_c_y)
                            * camera_fov
                            / (distance_camera_triangle_c_z),
                        ),
                    )
                )

    @property
    def frames(self) -> FramesType:
        self._camera = Camera(15, (0.0, 0.0, -75.0), (0.0, 0.0, 0.0))
        self._triangle_vertices = [
            ((-25.0, -25.0, -25.0), (-25.0, 25.0, -25.0), (25.0, -25.0, -25.0)),  # ◣
            ((25.0, 25.0, -25.0), (25.0, -25.0, -25.0), (-25.0, 25.0, -25.0)),  # ◥
            ((-25.0, -25.0, 25.0), (-25.0, 25.0, 25.0), (25.0, -25.0, 25.0)),  # ◺
            ((25.0, 25.0, 25.0), (25.0, -25.0, 25.0), (-25.0, 25.0, 25.0)),  # ◹
        ]
        camera_info_length = len(self._camera.info)
        self._update_triangles()
        while True:
            screen_width, screen_height = os.get_terminal_size()
            screen_width = (screen_width // 2) * 2 or 1
            screen_height = screen_height - camera_info_length or 1
            half_width, half_height = screen_width / 2, screen_height / 2

            # Camera controlling
            # Third-party modules
            # Position
            if self._keyboard is not None:
                # Forward
                if self._keyboard.is_pressed("shift+w"):
                    self._camera.dash_forward()
                elif self._keyboard.is_pressed("w"):
                    self._camera.move_forward()
                # Backward
                if self._keyboard.is_pressed("shift+s"):
                    self._camera.dash_backward()
                elif self._keyboard.is_pressed("s"):
                    self._camera.move_backward()
                # Leftward
                if self._keyboard.is_pressed("shift+a"):
                    self._camera.dash_leftward()
                elif self._keyboard.is_pressed("a"):
                    self._camera.move_leftward()
                # Rightward
                if self._keyboard.is_pressed("shift+d"):
                    self._camera.dash_rightward()
                elif self._keyboard.is_pressed("d"):
                    self._camera.move_rightward()
                # Upward
                if self._keyboard.is_pressed("shift+space"):
                    self._camera.dash_up()
                elif self._keyboard.is_pressed("space"):
                    self._camera.move_upward()
                # Downward
                if self._keyboard.is_pressed("ctrl+shift"):
                    self._camera.dash_down()
                elif self._keyboard.is_pressed("ctrl"):
                    self._camera.move_downward()
                # Reset
                if self._keyboard.is_pressed("r"):
                    self._camera.reset()
                # Exit
                if self._keyboard.is_pressed("escape"):
                    sys.exit(0)
            # Rotation
            if self._mouse is not None:
                # Yaw/Pitch
                mouse_x, mouse_y = self._mouse.get_position()
                if (mouse_x, mouse_y) != (960, 540):
                    self._mouse.move(960, 540)  # type: ignore
                    self._camera.rotate(
                        yaw=-(mouse_y - 540) / 18,
                        pitch=+(mouse_x - 960) / 18,
                    )
            # Custom `KeyboardListener`
            if self._keyboard_listener is not None:
                key = self._keyboard_listener.get()
                # Position
                if self._keyboard is None:
                    # Forward
                    if key == "W":
                        self._camera.dash_forward()
                    elif key == "w":
                        self._camera.move_forward()
                    # Backward
                    elif key == "S":
                        self._camera.dash_backward()
                    elif key == "s":
                        self._camera.move_backward()
                    # Leftward
                    elif key == "A":
                        self._camera.dash_leftward()
                    elif key == "a":
                        self._camera.move_leftward()
                    # Rightward
                    elif key == "D":
                        self._camera.dash_rightward()
                    elif key == "d":
                        self._camera.move_rightward()
                    # Upward
                    elif key == " ":
                        self._camera.move_upward()
                    # Downward
                    elif key == "\r":
                        self._camera.move_downward()
                    # Reset
                    elif key == "r":
                        self._camera.reset()
                    # Exit
                    elif key == "\x1b":
                        self._keyboard_listener.stop()
                        sys.exit(0)
                # Rotation
                if self._mouse is None:
                    # Yaw
                    if key == "8":
                        self._camera.rotate(yaw=+1.0, pitch=0.0, roll=0.0)
                    elif key == "2":
                        self._camera.rotate(yaw=-1.0, pitch=0.0, roll=0.0)
                    # Pitch
                    elif key == "4":
                        self._camera.rotate(yaw=0.0, pitch=-1.0, roll=0.0)
                    elif key == "6":
                        self._camera.rotate(yaw=0.0, pitch=+1.0, roll=0.0)
                    # Roll
                    elif key == "e":
                        self._camera.rotate(yaw=0.0, pitch=0.0, roll=+1.0)
                    elif key == "q":
                        self._camera.rotate(yaw=0.0, pitch=0.0, roll=-1.0)

            # Frame generation
            # Camera view
            frame_buffer: FrameType = []
            for y in range(0, screen_height):
                row_buffer: RowType = []
                for x in range(0, screen_width):
                    vertex_x, vertex_y = (x - half_width) / 2, y - half_height
                    if any(
                        (vertex_x, vertex_y) in triangle for triangle in self._triangles
                    ):
                        character = "█"
                    else:
                        character = " "
                    row_buffer.append((255, 255, 255, 255, ord(character)))
                frame_buffer.insert(0, row_buffer)
            # Camera info
            for camera_info in self._camera.info:
                row_buffer: RowType = []
                for character in camera_info.ljust(screen_width)[:screen_width]:
                    row_buffer.append((255, 255, 255, 255, ord(character)))
                frame_buffer.append(row_buffer)

            yield frame_buffer
