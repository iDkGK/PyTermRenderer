import os
import random
import string
import sys
import warnings
from datetime import datetime

from decoder import PNGSequence
from hintings import FrameType, FramesType, RowType, EffectModeType
from utilities import Triangle, PlayerCamera

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

    @property
    def frames(self) -> FramesType:
        self._player_camera = PlayerCamera(
            fov=90,
            view=os.get_terminal_size(),
            coordinate=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
        )
        self._triangle_vertices = [
            ((-25.0, -25.0, -25.0), (-25.0, 25.0, -25.0), (25.0, -25.0, -25.0)),  # ◣
            ((25.0, 25.0, -25.0), (25.0, -25.0, -25.0), (-25.0, 25.0, -25.0)),  # ◥
            ((-25.0, -25.0, 25.0), (-25.0, 25.0, 25.0), (25.0, -25.0, 25.0)),  # ◺
            ((25.0, 25.0, 25.0), (25.0, -25.0, 25.0), (-25.0, 25.0, 25.0)),  # ◹
        ]
        camera_info_length = len(self._player_camera.info)
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
                    self._player_camera.dash_forward()
                elif self._keyboard.is_pressed("w"):
                    self._player_camera.move_forward()
                # Backward
                if self._keyboard.is_pressed("shift+s"):
                    self._player_camera.dash_backward()
                elif self._keyboard.is_pressed("s"):
                    self._player_camera.move_backward()
                # Leftward
                if self._keyboard.is_pressed("shift+a"):
                    self._player_camera.dash_leftward()
                elif self._keyboard.is_pressed("a"):
                    self._player_camera.move_leftward()
                # Rightward
                if self._keyboard.is_pressed("shift+d"):
                    self._player_camera.dash_rightward()
                elif self._keyboard.is_pressed("d"):
                    self._player_camera.move_rightward()
                # Upward
                if self._keyboard.is_pressed("shift+space"):
                    self._player_camera.dash_upward()
                elif self._keyboard.is_pressed("space"):
                    self._player_camera.move_upward()
                # Downward
                if self._keyboard.is_pressed("ctrl+shift"):
                    self._player_camera.dash_downward()
                elif self._keyboard.is_pressed("ctrl"):
                    self._player_camera.move_downward()
                # Reset
                if self._keyboard.is_pressed("r"):
                    self._player_camera.reset()
                # Exit
                if self._keyboard.is_pressed("escape"):
                    sys.exit(0)
            # Rotation
            if self._mouse is not None:
                # Yaw/Pitch
                mouse_x, mouse_y = self._mouse.get_position()
                if mouse_x != 960 or mouse_y != 540:
                    self._mouse.move(960, 540)  # type: ignore
                    self._player_camera.rotate(
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
                        self._player_camera.dash_forward()
                    elif key == "w":
                        self._player_camera.move_forward()
                    # Backward
                    elif key == "S":
                        self._player_camera.dash_backward()
                    elif key == "s":
                        self._player_camera.move_backward()
                    # Leftward
                    elif key == "A":
                        self._player_camera.dash_leftward()
                    elif key == "a":
                        self._player_camera.move_leftward()
                    # Rightward
                    elif key == "D":
                        self._player_camera.dash_rightward()
                    elif key == "d":
                        self._player_camera.move_rightward()
                    # Upward
                    elif key == " ":
                        self._player_camera.move_upward()
                    # Downward
                    elif key == "\r":
                        self._player_camera.move_downward()
                    # Reset
                    elif key == "r":
                        self._player_camera.reset()
                    # Exit
                    elif key == "\x1b":
                        self._keyboard_listener.stop()
                        sys.exit(0)
                # Rotation
                if self._mouse is None:
                    # Yaw
                    if key == "8":
                        self._player_camera.rotate(yaw=+1.0, pitch=0.0, roll=0.0)
                    elif key == "2":
                        self._player_camera.rotate(yaw=-1.0, pitch=0.0, roll=0.0)
                    # Pitch
                    elif key == "4":
                        self._player_camera.rotate(yaw=0.0, pitch=-1.0, roll=0.0)
                    elif key == "6":
                        self._player_camera.rotate(yaw=0.0, pitch=+1.0, roll=0.0)
                    # Roll
                    elif key == "e":
                        self._player_camera.rotate(yaw=0.0, pitch=0.0, roll=+1.0)
                    elif key == "q":
                        self._player_camera.rotate(yaw=0.0, pitch=0.0, roll=-1.0)
            self._player_camera.update()
            self._update_triangles()

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
            for camera_info in self._player_camera.info:
                row_buffer: RowType = []
                for character in camera_info.ljust(screen_width)[:screen_width]:
                    row_buffer.append((255, 255, 255, 255, ord(character)))
                frame_buffer.append(row_buffer)

            yield frame_buffer

    def _update_triangles(self):
        # Triangles
        self._triangles: list[Triangle] = []
        # Camera properties
        camera_focal = self._player_camera.focal
        camera_x, camera_y, camera_z = (
            self._player_camera.x,
            self._player_camera.y,
            self._player_camera.z,
        )
        (
            camera_sin_yaw,
            camera_cos_yaw,
            camera_sin_pitch,
            camera_cos_pitch,
            camera_sin_roll,
            camera_cos_roll,
        ) = self._player_camera.trigonometrics
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
                            camera_focal
                            * distance_camera_triangle_a_x
                            / distance_camera_triangle_a_z,
                            camera_focal
                            * distance_camera_triangle_a_y
                            / distance_camera_triangle_a_z,
                        ),
                        vertex_b=(
                            camera_focal
                            * distance_camera_triangle_b_x
                            / distance_camera_triangle_b_z,
                            camera_focal
                            * distance_camera_triangle_b_y
                            / distance_camera_triangle_b_z,
                        ),
                        vertex_c=(
                            camera_focal
                            * distance_camera_triangle_c_x
                            / distance_camera_triangle_c_z,
                            camera_focal
                            * distance_camera_triangle_c_y
                            / distance_camera_triangle_c_z,
                        ),
                    )
                )
