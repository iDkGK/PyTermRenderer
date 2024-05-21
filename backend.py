import math
import os
import random
import string
import sys
from datetime import datetime

from controller import KeyboardListener
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


class Fake3DSceneGame(Backend):

    def __init__(self) -> None:
        screen_width, screen_height = os.get_terminal_size()
        # Width:height of single character of terminal is about 1:2
        # So we make width twice as its original value
        screen_width, screen_height = screen_width // 2 or 1, screen_height or 1
        self._keyboard_listener = KeyboardListener()
        self._camera_fov = screen_height * screen_height / screen_width
        self._camera_coordinate = [0.0, 0.0, 0.0]  # camera coordinate
        self._camera_rotation = [0.0, 0.0, 0.0]  # Camera X/Y/Z rotations in degrees
        self._triangle_vertices = [
            ((-25.0, -25.0, 50.0), (-25.0, 25.0, 50.0), (25.0, -25.0, 50.0)),  # ◣
            ((25.0, 25.0, 50.0), (25.0, -25.0, 50.0), (-25.0, 25.0, 50.0)),  # ◥
            ((-25.0, -25.0, 100.0), (-25.0, 25.0, 100.0), (25.0, -25.0, 100.0)),  # ◺
            ((25.0, 25.0, 100.0), (25.0, -25.0, 100.0), (-25.0, 25.0, 100.0)),  # ◹
        ]

    @property
    def frames(self) -> FramesType:
        self._triangles: list[Triangle] = []
        for triangle_vertices in self._triangle_vertices:
            x_p, y_p, z_p = self._camera_coordinate
            xd_p, yd_p, zd_p = self._camera_rotation
            (x_a, y_a, z_a), (x_b, y_b, z_b), (x_c, y_c, z_c) = triangle_vertices
            # Position
            # Using vector for relative position
            vx_ap, vy_ap, vz_ap = x_a - x_p, y_a - y_p, z_a - z_p
            vx_bp, vy_bp, vz_bp = x_b - x_p, y_b - y_p, z_b - z_p
            vx_cp, vy_cp, vz_cp = x_c - x_p, y_c - y_p, z_c - z_p
            # Rotation
            # Trigonometric values of X/Y/Z rotations
            xr_p, yr_p, zr_p = (
                math.radians(-xd_p),
                math.radians(yd_p),
                math.radians(zd_p),
            )
            xs_p, xc_p, ys_p, yc_p, zs_p, zc_p = (
                math.sin(xr_p),
                math.cos(xr_p),
                math.sin(yr_p),
                math.cos(yr_p),
                math.sin(zr_p),
                math.cos(zr_p),
            )
            # X-axis rotation that affects Y/Z coordinates
            vy_ap, vz_ap = vy_ap * xc_p + vz_ap * xs_p, vz_ap * xc_p - vy_ap * xs_p
            vy_bp, vz_bp = vy_bp * xc_p + vz_bp * xs_p, vz_bp * xc_p - vy_bp * xs_p
            vy_cp, vz_cp = vy_cp * xc_p + vz_cp * xs_p, vz_cp * xc_p - vy_cp * xs_p
            # Y-axis rotation that affects X/Z coordinates
            vx_ap, vz_ap = vx_ap * yc_p - vz_ap * ys_p, vx_ap * ys_p + vz_ap * yc_p
            vx_bp, vz_bp = vx_bp * yc_p - vz_bp * ys_p, vx_bp * ys_p + vz_bp * yc_p
            vx_cp, vz_cp = vx_cp * yc_p - vz_cp * ys_p, vx_cp * ys_p + vz_cp * yc_p
            # Z-axis rotation that affects X/Y coordinates
            vx_ap, vy_ap = vx_ap * zc_p - vy_ap * zs_p, vx_ap * zs_p + vy_ap * zc_p
            vx_bp, vy_bp = vx_bp * zc_p - vy_bp * zs_p, vx_bp * zs_p + vy_bp * zc_p
            vx_cp, vy_cp = vx_cp * zc_p - vy_cp * zs_p, vx_cp * zs_p + vy_cp * zc_p
            # Simple culling. TODO: advanced culling mechanism.
            if vz_ap <= 0.0 or vz_bp <= 0.0 or vz_cp <= 0.0:
                self._triangles.append(Triangle(null=True))
            else:
                self._triangles.append(
                    Triangle(
                        vertex_a=(
                            (vx_ap) * self._camera_fov / (vz_ap),
                            (vy_ap) * self._camera_fov / (vz_ap),
                        ),
                        vertex_b=(
                            (vx_bp) * self._camera_fov / (vz_bp),
                            (vy_bp) * self._camera_fov / (vz_bp),
                        ),
                        vertex_c=(
                            (vx_cp) * self._camera_fov / (vz_cp),
                            (vy_cp) * self._camera_fov / (vz_cp),
                        ),
                    )
                )
        while True:
            screen_width, screen_height = os.get_terminal_size()
            screen_width, screen_height = screen_width // 2 or 1, screen_height or 1
            half_width, half_height = (screen_width // 2, screen_height // 2)
            key = self._keyboard_listener.get()
            frame_buffer: FrameType = []
            # Position
            xd_p, yd_p, zd_p = self._camera_rotation
            # Trigonometric values of X/Y/Z rotations
            xr_p, yr_p, zr_p = (
                math.radians(-xd_p),
                math.radians(yd_p),
                math.radians(zd_p),
            )
            xs_p, xc_p, ys_p, yc_p, zs_p, zc_p = (
                math.sin(xr_p),
                math.cos(xr_p),
                math.sin(yr_p),
                math.cos(yr_p),
                math.sin(zr_p),
                math.cos(zr_p),
            )
            cv_x, cv_y, cv_z = (
                xc_p * ys_p * zc_p + xs_p * zs_p,
                xs_p * ys_p * zc_p - xc_p * zs_p,
                yc_p * zc_p,
            )
            if key == "w":
                self._camera_coordinate[0] += 2.0 * cv_x
                self._camera_coordinate[1] += 2.0 * cv_y
                self._camera_coordinate[2] += 2.0 * cv_z
            elif key == "s":
                self._camera_coordinate[0] -= 2.0 * cv_x
                self._camera_coordinate[1] -= 2.0 * cv_y
                self._camera_coordinate[2] -= 2.0 * cv_z
            elif key == "a":
                self._camera_coordinate[0] -= 2.0 * cv_z
                self._camera_coordinate[1] -= 2.0 * cv_y
                self._camera_coordinate[2] += 2.0 * cv_x
            elif key == "d":
                self._camera_coordinate[0] += 2.0 * cv_z
                self._camera_coordinate[1] += 2.0 * cv_y
                self._camera_coordinate[2] -= 2.0 * cv_x
            elif key == "W":
                self._camera_coordinate[0] += 8.0 * cv_x
                self._camera_coordinate[1] += 8.0 * cv_y
                self._camera_coordinate[2] += 8.0 * cv_z
            elif key == "S":
                self._camera_coordinate[0] -= 8.0 * cv_x
                self._camera_coordinate[1] -= 8.0 * cv_y
                self._camera_coordinate[2] -= 8.0 * cv_z
            elif key == "A":
                self._camera_coordinate[0] -= 8.0 * cv_z
                self._camera_coordinate[1] -= 8.0 * cv_y
                self._camera_coordinate[2] += 8.0 * cv_x
            elif key == "D":
                self._camera_coordinate[0] += 8.0 * cv_z
                self._camera_coordinate[1] += 8.0 * cv_y
                self._camera_coordinate[2] -= 8.0 * cv_x
            elif key == " ":
                self._camera_coordinate[1] += 2.0
            elif key == "\r":
                self._camera_coordinate[1] -= 2.0
            # Rotation
            elif key == "8":
                self._camera_rotation[0] += 1.0
                self._camera_rotation[0] = max(
                    min(self._camera_rotation[0], 90.0), -90.0
                )
            elif key == "2":
                self._camera_rotation[0] -= 1.0
                self._camera_rotation[0] = max(
                    min(self._camera_rotation[0], 90.0), -90.0
                )
            elif key == "4":
                self._camera_rotation[1] -= 1.0
                self._camera_rotation[1] %= 360.0
            elif key == "6":
                self._camera_rotation[1] += 1.0
                self._camera_rotation[1] %= 360.0
            elif key == "e":
                self._camera_rotation[2] += 1.0
                self._camera_rotation[2] %= 360.0
            elif key == "q":
                self._camera_rotation[2] -= 1.0
                self._camera_rotation[2] %= 360.0
            # Reset
            if key == "5":
                self._camera_coordinate = [0.0, 0.0, 0.0]
                self._camera_rotation = [0.0, 0.0, 0.0]
            if key in ("\x03", "\x1a", "\x1c"):
                self._keyboard_listener.stop()
                sys.exit(0)
            if key:
                self._triangles: list[Triangle] = []
                for triangle_vertices in self._triangle_vertices:
                    x_p, y_p, z_p = self._camera_coordinate
                    xd_p, yd_p, zd_p = self._camera_rotation
                    (x_a, y_a, z_a), (x_b, y_b, z_b), (x_c, y_c, z_c) = (
                        triangle_vertices
                    )
                    # Position
                    # Using vector for relative position
                    vx_ap, vy_ap, vz_ap = x_a - x_p, y_a - y_p, z_a - z_p
                    vx_bp, vy_bp, vz_bp = x_b - x_p, y_b - y_p, z_b - z_p
                    vx_cp, vy_cp, vz_cp = x_c - x_p, y_c - y_p, z_c - z_p
                    # Rotation
                    # Trigonometric values of X/Y/Z rotations
                    xr_p, yr_p, zr_p = (
                        math.radians(-xd_p),
                        math.radians(yd_p),
                        math.radians(zd_p),
                    )
                    xs_p, xc_p, ys_p, yc_p, zs_p, zc_p = (
                        math.sin(xr_p),
                        math.cos(xr_p),
                        math.sin(yr_p),
                        math.cos(yr_p),
                        math.sin(zr_p),
                        math.cos(zr_p),
                    )
                    # X-axis rotation that affects Y/Z coordinates
                    vy_ap, vz_ap = (
                        vy_ap * xc_p + vz_ap * xs_p,
                        vz_ap * xc_p - vy_ap * xs_p,
                    )
                    vy_bp, vz_bp = (
                        vy_bp * xc_p + vz_bp * xs_p,
                        vz_bp * xc_p - vy_bp * xs_p,
                    )
                    vy_cp, vz_cp = (
                        vy_cp * xc_p + vz_cp * xs_p,
                        vz_cp * xc_p - vy_cp * xs_p,
                    )
                    # Y-axis rotation that affects X/Z coordinates
                    vx_ap, vz_ap = (
                        vx_ap * yc_p - vz_ap * ys_p,
                        vx_ap * ys_p + vz_ap * yc_p,
                    )
                    vx_bp, vz_bp = (
                        vx_bp * yc_p - vz_bp * ys_p,
                        vx_bp * ys_p + vz_bp * yc_p,
                    )
                    vx_cp, vz_cp = (
                        vx_cp * yc_p - vz_cp * ys_p,
                        vx_cp * ys_p + vz_cp * yc_p,
                    )
                    # Z-axis rotation that affects X/Y coordinates
                    vx_ap, vy_ap = (
                        vx_ap * zc_p - vy_ap * zs_p,
                        vx_ap * zs_p + vy_ap * zc_p,
                    )
                    vx_bp, vy_bp = (
                        vx_bp * zc_p - vy_bp * zs_p,
                        vx_bp * zs_p + vy_bp * zc_p,
                    )
                    vx_cp, vy_cp = (
                        vx_cp * zc_p - vy_cp * zs_p,
                        vx_cp * zs_p + vy_cp * zc_p,
                    )
                    # Simple culling. TODO: advanced culling mechanism.
                    if vz_ap <= 0.0 or vz_bp <= 0.0 or vz_cp <= 0.0:
                        self._triangles.append(Triangle(null=True))
                    else:
                        self._triangles.append(
                            Triangle(
                                vertex_a=(
                                    (vx_ap) * self._camera_fov / (vz_ap),
                                    (vy_ap) * self._camera_fov / (vz_ap),
                                ),
                                vertex_b=(
                                    (vx_bp) * self._camera_fov / (vz_bp),
                                    (vy_bp) * self._camera_fov / (vz_bp),
                                ),
                                vertex_c=(
                                    (vx_cp) * self._camera_fov / (vz_cp),
                                    (vy_cp) * self._camera_fov / (vz_cp),
                                ),
                            )
                        )
            for y in range(0, screen_height):
                row_buffer: RowType = []
                for x in range(0, screen_width):
                    row_buffer.append(
                        (
                            255,
                            255,
                            255,
                            255,
                            ord(
                                "█"
                                if any(
                                    (x - half_width, y - half_height) in triangle
                                    for triangle in self._triangles
                                )
                                else " "
                            ),
                        )
                    )
                frame_buffer.insert(0, row_buffer)
            yield frame_buffer
