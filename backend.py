import os
import random
import string
import sys
from datetime import datetime

from controller import KeyboardListener
from decoder import PNGSequence
from hintings import (
    FrameType,
    FramesType,
    RowType,
    EffectModeType,
    TriangleVerticesType,
)
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
        self._keyboard_listener = KeyboardListener()
        self._camera = [0.0, 0.0, 0.0, 0.0, 0.0]  # x, y, z, x_rotation, y_rotation
        self._fov = screen_height * screen_height / screen_width
        self._triangle_vertices: list[TriangleVerticesType] = []
        # self._triangle_vertices.append(
        #     (
        #         (random.random() * screen_width, random.random() * screen_height, 50),
        #         (random.random() * screen_width, random.random() * screen_height, 50),
        #         (random.random() * screen_width, random.random() * screen_height, 50),
        #     )
        # )
        self._triangle_vertices.append(((0, 0, 50), (0, 50, 50), (50, 0, 50)))
        self._triangle_vertices.append(((50, 50, 50), (50, 0, 50), (0, 50, 50)))
        self._triangle_vertices.append(((0, 0, 100), (0, 50, 100), (50, 0, 100)))
        self._triangle_vertices.append(((50, 50, 100), (50, 0, 100), (0, 50, 100)))

    def _update_triangles(self):
        self._triangles: list[Triangle] = []
        for triangle_vertices in self._triangle_vertices:
            x_p, y_p, z_p, xr_p, yr_p = self._camera
            (x_a, y_a, z_a), (x_b, y_b, z_b), (x_c, y_c, z_c) = triangle_vertices
            vx_ap, vy_ap, vz_ap = x_a - x_p, y_a - y_p, z_a - z_p
            vx_bp, vy_bp, vz_bp = x_b - x_p, y_b - y_p, z_b - z_p
            vx_cp, vy_cp, vz_cp = x_c - x_p, y_c - y_p, z_c - z_p
            # TODO: x/y rotation calculation via xr_p and yr_p
            if vz_ap <= 0.0 or vz_bp <= 0.0 or vz_cp <= 0.0:
                self._triangles.append(Triangle(null=True))
            else:
                self._triangles.append(
                    Triangle(
                        vertex_a=(
                            (vx_ap) * self._fov / (vz_ap),
                            (vy_ap) * self._fov / (vz_ap),
                        ),
                        vertex_b=(
                            (vx_bp) * self._fov / (vz_bp),
                            (vy_bp) * self._fov / (vz_bp),
                        ),
                        vertex_c=(
                            (vx_cp) * self._fov / (vz_cp),
                            (vy_cp) * self._fov / (vz_cp),
                        ),
                    )
                )

    @property
    def frames(self) -> FramesType:
        self._update_triangles()
        while True:
            screen_width, screen_height = os.get_terminal_size()
            key = self._keyboard_listener.get()
            frame_buffer: FrameType = []
            # Move
            if key == "w":
                self._camera[2] += 1.0
            elif key == "s":
                self._camera[2] -= 1.0
            elif key == "a":
                self._camera[0] -= 1.0
            elif key == "d":
                self._camera[0] += 1.0
            elif key == "W":
                self._camera[2] += 2.0
            elif key == "S":
                self._camera[2] -= 2.0
            elif key == "A":
                self._camera[0] -= 2.0
            elif key == "D":
                self._camera[0] += 2.0
            elif key == " ":
                self._camera[1] -= 1.0
            elif key == "\r":
                self._camera[1] += 1.0
            # Rotate
            elif key == "8":
                self._camera[3] += 1.0
                self._camera[3] = max(min(self._camera[3], 180.0), 0.0)
            elif key == "2":
                self._camera[3] -= 1.0
                self._camera[3] = max(min(self._camera[3], 180.0), 0.0)
            elif key == "4":
                self._camera[4] += 1.0
                self._camera[4] = self._camera[4] % 360.0
            elif key == "6":
                self._camera[4] -= 1.0
                self._camera[4] = self._camera[4] % 360.0
            # Reset
            elif key == "5":
                self._camera = [0.0, 0.0, 0.0, 0.0, 0.0]
            elif key in ("\x03", "\x1a", "\x1c"):
                # print("Manually Interrupted", flush=True)
                self._keyboard_listener.stop()
                sys.exit(0)
            if key:
                self._update_triangles()
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
                                    (x, y) in triangle for triangle in self._triangles
                                )
                                else " "
                            ),
                        )
                    )
                frame_buffer.insert(0, row_buffer)
            yield frame_buffer
