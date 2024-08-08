import os
import random
import string
import time
from datetime import datetime

from decoder import PNGSequence
from hintings import EffectModeType, FramesType, FrameType, RowType
from utilities import Camera, Object, RotatingObject, SmoothCamera  # type: ignore

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
        self._screen_width = -1
        self._screen_height = -1
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

    def _spawn_code_rain(self, screen_height: int) -> tuple[int, int, int, str]:
        length = random.randint(screen_height // 3, screen_height)
        offset = random.randint(0, screen_height - length)
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
            screen_width, screen_height = os.get_terminal_size()
            if (
                screen_width != self._screen_width
                or screen_height != self._screen_height
            ):
                self._screen_width = screen_width
                self._screen_height = screen_height
                self._code_rains = dict(
                    (
                        x_position,
                        self._spawn_code_rain(self._screen_height),
                    )
                    for x_position in range(0, self._screen_width)
                )
            frame_buffer: FrameType = []
            for y in range(0, screen_height):
                row_buffer: RowType = []
                for x in range(0, screen_width):
                    fr, fg, fb, fa = TheMatrixCodeRain.SKY_COLOR
                    c = 32
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
                            self._code_rains[x] = self._spawn_code_rain(screen_height)
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
                for image in PNGSequence("resource/images/sequence/digits")
            )
        )

    @property
    def frames(self) -> FramesType:
        while True:
            screen_width, screen_height = os.get_terminal_size()
            current_time = datetime.now()
            time_string = "%02d:%02d:%02d" % (
                current_time.hour,
                current_time.minute,
                current_time.second,
            )
            time_string_length = len(time_string)
            canvas_width, canvas_height = (
                screen_width * 80 // 100,
                screen_height * 40 // 100,
            )
            canvas_left_border = (screen_width - canvas_width) // 2
            canvas_right_border = screen_width - canvas_left_border
            canvas_top_border = (screen_height - canvas_height) // 2
            canvas_bottom_border = screen_height - canvas_top_border
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
            for y in range(0, screen_height):
                row_buffer: RowType = []
                for x in range(0, screen_width):
                    fr, fg, fb, fa = DigitalTimeUnit.BG_COLOR
                    c = 32
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
        pass

    @property
    def frames(self) -> FramesType:
        # Create objects
        target_objects: list[Object] = []
        target_objects.append(
            Object(
                "resource/models/fox.obj",
                coordinate=(0.0, 0.0, 0.0),
            )
        )
        target_objects.append(
            RotatingObject(
                "resource/models/crafting_table.obj",
                coordinate=(-8.0, 0.0, 0.0),
            )
        )
        target_objects.append(
            RotatingObject(
                "resource/models/crafting_table.obj",
                coordinate=(8.0, 0.0, 0.0),
            )
        )
        smooth_camera = SmoothCamera(
            screen_size=os.get_terminal_size(),
            field_of_view=90,
            near_plane=0.0,
            far_plane=125.0,
            # coordinate=(-12.24744871391589, 10.0, -12.24744871391589),
            # rotation=(-30.0, 45.0),
            coordinate=(0.0, 0.0, -25.0),
            rotation=(0.0, 0.0),
        )
        for target_object in target_objects:
            smooth_camera.show_object(target_object)
        perf_counter = time.perf_counter_ns()
        while True:

            # Object update
            delta_time = (time.perf_counter_ns() - perf_counter) / 1e9
            perf_counter = time.perf_counter_ns()
            for target_object in target_objects:
                target_object.update(delta_time)
            smooth_camera.update(delta_time)

            # Frame generation
            yield smooth_camera.get_frame()
