#!/usr/bin/env python3
import atexit
import math
import os
import time
from collections import deque

from hintings import FrameType

"""
"\x1b[39m\x1b[39m"                 - Reset color
"\x1b[<L>;<C>H" OR "\x1b[<L>;<C>f" - Puts the cursor at y L and x C.
"\x1b[<N>A"                        - Move the cursor up N rows
"\x1b[<N>B"                        - Move the cursor down N rows
"\x1b[<N>C"                        - Move the cursor forward N columns
"\x1b[<N>D"                        - Move the cursor backward N columns
"\x1b[2J"                          - Clear the screen, move to (0,0)
"\x1b[2K"                          - Clear row
"\x1b[K"                           - Erase to end of row
"\x1b[s"                           - Save cursor position
"\x1b[u"                           - Restore cursor position
"\x1b[4m"                          - Underline on
"\x1b[24m"                         - Underline off
"\x1b[1m"                          - Bold on
"\x1b[21m"                         - Bold off
"""

__SHORT_ASCII_SEQUENCE__ = "@%#*+=-:. "[::-1]
__STANDARD_ASCII_SEQUENCE__ = (
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]
)
__LONG_ASCII_SEQUENCE__ = "@MBHENR#KWXDFPQASUZbdehx*8Gm&04LOVYkpq5Tagns69owz$CIu23Jcfry%1v7l+it[]{}?j|()=~!-/<>\"^_';,:`. "[
    ::-1
]
__SHORT_ASCII_SEQUENCE_RANGE__ = len(__SHORT_ASCII_SEQUENCE__) - 1
__STANDARD_ASCII_SEQUENCE_RANGE__ = len(__STANDARD_ASCII_SEQUENCE__) - 1
__LONG_ASCII_SEQUENCE_RANGE__ = len(__LONG_ASCII_SEQUENCE__) - 1
__R_FACTOR__ = 0.2126 / 255.0
__G_FACTOR__ = 0.7152 / 255.0
__B_FACTOR__ = 0.0722 / 255.0
__C_POWER__ = 1.0 / 2.4

__screen_width__, __screen_height__ = os.get_terminal_size()
__frame_buffer__ = None
__ascii_buffer__ = None
__gray_buffer__ = None
__rgba_buffer__ = None
__fps_counters__: deque[float] = deque(maxlen=30)
__time_counter__ = time.perf_counter_ns()


def __pring_average_fps__() -> None:
    if len(__fps_counters__) == 0:
        return
    screen_width, screen_height = os.get_terminal_size()
    average_fps = "average rendering fps: %.1f" % (
        sum(__fps_counters__) / len(__fps_counters__)
    )
    string_length = len(average_fps)
    if screen_width <= string_length:
        return
    print(
        end="\x1b[0m\x1b[%d;1H\x1b[s\x1b[%d;%dH%s\x1b[u"  # end, save, move, restore
        % (
            screen_height - 1,
            screen_height // 2 + 1,
            (screen_width - string_length) // 2 + 1,
            average_fps,
        ),
        flush=True,
    )


def display_frame(frame: FrameType, fps: int) -> None:
    """Display target frame with provided RGBA colored foreground characters and background
    Single pixel data: (R, G, B, A, character)

    Args:
        frame (FrameType): frame buffer to display
        fps (int): fps limit. 0 means unlimited. Defaults to 0
    """
    global __screen_width__, __screen_height__, __frame_buffer__, __time_counter__
    if frame:
        screen_width, screen_height = os.get_terminal_size()
        frame_width, frame_height = len(frame[0]), len(frame)
        frame_buffer: list[str] = []
        if __screen_width__ != screen_width or __screen_height__ != screen_height:
            __screen_width__ = screen_width
            __screen_height__ = screen_height
            __frame_buffer__ = None
            clear_screen(mode=2)
        for y in range(0, screen_height):
            frame_y = y * frame_height // screen_height
            for x in range(0, screen_width):
                frame_x = x * frame_width // screen_width
                if __frame_buffer__ is None or (
                    __frame_buffer__[frame_y][frame_x] != frame[frame_y][frame_x]
                ):
                    fr, fg, fb, fa, c = frame[frame_y][frame_x]
                    frame_buffer.append(
                        "\x1b[%d;%dH\x1b[38;2;%d;%d;%d;%dm%c"
                        % (y + 1, x + 1, fr, fg, fb, fa // 255, c)
                    )
        __frame_buffer__ = frame
        print(end="".join(frame_buffer))
    time.sleep(
        max(
            0,
            1 / (fps or math.inf) - (time.perf_counter_ns() - __time_counter__) / 1e9,
        )
    )
    __fps_counters__.append(1e9 / (time.perf_counter_ns() - __time_counter__))
    __time_counter__ = time.perf_counter_ns()
    print(end="\x1b];%.1f fps\a" % __fps_counters__[-1], flush=True)


def display_ascii(frame: FrameType, fps: int) -> None:
    """Display target frame with varying-density ASCII foreground characters
    Single pixel data: (R, G, B, A)

    Args:
        frame (FrameType): frame buffer to display
        fps (int): fps limit. 0 means unlimited. Defaults to 0
    """
    global __screen_width__, __screen_height__, __ascii_buffer__, __time_counter__
    if frame:
        screen_width, screen_height = os.get_terminal_size()
        frame_width, frame_height = len(frame[0]), len(frame)
        frame_buffer: list[str] = []
        if __screen_width__ != screen_width or __screen_height__ != screen_height:
            __screen_width__ = screen_width
            __screen_height__ = screen_height
            __ascii_buffer__ = None
            clear_screen(mode=2)
        for y in range(0, screen_height):
            frame_y = y * frame_height // screen_height
            for x in range(0, screen_width):
                frame_x = x * frame_width // screen_width
                if __ascii_buffer__ is None or (
                    __ascii_buffer__[frame_y][frame_x] != frame[frame_y][frame_x]
                ):
                    r, g, b, a = frame[frame_y][frame_x]
                    c_linear = r * __R_FACTOR__ + g * __G_FACTOR__ + b * __B_FACTOR__
                    frame_buffer.append(
                        "\x1b[%d;%dH%c"
                        % (
                            y + 1,
                            x + 1,
                            __SHORT_ASCII_SEQUENCE__[
                                round(
                                    (
                                        1.055 * (c_linear**__C_POWER__) - 0.055
                                        if c_linear > 0.0031308
                                        else 12.92 * c_linear
                                    )
                                    * (a / 255)
                                    * __SHORT_ASCII_SEQUENCE_RANGE__
                                )
                            ],
                        )
                    )
        __ascii_buffer__ = frame
        print(end="".join(frame_buffer))
    time.sleep(
        max(
            0,
            1 / (fps or math.inf) - (time.perf_counter_ns() - __time_counter__) / 1e9,
        )
    )
    __fps_counters__.append(1e9 / (time.perf_counter_ns() - __time_counter__))
    __time_counter__ = time.perf_counter_ns()
    print(end="\x1b];%.1f fps\a" % __fps_counters__[-1], flush=True)


def display_gray(frame: FrameType, fps: int) -> None:
    """Display target frame with grayscale colored background
    Single pixel data: (R, G, B, A)

    Args:
        frame (FrameType): frame buffer to display
        fps (int): fps limit. 0 means unlimited. Defaults to 0
    """
    global __screen_width__, __screen_height__, __gray_buffer__, __time_counter__
    if frame:
        screen_width, screen_height = os.get_terminal_size()
        frame_width, frame_height = len(frame[0]), len(frame)
        frame_buffer: list[str] = []
        if __screen_width__ != screen_width or __screen_height__ != screen_height:
            __screen_width__ = screen_width
            __screen_height__ = screen_height
            __gray_buffer__ = None
            clear_screen(mode=2)
        for y in range(0, screen_height):
            frame_y = y * frame_height // screen_height
            for x in range(0, screen_width):
                frame_x = x * frame_width // screen_width
                if __gray_buffer__ is None or (
                    __gray_buffer__[frame_y][frame_x] != frame[frame_y][frame_x]
                ):
                    r, g, b, a = frame[frame_y][frame_x]
                    c_linear = r * __R_FACTOR__ + g * __G_FACTOR__ + b * __B_FACTOR__
                    r = g = b = round(
                        (
                            1.055 * (c_linear**__C_POWER__) - 0.055
                            if c_linear > 0.0031308
                            else 12.92 * c_linear
                        )
                        * 255
                    )
                    frame_buffer.append(
                        "\x1b[%d;%dH\x1b[38;2;%d;%d;%d;%dm█"
                        % (y + 1, x + 1, r, g, b, a // 255)
                    )
        __gray_buffer__ = frame
        print(end="".join(frame_buffer))
    time.sleep(
        max(
            0,
            1 / (fps or math.inf) - (time.perf_counter_ns() - __time_counter__) / 1e9,
        )
    )
    __fps_counters__.append(1e9 / (time.perf_counter_ns() - __time_counter__))
    __time_counter__ = time.perf_counter_ns()
    print(end="\x1b];%.1f fps\a" % __fps_counters__[-1], flush=True)


def display_rgba(frame: FrameType, fps: int) -> None:
    """Display target frame with RGBA colored background
    Single pixel data: (R, G, B, A)

    Args:
        frame (FrameType): frame buffer to display
        fps (int): fps limit. 0 means unlimited. Defaults to 0
    """
    global __screen_width__, __screen_height__, __rgba_buffer__, __time_counter__
    if frame:
        screen_width, screen_height = os.get_terminal_size()
        frame_width, frame_height = len(frame[0]), len(frame)
        frame_buffer: list[str] = []
        if __screen_width__ != screen_width or __screen_height__ != screen_height:
            __screen_width__ = screen_width
            __screen_height__ = screen_height
            __rgba_buffer__ = None
            clear_screen(mode=2)
        for y in range(0, screen_height):
            frame_y = y * frame_height // screen_height
            for x in range(0, screen_width):
                frame_x = x * frame_width // screen_width
                if __rgba_buffer__ is None or (
                    __rgba_buffer__[frame_y][frame_x] != frame[frame_y][frame_x]
                ):
                    r, g, b, a = frame[frame_y][frame_x]
                    frame_buffer.append(
                        "\x1b[%d;%dH\x1b[38;2;%d;%d;%d;%dm█"
                        % (y + 1, x + 1, r, g, b, a // 255)
                    )
        __rgba_buffer__ = frame
        print(end="".join(frame_buffer))
    time.sleep(
        max(
            0,
            1 / (fps or math.inf) - (time.perf_counter_ns() - __time_counter__) / 1e9,
        )
    )
    __fps_counters__.append(1e9 / (time.perf_counter_ns() - __time_counter__))
    __time_counter__ = time.perf_counter_ns()
    print(end="\x1b];%.1f fps\a" % __fps_counters__[-1], flush=True)


def clear_screen(mode: int = 2) -> None:
    """Clear terminal

    Args:
        mode (int, optional): Defaults to 2
    """
    print(end="\x1b[%sJ" % mode, flush=True)


atexit.register(__pring_average_fps__)


if __name__ == "__main__":
    screen_width, screen_height = os.get_terminal_size()
    frame_buffer = [
        [
            (
                255 * x * y // (screen_width * screen_height),
                255 * x * y // (screen_width * screen_height),
                255 * x * y // (screen_width * screen_height),
                255,
            )
            for x in range(0, screen_width)
        ]
        for y in range(0, screen_height)
    ]
    for display_funcion in (display_ascii, display_gray, display_rgba):
        display_funcion(frame_buffer, fps=0)
        time.sleep(1)
