#!/usr/bin/env python3
import atexit
import math
import os
import time

from hintings import FrameType, RenderModeType


class InvalidRenderModeError(Exception):
    pass


class _Renderer(object):
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

    SHORT_ASCII_SEQUENCE = "@%#*+=-:. "[::-1]
    STANDARD_ASCII_SEQUENCE = (
        "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]
    )
    LONG_ASCII_SEQUENCE = "@MBHENR#KWXDFPQASUZbdehx*8Gm&04LOVYkpq5Tagns69owz$CIu23Jcfry%1v7l+it[]{}?j|()=~!-/<>\"^_';,:`. "[
        ::-1
    ]
    SHORT_ASCII_SEQUENCE_RANGE = len(SHORT_ASCII_SEQUENCE) - 1
    STANDARD_ASCII_SEQUENCE_RANGE = len(STANDARD_ASCII_SEQUENCE) - 1
    LONG_ASCII_SEQUENCE_RANGE = len(LONG_ASCII_SEQUENCE) - 1
    R_FACTOR = 0.2126 / 255.0
    G_FACTOR = 0.7152 / 255.0
    B_FACTOR = 0.0722 / 255.0
    C_POWER = 1.0 / 2.4

    _terminal_width, _terminal_height = os.get_terminal_size()
    _frame_buffer = None
    _ascii_buffer = None
    _gray_buffer = None
    _rgba_buffer = None
    _fps_counters: list[float] = []
    _time_counter = time.perf_counter_ns()

    @classmethod
    def __display_average_fps__(cls) -> None:
        if len(cls._fps_counters) == 0:
            return
        terminal_width, terminal_height = os.get_terminal_size()
        average_fps = "average rendering fps: %.1f" % (
            sum(cls._fps_counters) / len(cls._fps_counters)
        )
        string_length = len(average_fps)
        if terminal_width <= string_length:
            return
        print(
            end="\x1b[0m\x1b[%d;1H\x1b[s\x1b[%d;%dH%s\x1b[u"  # end, save, move, restore
            % (
                terminal_height - 1,
                terminal_height // 2 + 1,
                (terminal_width - string_length) // 2 + 1,
                average_fps,
            ),
            flush=True,
        )

    @classmethod
    def render(cls, frame: FrameType, fps: int, mode: RenderModeType) -> None:
        if not frame:
            return
        terminal_width, terminal_height = os.get_terminal_size()
        frame_width, frame_height = len(frame[0]), len(frame)
        frame_buffer: list[str] = []
        if (
            cls._terminal_width != terminal_width
            or cls._terminal_height != terminal_height
        ):
            cls._terminal_width = terminal_width
            cls._terminal_height = terminal_height
            cls._frame_buffer = None
            cls._ascii_buffer = None
            cls._gray_buffer = None
            cls._rgba_buffer = None
            cls.clear(mode=2)
        if mode == "frame":
            for y in range(0, terminal_height):
                frame_y = y * frame_height // terminal_height
                for x in range(0, terminal_width):
                    frame_x = x * frame_width // terminal_width
                    if cls._frame_buffer is None or (
                        cls._frame_buffer[frame_y][frame_x] != frame[frame_y][frame_x]
                    ):
                        fr, fg, fb, fa, c = frame[frame_y][frame_x]
                        frame_buffer.append(
                            "\x1b[%d;%dH\x1b[38;2;%d;%d;%d;%dm%c"
                            % (y + 1, x + 1, fr, fg, fb, fa // 255, c)
                        )
        elif mode == "ascii":
            for y in range(0, terminal_height):
                frame_y = y * frame_height // terminal_height
                for x in range(0, terminal_width):
                    frame_x = x * frame_width // terminal_width
                    if cls._ascii_buffer is None or (
                        cls._ascii_buffer[frame_y][frame_x] != frame[frame_y][frame_x]
                    ):
                        r, g, b, a = frame[frame_y][frame_x]
                        c_linear = (
                            r * cls.R_FACTOR + g * cls.G_FACTOR + b * cls.B_FACTOR
                        )
                        frame_buffer.append(
                            "\x1b[%d;%dH%c"
                            % (
                                y + 1,
                                x + 1,
                                cls.SHORT_ASCII_SEQUENCE[
                                    round(
                                        (
                                            1.055 * (c_linear**cls.C_POWER) - 0.055
                                            if c_linear > 0.0031308
                                            else 12.92 * c_linear
                                        )
                                        * (a / 255)
                                        * cls.SHORT_ASCII_SEQUENCE_RANGE
                                    )
                                ],
                            )
                        )
        elif mode == "gray":
            for y in range(0, terminal_height):
                frame_y = y * frame_height // terminal_height
                for x in range(0, terminal_width):
                    frame_x = x * frame_width // terminal_width
                    if cls._gray_buffer is None or (
                        cls._gray_buffer[frame_y][frame_x] != frame[frame_y][frame_x]
                    ):
                        r, g, b, a = frame[frame_y][frame_x]
                        c_linear = (
                            r * cls.R_FACTOR + g * cls.G_FACTOR + b * cls.B_FACTOR
                        )
                        r = g = b = round(
                            (
                                1.055 * (c_linear**cls.C_POWER) - 0.055
                                if c_linear > 0.0031308
                                else 12.92 * c_linear
                            )
                            * 255
                        )
                        frame_buffer.append(
                            "\x1b[%d;%dH\x1b[38;2;%d;%d;%d;%dm█"
                            % (y + 1, x + 1, r, g, b, a // 255)
                        )
        elif mode == "rgba":
            for y in range(0, terminal_height):
                frame_y = y * frame_height // terminal_height
                for x in range(0, terminal_width):
                    frame_x = x * frame_width // terminal_width
                    if cls._rgba_buffer is None or (
                        cls._rgba_buffer[frame_y][frame_x] != frame[frame_y][frame_x]
                    ):
                        r, g, b, a = frame[frame_y][frame_x]
                        frame_buffer.append(
                            "\x1b[%d;%dH\x1b[38;2;%d;%d;%d;%dm█"
                            % (y + 1, x + 1, r, g, b, a // 255)
                        )
        else:
            raise InvalidRenderModeError('unknown render mode - "%s"' % mode)
        cls._frame_buffer = frame
        print(end="".join(frame_buffer))
        time.sleep(
            max(
                0,
                1 / (fps or math.inf)
                - (time.perf_counter_ns() - cls._time_counter) / 1e9,
            )
        )
        cls._fps_counters.append(1e9 / (time.perf_counter_ns() - cls._time_counter))
        cls._time_counter = time.perf_counter_ns()
        print(end="\x1b];%.1f fps\a" % cls._fps_counters[-1], flush=True)

    @staticmethod
    def clear(mode: int) -> None:
        print(end="\x1b[%sJ" % mode, flush=True)


atexit.register(_Renderer.__display_average_fps__)


def clear_terminal(mode: int = 2) -> None:
    """Clear terminal

    Args:
        mode (int, optional): Defaults to 2
    """
    _Renderer.clear(mode=mode)


def render_frame(frame: FrameType, fps: int = 0) -> None:
    """Render target frame with provided RGBA colored foreground characters and background
    Single pixel data: (R, G, B, A, character)

    Args:
        frame (FrameType): frame buffer to render
        fps (int): fps limit. 0 means unlimited. Defaults to 0
    """
    _Renderer.render(frame=frame, fps=fps, mode="frame")


def render_ascii(frame: FrameType, fps: int = 0) -> None:
    """Render target frame with varying-density ASCII foreground characters
    Single pixel data: (R, G, B, A)

    Args:
        frame (FrameType): frame buffer to render
        fps (int): fps limit. 0 means unlimited. Defaults to 0
    """
    _Renderer.render(frame=frame, fps=fps, mode="ascii")


def render_gray(frame: FrameType, fps: int = 0) -> None:
    """Render target frame with grayscale colored background
    Single pixel data: (R, G, B, A)

    Args:
        frame (FrameType): frame buffer to render
        fps (int): fps limit. 0 means unlimited. Defaults to 0
        size (tuple[int, int]): target rendering size
    """
    _Renderer.render(frame=frame, fps=fps, mode="gray")


def render_rgba(frame: FrameType, fps: int = 0) -> None:
    """Render target frame with RGBA colored background
    Single pixel data: (R, G, B, A)

    Args:
        frame (FrameType): frame buffer to render
        fps (int): fps limit. 0 means unlimited. Defaults to 0
        size (tuple[int, int]): target rendering size
    """
    _Renderer.render(frame=frame, fps=fps, mode="rgba")


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
    for render in (render_ascii, render_gray, render_rgba):
        render(frame_buffer, fps=0)
        time.sleep(1)
