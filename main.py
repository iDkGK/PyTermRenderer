#!/usr/bin/env python3
import os
from itertools import chain
from multiprocessing import Process, Queue
from pathlib import Path
from threading import Lock

from backend import TheMatrixCodeRain, DigitalTimeUnit, Fake3DSceneGame
from controller import KeyboardListener
from decoder import PNG, PNGSequence
from hintings import FramesType
from renderer import (
    clear_terminal,
    render_frame,
    render_ascii,
    render_gray,
    render_rgba,
)


class InvalidLockStatusError(Exception):
    pass


def play_code_rain_animation(fps: int = 15):
    code_rain = TheMatrixCodeRain(mode="long")
    for frame in code_rain.frames:
        render_frame(frame, fps=fps)


def play_digital_clock_animation(fps: int = 10):
    time_clock = DigitalTimeUnit()
    for frame in time_clock.frames:
        render_frame(frame, fps=fps)


def play_fake_3d_scene_game(fps: int = 60):
    scene_game = Fake3DSceneGame()
    for frame in scene_game.frames:
        render_frame(frame, fps=fps)


def display_static_png_image(fps: int = 0):
    example_png = PNG("resource/example.png").decode()
    render_ascii(example_png.image_data, fps)
    render_gray(example_png.image_data, fps)
    render_rgba(example_png.image_data, fps)


def play_dynamic_png_images_ascii(fps: int = 0):

    holder_lock = Lock()
    renderer_lock = Lock()
    keyboard_listener = KeyboardListener()
    example_png_sequence_ascii = PNGSequence("resource/example").decode_all()
    example_png_sequence_gray = PNGSequence("resource/example").decode_all()
    example_png_sequence_rgba = PNGSequence("resource/example").decode_all()

    def switch() -> None:
        if holder_lock.locked() and renderer_lock.locked():
            holder_lock.release()
            renderer_lock.release()
        elif holder_lock.locked() and not renderer_lock.locked():
            raise InvalidLockStatusError(
                "holder lock is locked while renderer lock is released."
            )
        else:
            while not holder_lock.locked():
                if renderer_lock.acquire():
                    holder_lock.acquire()

    keyboard_listener.register(" ", switch)

    for example_png in example_png_sequence_ascii:
        hit_key = keyboard_listener.get()
        hit_key_hex = hit_key.encode().hex()
        if hit_key_hex in ("03", "1a", "1b", "1c"):
            break
        renderer_lock.acquire()
        render_ascii(example_png.image_data, fps)
        renderer_lock.release()

    for example_png in example_png_sequence_gray:
        hit_key = keyboard_listener.get()
        hit_key_hex = hit_key.encode().hex()
        if hit_key_hex in ("03", "1a", "1b", "1c"):
            break
        renderer_lock.acquire()
        render_gray(example_png.image_data, fps)
        renderer_lock.release()

    for example_png in example_png_sequence_rgba:
        hit_key = keyboard_listener.get()
        hit_key_hex = hit_key.encode().hex()
        if hit_key_hex in ("03", "1a", "1b", "1c"):
            break
        renderer_lock.acquire()
        render_rgba(example_png.image_data, fps)
        renderer_lock.release()

    keyboard_listener.stop()
    if holder_lock.locked():
        holder_lock.release()
    if renderer_lock.locked():
        renderer_lock.release()


def _decode_processor(
    generator_queue,  # type: Queue[tuple[int, FramesType]]
    indexed_sequence: list[PNG],
    generator_index: int,
) -> None:
    generator_queue.put_nowait(
        (generator_index, [png.image_data for png in indexed_sequence])
    )


def play_dynamic_png_images_in_parallel() -> FramesType:
    png_dirpath = Path("resource/example")
    if not png_dirpath.exists():
        raise FileNotFoundError(
            "directory %s does not exists." % png_dirpath.as_posix()
        )
    if not png_dirpath.is_dir():
        raise NotADirectoryError("%s is not a directory." % png_dirpath.as_posix())
    png_sequence = [
        PNG(filepath.as_posix())
        for filepath in sorted(
            filter(
                lambda filepath: filepath.stem.lstrip(
                    "%s-" % png_dirpath.name
                ).isdecimal(),
                png_dirpath.glob("%s-*" % png_dirpath.name),
            ),
            key=lambda filepath: int(
                filepath.name.lstrip("%s-" % png_dirpath.name).rstrip(".png")
            ),
        )
    ]
    task_count = len(png_sequence)
    generator_count = (os.cpu_count() or 2) - 1
    allocated_task_count = 0
    worker_processes: list[Process] = []
    generator_queue = Queue()  # type: Queue[tuple[int, FramesType]] # type: ignore
    for worker_index in range(0, generator_count):
        if worker_index < generator_count - 1:
            allocate_task_count = task_count // (generator_count - 1)
        else:
            allocate_task_count = task_count % (generator_count - 1)
        worker_processes.append(
            Process(
                target=_decode_processor,
                args=(
                    generator_queue,
                    png_sequence[
                        allocated_task_count : allocated_task_count
                        + allocate_task_count
                    ],
                    worker_index,
                ),
            )
        )
        allocated_task_count += allocate_task_count
    for worker_process in worker_processes:
        worker_process.start()
    indexed_frames_lists: list[tuple[int, FramesType]] = []
    while len(indexed_frames_lists) < generator_count:
        indexed_frames_lists.append(generator_queue.get())
    return chain(
        *map(
            lambda indexed_frames_list: indexed_frames_list[1],
            sorted(indexed_frames_lists, key=lambda indexed_frame: indexed_frame[0]),
        )
    )


if __name__ == "__main__":
    clear_terminal()
    # play_code_rain_animation(fps=15)
    # play_digital_clock_animation(fps=10)
    play_fake_3d_scene_game(fps=30)
    # display_static_png_image(fps=1)
    # play_dynamic_png_images_ascii(fps=15)
    # decode_png_images_in_parallel()
