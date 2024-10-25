#!/usr/bin/env python3
import os
from itertools import chain
from multiprocessing import Process, Queue
from pathlib import Path
from threading import Lock

from backend import DigitalTimeUnit, Fake3DSceneGame, TheMatrixCodeRain
from controller import KeyboardListener
from decoder import PNG, PNGSequence
from display import (
    clear_screen,
    display_ascii,
    display_frame,
    display_gray,
    display_rgba,
)
from exception import InvalidLockStatusError
from hinting import FramesType


def play_code_rain_animation(fps: int = 15) -> None:
    code_rain = TheMatrixCodeRain(mode="long")
    for frame in code_rain.frames:
        display_frame(frame, fps=fps)


def play_digital_clock_animation(fps: int = 10) -> None:
    time_clock = DigitalTimeUnit()
    for frame in time_clock.frames:
        display_frame(frame, fps=fps)


def play_fake_3d_scene_game(fps: int = 60) -> None:
    scene_game = Fake3DSceneGame()
    for frame in scene_game.frames:
        display_frame(frame, fps=fps)


def display_static_png_image(fps: int = 0) -> None:
    example_png = PNG("resource/images/example.png").decode()
    display_ascii(example_png.image_data, fps)
    display_gray(example_png.image_data, fps)
    display_rgba(example_png.image_data, fps)


def play_dynamic_png_images_ascii(fps: int = 0) -> None:

    holder_lock = Lock()
    display_lock = Lock()
    keyboard_listener = KeyboardListener()
    example_png_sequence_ascii = PNGSequence(
        "resource/images/sequence/tom&jerry"
    ).decode_all()
    example_png_sequence_gray = PNGSequence(
        "resource/images/sequence/tom&jerry"
    ).decode_all()
    example_png_sequence_rgba = PNGSequence(
        "resource/images/sequence/tom&jerry"
    ).decode_all()

    def switch() -> None:
        if holder_lock.locked() and display_lock.locked():
            holder_lock.release()
            display_lock.release()
        elif holder_lock.locked() and not display_lock.locked():
            raise InvalidLockStatusError(
                "holder lock is locked while display lock is released."
            )
        else:
            while not holder_lock.locked():
                if display_lock.acquire():
                    holder_lock.acquire()

    keyboard_listener.register(" ", switch)

    for example_png in example_png_sequence_ascii:
        hit_key = keyboard_listener.get()
        hit_key_hex = hit_key.encode().hex()
        if hit_key_hex in ("03", "1a", "1b", "1c"):
            break
        display_lock.acquire()
        display_ascii(example_png.image_data, fps)
        display_lock.release()

    for example_png in example_png_sequence_gray:
        hit_key = keyboard_listener.get()
        hit_key_hex = hit_key.encode().hex()
        if hit_key_hex in ("03", "1a", "1b", "1c"):
            break
        display_lock.acquire()
        display_gray(example_png.image_data, fps)
        display_lock.release()

    for example_png in example_png_sequence_rgba:
        hit_key = keyboard_listener.get()
        hit_key_hex = hit_key.encode().hex()
        if hit_key_hex in ("03", "1a", "1b", "1c"):
            break
        display_lock.acquire()
        display_rgba(example_png.image_data, fps)
        display_lock.release()

    keyboard_listener.stop()
    if holder_lock.locked():
        holder_lock.release()
    if display_lock.locked():
        display_lock.release()


def _decode_processor(
    generator_queue,  # type: Queue[tuple[int, FramesType]]
    indexed_sequence: list[PNG],
    generator_index: int,
) -> None:
    generator_queue.put_nowait(
        (generator_index, [png.image_data for png in indexed_sequence])
    )


def play_dynamic_png_images_in_parallel(fps: int = 0) -> None:
    png_dirpath = Path("resource/images/sequence/tom&jerry")
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
    holder_lock = Lock()
    display_lock = Lock()
    keyboard_listener = KeyboardListener()

    def switch() -> None:
        if holder_lock.locked() and display_lock.locked():
            holder_lock.release()
            display_lock.release()
        elif holder_lock.locked() and not display_lock.locked():
            raise InvalidLockStatusError(
                "holder lock is locked while display lock is released."
            )
        else:
            while not holder_lock.locked():
                if display_lock.acquire():
                    holder_lock.acquire()

    keyboard_listener.register(" ", switch)

    for frame in chain(
        *map(
            lambda indexed_frames_list: indexed_frames_list[1],
            sorted(indexed_frames_lists, key=lambda indexed_frame: indexed_frame[0]),
        )
    ):
        hit_key = keyboard_listener.get()
        hit_key_hex = hit_key.encode().hex()
        if hit_key_hex in ("03", "1a", "1b", "1c"):
            break
        display_lock.acquire()
        display_ascii(frame, fps)
        display_lock.release()

    keyboard_listener.stop()
    if holder_lock.locked():
        holder_lock.release()
    if display_lock.locked():
        display_lock.release()


if __name__ == "__main__":
    clear_screen()
    if False:
        display_static_png_image(fps=1)
    if False:
        play_dynamic_png_images_ascii(fps=15)
    if False:
        play_dynamic_png_images_in_parallel(fps=15)
    if False:
        play_code_rain_animation(fps=15)
    if False:
        play_digital_clock_animation(fps=10)
    if True:
        play_fake_3d_scene_game(fps=30)
