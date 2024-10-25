import os
import sys
import tempfile
import time
import wave
from concurrent.futures import Future, ProcessPoolExecutor
from threading import Thread

from exception import UnsupportedPlatformError


def windows_wrapper(filepath: str, delay: float) -> None:
    import winsound
    from winsound import SND_ASYNC, SND_FILENAME  # type: ignore

    time.sleep(delay)
    winsound.PlaySound(filepath, SND_ASYNC | SND_FILENAME)


def linux_wrapper(filepath: str, delay: float) -> None:
    time.sleep(delay)
    os.popen("aplay %s -q" % filepath)


def macos_wrapper(filepath: str, delay: float) -> None:
    time.sleep(delay)
    os.popen("afplay %s -q 1" % filepath)


if sys.platform.startswith("win"):
    wrapper = windows_wrapper
elif sys.platform.startswith("linux"):
    wrapper = linux_wrapper
elif sys.platform.startswith("darwin"):
    wrapper = macos_wrapper
else:
    raise UnsupportedPlatformError("current platform is not supported")


class WavePlayer(object):
    def __init__(self, max_workers: int = os.cpu_count() or 1) -> None:
        self._max_workers = max_workers
        self._pool_executor = ProcessPoolExecutor(max_workers=max_workers)

    def _play_wav(self, filepath: str) -> None:
        with wave.open(filepath, "rb") as wav:
            wav_total_frames = wav.getnframes()
            wav_framerate = wav.getframerate()
            wav_params = wav.getparams()
            wav_time = wav_total_frames / wav_framerate
            per_part_time = wav_time / self._max_workers
            per_part_frames = wav_total_frames // self._max_workers
            last_part_frames = (
                wav_total_frames - (self._max_workers - 1) * per_part_frames
            )
            async_pool_arguments: list[tuple[str, float]] = []
            time_start = time.perf_counter()
            part_index = 0
            try:
                for part_index in range(0, self._max_workers - 1):
                    part_filepath = tempfile.mktemp(suffix=".wav", prefix="tmp_PTR_")
                    with wave.open(part_filepath, "wb") as part_tmpfile:
                        part_tmpfile.setparams(wav_params)
                        part_tmpfile.writeframes(wav.readframes(per_part_frames))
                        async_pool_arguments.append(
                            (
                                part_filepath,
                                part_index * per_part_time,
                            )
                        )
                else:
                    part_filepath = tempfile.mktemp(suffix=".wav", prefix="tmp_PTR_")
                    with wave.open(part_filepath, "wb") as part_tmpfile:
                        part_tmpfile.setparams(wav_params)
                        part_tmpfile.writeframes(wav.readframes(last_part_frames))
                        async_pool_arguments.append(
                            (
                                part_filepath,
                                (part_index + 1) * per_part_time,
                            )
                        )
                # Process pool
                futures: list[Future[None]] = []
                async_time_start = time.perf_counter()
                for part_filepath, part_delay in async_pool_arguments:
                    futures.append(
                        self._pool_executor.submit(
                            wrapper,
                            part_filepath,
                            (
                                part_delay
                                if part_delay == 0
                                else part_delay
                                - (time.perf_counter() - async_time_start)
                            ),
                        )
                    )
                remaining_time = wav_time - (time.perf_counter() - time_start)
                if remaining_time > 0:
                    time.sleep(remaining_time)
            finally:
                for part_filepath, *_ in async_pool_arguments:
                    os.remove(part_filepath)

    def play(self, filepath: str, asynchronous: bool = False) -> None:
        if asynchronous:
            Thread(target=self._play_wav, args=(filepath,), daemon=True).start()
        else:
            self._play_wav(filepath=filepath)


if __name__ == "__main__":
    wave_player = WavePlayer()
    target = wave_player.play("resource/audio/piano.wav", asynchronous=False)
