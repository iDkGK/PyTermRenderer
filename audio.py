import os
import sys
import time
import tempfile
import wave
import winsound
from concurrent.futures import Future, ProcessPoolExecutor
from threading import Thread


def windows_wrapper(filepath: str, data: bytes, delay: float) -> None:
    from winsound import SND_MEMORY  # type: ignore

    time.sleep(delay)
    winsound.PlaySound(data, SND_MEMORY)


def linux_wrapper(filepath: str, data: bytes, delay: float) -> None:
    time.sleep(delay)
    os.popen("aplay %s" % filepath)


def macos_wrapper(filepath: str, data: bytes, delay: float) -> None:
    time.sleep(delay)
    os.popen("afplay %s" % filepath)


if sys.platform.startswith("win"):
    wrapper = windows_wrapper
elif sys.platform.startswith("linux"):
    wrapper = linux_wrapper
elif sys.platform.startswith("darwin"):
    wrapper = macos_wrapper
else:

    class UnsupportedPlatformError(Exception):
        pass

    raise UnsupportedPlatformError("current platform is not supported")


class WavePlayer(object):
    def __init__(self, max_workers: int = os.cpu_count() or 1) -> None:
        self._max_workers = max_workers
        self._pool_executor = ProcessPoolExecutor(max_workers=max_workers)

    def play(self, filepath: str) -> None:
        with wave.open(filepath, "rb") as wav:
            wav_total_frames = wav.getnframes()
            wav_framerate = wav.getframerate()
            wav_params = wav.getparams()
            per_part_time = wav_total_frames / wav_framerate / self._max_workers
            per_part_frames = wav_total_frames // self._max_workers
            last_part_frames = (
                wav_total_frames - (self._max_workers - 1) * per_part_frames
            )
            async_pool_arguments: list[tuple[str, bytes, float]] = []
            part_index = 0
            try:
                for part_index in range(0, self._max_workers - 1):
                    part_filepath = tempfile.mktemp(suffix=".wav", prefix="tmp_PTR_")
                    part_data = wav.readframes(per_part_frames)
                    with wave.open(part_filepath, "wb") as part_tmpfile:
                        part_tmpfile.setparams(wav_params)
                        part_tmpfile.writeframes(part_data)
                    with open(part_filepath, "rb") as temp_file:
                        async_pool_arguments.append(
                            (
                                part_filepath,
                                temp_file.read(),
                                part_index * per_part_time,
                            )
                        )
                else:
                    part_filepath = tempfile.mktemp(suffix=".wav", prefix="tmp_PTR_")
                    part_data = wav.readframes(last_part_frames)
                    with wave.open(part_filepath, "wb") as part_tmpfile:
                        part_tmpfile.setparams(wav_params)
                        part_tmpfile.writeframes(part_data)
                    with open(part_filepath, "rb") as temp_file:
                        async_pool_arguments.append(
                            (
                                part_filepath,
                                temp_file.read(),
                                (part_index + 1) * per_part_time,
                            )
                        )
                # Process pool
                futures: list[Future[None]] = []
                time_start = time.perf_counter()
                for part_filepath, part_data, part_delay in async_pool_arguments:
                    futures.append(
                        self._pool_executor.submit(
                            wrapper,
                            part_filepath,
                            part_data,
                            (
                                part_delay
                                if part_delay == 0
                                else part_delay - (time.perf_counter() - time_start)
                            ),
                        )
                    )
                list(map(lambda future: future.result(), futures))
            finally:
                for part_filepath, *_ in async_pool_arguments:
                    os.remove(part_filepath)

    def async_play(self, filepath: str) -> None:
        Thread(target=self.play, args=(filepath,)).start()


if __name__ == "__main__":
    wave_player = WavePlayer(max_workers=16)
    target = wave_player.async_play("resource/audio/piano.wav")
    target = wave_player.play("resource/audio/piano.wav")
