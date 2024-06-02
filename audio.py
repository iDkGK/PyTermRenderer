import os
import platform

if platform.system() == "Windows":
    import winsound
    from winsound import SND_ASYNC, SND_FILENAME  # type: ignore

    def play_sound_async(wav_filepath: str) -> None:
        winsound.PlaySound(wav_filepath, SND_FILENAME | SND_ASYNC)  # type: ignore

elif platform.system() == "Linux":

    def play_sound_async(wav_filepath: str) -> None:
        os.popen("aplay %s" % wav_filepath)

elif platform.system() == "macOS":

    def play_sound_async(wav_filepath: str) -> None:
        os.popen("afplay %s" % wav_filepath)

else:
    raise NotImplementedError

if __name__ == "__main__":
    play_sound_async("resource/audio/piano.wav")
