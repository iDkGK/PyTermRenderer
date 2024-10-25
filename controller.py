#!/usr/bin/env python3
import atexit

try:
    from msvcrt import getwch  # type: ignore
except ImportError:
    import sys
    import termios
    import tty

    _stdin_attributes = termios.tcgetattr(sys.stdin.fileno())  # type: ignore

    def _set_stdin_tty() -> None:
        _stdin_attributes[3] &= ~termios.ECHO  # suppress echo  # type: ignore
        termios.tcsetattr(_stdin_fd, termios.TCSANOW, _stdin_attributes)  # type: ignore
        tty.setraw(_stdin_fd)  # type: ignore

    def _reset_stdin_tty() -> None:
        _stdin_attributes[3] |= termios.ECHO  # type: ignore
        tty.setcbreak(_stdin_fd)  # type: ignore
        termios.tcsetattr(_stdin_fd, termios.TCSADRAIN, _stdin_attributes)  # type: ignore

    def getwch() -> str:
        """
        Gets a single character from STDIO.
        """
        return sys.stdin.read(1)

    _set_stdin_tty()
    atexit.register(_reset_stdin_tty)


from queue import LifoQueue
from threading import Event, Thread

from exception import AlreadyExitedError, AlreadyInstantiatedError
from hinting import AnyType, CallableType, IterableType


class SingletonMeta(type):
    __instances__: dict["SingletonMeta", "SingletonMeta"] = {}

    def __call__(cls, *args: AnyType, **kwargs: AnyType) -> "SingletonMeta":
        if cls in cls.__instances__:
            raise AlreadyInstantiatedError(
                "attempted to instantiate KeyboardListener more than once"
            )
        cls.__instances__[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls.__instances__[cls]


class KeyboardListener(metaclass=SingletonMeta):
    def __init__(self) -> None:
        """Create an object of KeyboardListener to respond to keyboard inputs"""
        self._stopping_event = Event()
        self._hit_key_counter = 0
        self._hit_key_queue: LifoQueue[str] = LifoQueue()
        self._callback_registry: dict[str, list[CallableType[..., AnyType]]] = {}
        self._escape_keys: list[str] = []
        self._listen_thread = Thread(target=self._listen, daemon=True)
        self._listen_thread.start()

    def _listen(self) -> None:
        while not self._stopping_event.is_set():
            hit_key = getwch()
            if len(self._escape_keys) == 1:
                self._escape_keys.append(hit_key)
            if len(self._escape_keys) == 2:
                hit_key = "".join(self._escape_keys)
                self._escape_keys.clear()
            if hit_key == "\xe0":
                self._escape_keys.append(hit_key)
            else:
                self._hit_key_counter += 1
                self._hit_key_queue.put_nowait(hit_key)
            for key, callbacks in self._callback_registry.items():
                if key == hit_key:
                    for callback in callbacks:
                        callback()

    def stop(self) -> None:
        if self._stopping_event.is_set():
            raise AlreadyExitedError(
                "attempted to stop an exited KeyboardListener object"
            )
        for _ in range(0, self._hit_key_counter):
            self._hit_key_queue.task_done()
        self._hit_key_queue.join()
        self._stopping_event.set()

    def get(self, block: bool = False) -> str:
        if self._stopping_event.is_set():
            return ""
        if block or not self._hit_key_queue.empty():
            key = self._hit_key_queue.get()
            self._hit_key_queue.queue.clear()
            return key
        return ""

    def register(
        self,
        keys: IterableType[str],
        callbacks: IterableType[CallableType[..., AnyType]],
        override: bool = False,
    ) -> None:
        if self._stopping_event.is_set():
            raise AlreadyExitedError(
                "attempted to register key-callbacks to an exited KeyboardListener object"
            )
        callback_list = list(callbacks)
        for key in keys:
            if override:
                self._callback_registry[key] = callback_list
            else:
                self._callback_registry.setdefault(key, [])
                self._callback_registry[key] += callback_list

    def unregister(
        self,
        keys: IterableType[str],
        callbacks: IterableType[CallableType[..., AnyType]],
    ) -> None:
        if self._stopping_event.is_set():
            raise AlreadyExitedError(
                "attempted to unregister callbacks from an exited KeyboardListener object"
            )
        callback_list = list(callbacks)
        for key in keys:
            if key in self._callback_registry:
                for callback in callback_list:
                    if callback in self._callback_registry[key]:
                        del self._callback_registry[key][
                            self._callback_registry[key].index(callback)
                        ]
                if len(self._callback_registry[key]) == 0:
                    del self._callback_registry[key]


if __name__ == "__main__":
    stop_keys = ("\x03", "\x1a", "\x1b", "\x1c")
    keyboard_listener = KeyboardListener()
    keyboard_listener.register(stop_keys, (keyboard_listener.stop,))
    while True:
        hit_key = keyboard_listener.get(block=True)
        hit_key_hex = hit_key.encode().hex()
        if not hit_key:
            break
        print(
            "Hit Key: %s, Hex Code: %s" % (hit_key, hit_key_hex),
            end="\r\n",
        )
