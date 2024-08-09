class AlreadyExitedError(Exception):
    pass


class AlreadyInstantiatedError(Exception):
    pass


class CameraScreenTooSmallError(Exception):
    pass


class FileCorruptionError(Exception):
    pass


class InvalidEffectModeError(Exception):
    pass


class InvalidLockStatusError(Exception):
    pass


class UnsupportedPlatformError(Exception):
    pass
