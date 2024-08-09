#!/usr/bin/env python3
import binascii
import math
import time
import warnings
import zlib
from functools import cached_property
from pathlib import Path

from exceptions import FileCorruptionError
from hintings import ImagesType, ImageType, RowType


class PNG(object):
    SIGN = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"
    IHDR = b"\x00\x00\x00\x0dIHDR"
    IDAT = b"IDAT"
    IEND = b"\x00\x00\x00\x00IEND"
    PLTE = b"PLTE"
    IEND_CRC = b"\xae\x42\x60\x82"
    DEFAULT_ALPHA_VALUE = 255

    def __init__(self, filepath: str, debug: bool = False) -> None:
        """Create an object of PNG and validate PNG image signature, header, crc32 and trailer

        Args:
            filepath (str): path to PNG image
            debug (bool, optional): True to print debug information after PNG images decoded.\
                                    Defaults to False

        Exceptions:
            FileNotFoundError: raise FileNotFoundError if path to filepath does not exists
            IsADirectoryError: raise IsADirectoryError if path to filepath is a directory
            FileCorruptionError: raise FileCorruptionError if validation fails
        """
        self._filepath = Path(filepath)
        if not self._filepath.exists():
            raise FileNotFoundError("file %s does not exists." % filepath)
        if not self._filepath.is_file():
            raise IsADirectoryError("%s is a directory." % filepath)
        self._binary_data = self._filepath.read_bytes()
        if not self._binary_data[:8] == self.SIGN:
            raise FileCorruptionError(
                "invalid signature. The image may not be in PNG format."
            )
        if not self._binary_data[8:16] == self.IHDR:
            raise FileCorruptionError(
                "invalid header. The image may not be in PNG format."
            )
        if not binascii.crc32(self._binary_data[12:29]) == self._crc32:
            raise FileCorruptionError(
                "invalid crc32. The image may not be in PNG format."
            )
        if not self._binary_data.find(self.IEND) != -1:
            raise FileCorruptionError(
                "invalid trailer. The image may not be in PNG format."
            )
        if not self._binary_data.find(self.IEND_CRC) != -1:
            raise FileCorruptionError(
                "invalid trailer crc32. The image may not be in PNG format."
            )
        self._debug = debug
        self._decoded = False

    def decode(self) -> "PNG":
        """Decode the PNG image for future use

        Returns:
            PNG: the PNG object itself
        """
        if self._decoded:
            return self
        time_start = time.perf_counter()
        index = self._binary_data.find(self.IDAT)
        if index == -1:
            raise FileCorruptionError("missing idat chunk. The image may be corrupted.")
        index -= 4
        binary_data: list[bytes] = []
        for _ in range(0, self._binary_data.count(self.IDAT)):
            data_length = int.from_bytes(
                self._binary_data[index : index + 4], byteorder="big"
            )
            idat_index, idat_length, crc32_length = index + 4, 4, 4
            index = idat_index + idat_length + data_length + crc32_length
            if binascii.crc32(
                self._binary_data[idat_index : idat_index + idat_length + data_length]
            ) != int.from_bytes(
                self._binary_data[
                    idat_index
                    + idat_length
                    + data_length : idat_index
                    + idat_length
                    + data_length
                    + crc32_length
                ],
                byteorder="big",
            ):
                raise FileCorruptionError("invalid crc32. The image may be corrupted.")
            binary_data.append(
                self._binary_data[
                    idat_index + idat_length : idat_index + idat_length + data_length
                ]
            )
        filtered_bytes = zlib.decompress(b"".join(binary_data))
        defiltered_bytes: list[bytes] = []
        self._image_data: ImageType = []
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(self._color_type, 0)
        if channels == 0:
            raise FileCorruptionError("unknown color type. The image may be corrupted.")
        row_length = math.ceil(self._width * channels * self._bit_depth / 8 + 1)
        if self._bit_depth == 16:
            left_distance = channels * 2
        else:
            left_distance = channels
        for row_index in range(0, self._height):
            filtered_row = filtered_bytes[
                row_index * row_length : (row_index + 1) * row_length
            ]
            filter_type, filtered_row_bytes = filtered_row[0], filtered_row[1:]
            if filter_type == 0:
                defiltered_row_bytes = filtered_row_bytes
            elif filter_type == 1:
                temp_defiltered_row_bytes: list[bytes] = []
                for byte_index in range(0, len(filtered_row_bytes)):
                    if byte_index - left_distance >= 0:
                        left_value = int.from_bytes(
                            temp_defiltered_row_bytes[byte_index - left_distance],
                            byteorder="big",
                        )
                    else:
                        left_value = 0
                    temp_defiltered_row_bytes.append(
                        ((filtered_row_bytes[byte_index] + left_value) % 256).to_bytes(
                            1, byteorder="big"
                        )
                    )
                defiltered_row_bytes = b"".join(temp_defiltered_row_bytes)
            elif filter_type == 2:
                temp_defiltered_row_bytes = []
                for byte_index in range(0, len(filtered_row_bytes)):
                    if row_index != 0:
                        upper_value = defiltered_bytes[row_index - 1][byte_index]
                    else:
                        upper_value = 0
                    temp_defiltered_row_bytes.append(
                        ((filtered_row_bytes[byte_index] + upper_value) % 256).to_bytes(
                            1, byteorder="big"
                        )
                    )
                defiltered_row_bytes = b"".join(temp_defiltered_row_bytes)
            elif filter_type == 3:
                temp_defiltered_row_bytes = []
                for byte_index in range(0, len(filtered_row_bytes)):
                    if byte_index - left_distance >= 0:
                        left_value = int.from_bytes(
                            temp_defiltered_row_bytes[byte_index - left_distance],
                            byteorder="big",
                        )
                    else:
                        left_value = 0
                    if row_index != 0:
                        upper_value = defiltered_bytes[row_index - 1][byte_index]
                    else:
                        upper_value = 0
                    temp_defiltered_row_bytes.append(
                        (
                            (
                                filtered_row_bytes[byte_index]
                                + (left_value + upper_value) // 2
                            )
                            % 256
                        ).to_bytes(1, byteorder="big")
                    )
                defiltered_row_bytes = b"".join(temp_defiltered_row_bytes)
            elif filter_type == 4:
                temp_defiltered_row_bytes = []
                for byte_index in range(0, len(filtered_row_bytes)):
                    if byte_index - left_distance >= 0:
                        left_value = int.from_bytes(
                            temp_defiltered_row_bytes[byte_index - left_distance],
                            byteorder="big",
                        )
                    else:
                        left_value = 0
                    if row_index != 0:
                        upper_value = defiltered_bytes[row_index - 1][byte_index]
                    else:
                        upper_value = 0
                    if byte_index - left_distance < 0 or row_index == 0:
                        upper_left_value = 0
                    else:
                        upper_left_value = defiltered_bytes[row_index - 1][
                            byte_index - left_distance
                        ]
                    left_value_delta = abs(upper_value - upper_left_value)
                    upper_value_delta = abs(left_value - upper_left_value)
                    upper_left_value_delta = abs(
                        left_value + upper_value - upper_left_value * 2
                    )
                    minimal_value_delta = min(
                        left_value_delta, upper_value_delta, upper_left_value_delta
                    )
                    if minimal_value_delta == left_value_delta:
                        paeth_pixel_value = left_value
                    elif minimal_value_delta == upper_value_delta:
                        paeth_pixel_value = upper_value
                    else:
                        paeth_pixel_value = upper_left_value
                    temp_defiltered_row_bytes.append(
                        (
                            (filtered_row_bytes[byte_index] + paeth_pixel_value) % 256
                        ).to_bytes(1, byteorder="big")
                    )
                defiltered_row_bytes = b"".join(temp_defiltered_row_bytes)
            else:
                raise FileCorruptionError(
                    "unknown filter type. The image may be corrupted."
                )
            defiltered_bytes.append(defiltered_row_bytes)
            rgba_row_pixel_tuples: RowType = []
            if self._color_type == 0:
                if self._bit_depth == 8:
                    for rgb_value in defiltered_row_bytes:
                        rgba_row_pixel_tuples.append(
                            (
                                rgb_value,
                                rgb_value,
                                rgb_value,
                                self.DEFAULT_ALPHA_VALUE,
                            )
                        )
                elif self._bit_depth == 16:
                    for rgb_index in range(0, len(defiltered_row_bytes) // 2):
                        rgba_row_pixel_tuples.append(
                            (
                                int.from_bytes(
                                    defiltered_row_bytes[
                                        rgb_index * 2 : (rgb_index + 1 * 2)
                                    ],
                                    byteorder="big",
                                ),
                                int.from_bytes(
                                    defiltered_row_bytes[
                                        rgb_index * 2 : (rgb_index + 1 * 2)
                                    ],
                                    byteorder="big",
                                ),
                                int.from_bytes(
                                    defiltered_row_bytes[
                                        rgb_index * 2 : (rgb_index + 1 * 2)
                                    ],
                                    byteorder="big",
                                ),
                                self.DEFAULT_ALPHA_VALUE,
                            )
                        )
                else:
                    for byte in defiltered_row_bytes:
                        if self._bit_depth == 1:
                            palette_indices = map(int, format(byte, "08b"))
                        elif self._bit_depth == 2:
                            palette_indices = (
                                byte // 64,
                                byte % 64 // 16,
                                byte % 16 // 4,
                                byte % 4,
                            )
                        elif self._bit_depth == 4:
                            palette_indices = (byte // 16, byte % 16)
                        else:
                            palette_indices = (byte,)
                        for palette_index in palette_indices:
                            rgba_row_pixel_tuples.append(self._palette[palette_index])
            elif self._color_type == 2:
                if self._bit_depth == 8:
                    for byte_index in range(0, self._width * channels, channels):
                        channel_pixel_tuples = tuple(
                            defiltered_row_bytes[byte_index + channel]
                            for channel in range(0, channels)
                        ) + (self.DEFAULT_ALPHA_VALUE,)
                        rgba_row_pixel_tuples.append(channel_pixel_tuples)
                elif self._bit_depth == 16:
                    for byte_index in range(
                        0, self._width * channels * 2, channels * 2
                    ):
                        channel_pixel_tuples = tuple(
                            int.from_bytes(
                                defiltered_row_bytes[
                                    byte_index + channel : byte_index + channel + 2
                                ],
                                byteorder="big",
                            )
                            for channel in range(0, channels)
                        ) + (self.DEFAULT_ALPHA_VALUE,)
                        rgba_row_pixel_tuples.append(channel_pixel_tuples)
                else:
                    raise FileCorruptionError(
                        "invalid bit depth. The image may be corrupted."
                    )
            elif self._color_type == 3:
                for byte in defiltered_row_bytes:
                    if self._bit_depth == 1:
                        palette_indices = map(int, format(byte, "08b"))
                    elif self._bit_depth == 2:
                        palette_indices = (
                            byte // 64,
                            byte % 64 // 16,
                            byte % 16 // 4,
                            byte % 4,
                        )
                    elif self._bit_depth == 4:
                        palette_indices = (byte // 16, byte % 16)
                    else:
                        palette_indices = (byte,)
                    for palette_index in palette_indices:
                        rgba_row_pixel_tuples.append(self._palette[palette_index])
            elif self._color_type == 4:
                if self._bit_depth == 8:
                    for byte_index in range(0, self._width * 2, 2):
                        rgba_row_pixel_tuples.append(
                            (
                                defiltered_row_bytes[byte_index],
                                defiltered_row_bytes[byte_index],
                                defiltered_row_bytes[byte_index],
                                defiltered_row_bytes[byte_index + 1],
                            )
                        )
                elif self._bit_depth == 16:
                    for byte_index in range(0, self._width * 4, 4):
                        rgb_value, a_value = int.from_bytes(
                            defiltered_row_bytes[byte_index : byte_index + 2],
                            byteorder="big",
                        ), int.from_bytes(
                            defiltered_row_bytes[byte_index + 2 : byte_index + 4],
                            byteorder="big",
                        )
                        rgba_row_pixel_tuples.append(
                            (rgb_value, rgb_value, rgb_value, a_value)
                        )
                else:
                    raise FileCorruptionError(
                        "invalid bit depth. The image may be corrupted."
                    )
            elif self._color_type == 6:
                if self._bit_depth == 8:
                    for byte_index in range(0, self._width * channels, channels):
                        channel_pixel_tuples = tuple(
                            defiltered_row_bytes[byte_index + channel]
                            for channel in range(0, channels)
                        )
                        rgba_row_pixel_tuples.append(channel_pixel_tuples)
                elif self._bit_depth == 16:
                    for byte_index in range(
                        0, self._width * channels * 2, channels * 2
                    ):
                        channel_pixel_tuples = tuple(
                            int.from_bytes(
                                defiltered_row_bytes[
                                    byte_index + channel : byte_index + channel + 2
                                ],
                                byteorder="big",
                            )
                            for channel in range(0, channels)
                        )
                        rgba_row_pixel_tuples.append(channel_pixel_tuples)
                else:
                    raise FileCorruptionError(
                        "invalid bit depth. The image may be corrupted."
                    )
            else:
                raise FileCorruptionError(
                    "unknown color type. The image may be corrupted."
                )
            self._image_data.append(rgba_row_pixel_tuples)
        time_middle = time.perf_counter()
        if self._debug:
            for _ in range(0, self._height):
                for _ in range(0, self._width):
                    pass
            time_end = time.perf_counter()
            processing_image_time_consuming = time_middle - time_start
            no_processing_image_time_consuming = time_end - time_middle
            if no_processing_image_time_consuming == 0:
                slow_down_rate = math.inf
            else:
                slow_down_rate = (
                    processing_image_time_consuming / no_processing_image_time_consuming
                )
            print(
                "processing image time consuming: %ss."
                % processing_image_time_consuming
            )
            print(
                "no processing image time consuming: %ss."
                % no_processing_image_time_consuming
            )
            print("slow-down rate: %s x." % slow_down_rate)
        self._decoded = True
        return self

    @cached_property
    def _width(self) -> int:
        """Get PNG image width

        Returns:
            int: image width as integer
        """
        return int.from_bytes(self._binary_data[16:20], byteorder="big")

    @cached_property
    def _height(self) -> int:
        """Get PNG image height

        Returns:
            int: image height as integer
        """
        return int.from_bytes(self._binary_data[20:24], byteorder="big")

    @cached_property
    def _bit_depth(self) -> int:
        """Get PNG bit depth

        Returns:
            int: Literal[1, 2, 4, 8, 16]
        """
        return self._binary_data[24]

    @cached_property
    def _color_type(self) -> int:
        """Get PNG color type

        Returns:
            int: Literal[0, 2, 3, 4, 6]
        """
        return self._binary_data[25]

    @cached_property
    def _compression_method(self) -> int:
        """Get PNG compression method

        Returns:
            int: Literal[0]
        """
        return self._binary_data[26]

    @cached_property
    def _filter_method(self) -> int:
        """Get PNG compression method

        Returns:
            int: Literal[0]
        """
        return self._binary_data[27]

    @cached_property
    def _interlace_method(self) -> int:
        """Get PNG interlace method

        Returns:
            int: Literal[0, 1]
        """
        return self._binary_data[28]

    @cached_property
    def _crc32(self) -> int:
        """Get PNG header crc32

        Returns:
            int: crc32 as integer
        """
        return int.from_bytes(self._binary_data[29:33], byteorder="big")

    @cached_property
    def _palette(self) -> RowType:
        """Get PNG palette

        Returns:
            RowType: list of RGBA values in tuple
        """
        index = self._binary_data.find(self.PLTE)
        if index == -1:
            raise FileCorruptionError(
                "missing palette chunk. The image may be corrupted."
            )
        index -= 4
        data_length = int.from_bytes(
            self._binary_data[index : index + 4], byteorder="big"
        )
        plte_index, plte_length, crc32_length = index + 4, 4, 4
        if binascii.crc32(
            self._binary_data[plte_index : plte_index + plte_length + data_length]
        ) != int.from_bytes(
            self._binary_data[
                plte_index
                + plte_length
                + data_length : plte_index
                + plte_length
                + data_length
                + crc32_length
            ],
            byteorder="big",
        ):
            raise FileCorruptionError("invalid crc32. The image may be corrupted.")
        binary_data = self._binary_data[
            plte_index + plte_length : plte_index + plte_length + data_length
        ]
        return [
            (
                binary_data[pixel_index * 3],
                binary_data[pixel_index * 3 + 1],
                binary_data[pixel_index * 3 + 2],
                self.DEFAULT_ALPHA_VALUE,
            )
            for pixel_index in range(0, len(binary_data) // 3)
        ]

    @cached_property
    def image_path(self) -> str:
        return self._filepath.as_posix()

    @cached_property
    def image_size(self) -> tuple[int, int]:
        """Get PNG image size

        Returns:
            tuple: width and height in tuple
        """
        return self._width, self._height

    @cached_property
    def image_data(self) -> ImageType:
        """Get PNG image data

        Returns:
            ImageType: 2D list of RGBA values in tuple
        """
        return self.decode()._image_data


class PNGSequence(object):
    def __init__(self, dirpath: str, debug: bool = False) -> None:
        """Create an object of PNGSequence

        Args:
            dirpath (str): path to directory containing multiple PNG images
            debug (bool, optional): True to print debug information after PNG images decoded.\
                                    Defaults to False

        Exceptions:
            FileNotFoundError: raise FileNotFoundError if path to dirpath does not exists
            NotADirectoryError: raise IsADirectoryError if path to dirpath is not a directory
        """
        image_dirpath = Path(dirpath)
        if not image_dirpath.exists():
            raise FileNotFoundError("directory %s does not exists." % dirpath)
        if not image_dirpath.is_dir():
            raise NotADirectoryError("%s is not a directory." % dirpath)
        self._filepaths = sorted(
            filter(
                lambda filepath: filepath.stem.lstrip(
                    "%s-" % image_dirpath.name
                ).isdecimal(),
                image_dirpath.glob("%s-*" % image_dirpath.name),
            ),
            key=lambda filepath: int(
                filepath.name.lstrip("%s-" % image_dirpath.name).rstrip(".png")
            ),
        )
        self._debug = debug
        self._decoded = False
        self._sequence = None

    def __iter__(self) -> "PNGSequence":
        """Create a Generator of PNG objects and make PNGSequence object iterable.

        Returns:
            PNGSequence: the PNGSequence object itself
        """
        if self._sequence is None:
            self._sequence = (
                PNG(filepath.as_posix(), debug=self._debug)
                for filepath in self._filepaths
            )
        return self

    def __next__(self) -> PNG:
        """Return PNG object in each iteration

        Returns:
            PNG: the next PNG object in the Generator of PNG objects
        """
        if self._sequence is None:
            self._sequence = (
                PNG(filepath.as_posix(), debug=self._debug)
                for filepath in self._filepaths
            )
        return next(self._sequence)

    def decode_all(
        self, runtime: bool = True, warning: bool = True, memory_threshold: float = 8.00
    ) -> "PNGSequence":
        """Decode all the PNG images in sequence for future use

        Args:
            runtime (bool): True to decode PNG images during iteration of image_data_sequence.\
                            Defaults to True
            warning (bool): True to print warning message before decoding PNG images.\
                            Defaults to True
            memory_threshold (float): Memory warning threshold for decoding PNG images at once.\
                                      Defaults to 8.00

        Returns:
            PNGSequence: the PNGSequence object itself
        """
        if self._decoded:
            return self
        if runtime:
            if warning:
                warnings.warn(
                    "decoding PNG images at runtime may result in low fps.",
                    RuntimeWarning,
                )
            self._image_data_sequence = (
                PNG(filepath.as_posix(), debug=self._debug).image_data
                for filepath in self._filepaths
            )
        else:
            png_sequence = [
                PNG(filepath.as_posix(), debug=self._debug)
                for filepath in self._filepaths
            ]
            expected_memory = sum(
                width * height * 4 * 24 * 1e-9
                for width, height in (image.image_size for image in png_sequence)
            )
            if warning and expected_memory >= memory_threshold:
                warnings.warn(
                    (
                        "decoding %d PNG images at once requires memory size of %.2fGB "
                        "while the memory size of warning threshold is %.2fGB "
                        "and may result in memory overflow."
                    )
                    % (len(self._filepaths), expected_memory, memory_threshold),
                    RuntimeWarning,
                )
            self._image_data_sequence = [image.image_data for image in png_sequence]
        self._decoded = True
        return self

    @cached_property
    def image_data_sequence(self) -> ImagesType:
        """Get the data of all PNG images in sequence

        Returns:
            ImagesType: list or Generator of 2D list of tuples of RGBA values
        """
        return self.decode_all()._image_data_sequence


if __name__ == "__main__":
    png_filepath = "resource/images/example.png"
    png_dirpath = "resource/images/sequence/tom&jerry"
    example_png = PNG(png_filepath, debug=True)
    print("size of %s: %dx%d" % (example_png.image_path, *example_png.image_size))
    example_png_sequence = PNGSequence(png_dirpath, debug=True)
    for example_png in example_png_sequence:
        print("size of %s: %dx%d" % (example_png.image_path, *example_png.image_size))
