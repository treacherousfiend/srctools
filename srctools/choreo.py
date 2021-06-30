"""Parses VCD choreo scenes, as well as data in scenes.image."""
import struct
from typing import List, IO, NewType, Dict, cast, Tuple
import attr
import lzma

from srctools.binformat import struct_read, read_nullstr, checksum as raw_checksum


CRC = NewType('CRC', int)
LZMA_DIC_MIN = (1 << 12)

def checksum_filename(filename: str) -> CRC:
    """Normalise the filename, then checksum it."""
    filename = filename.lower().replace('/', '\\')
    if not filename.startswith('scenes\\'):
        filename = 'scenes\\' + filename
    return cast(CRC, raw_checksum(filename.encode('ascii')))


def decompress(data: bytes) -> bytes:
    """Decompress LZMA, with Source's weird header.

    This means we have to decode all that ourself and setup the decoder.
    """
    if data[:4] != b'LZMA':
        raise ValueError(f'Invalid signature {data[:12]!r}')
    (uncomp_size, comp_size) = struct.unpack_from('<ii', data, 4)
    if len(data) - 17 != comp_size:
        raise ValueError(
            f"File size doesn't match. Got {len(data) - 17:,} "
            f"bytes, expected {comp_size:,} bytes"
        )
    # Parse properties,
    # region Code from LZMA spec
    d = data[12]
    if d >= (9 * 5 * 5):
        raise ValueError("Incorrect LZMA properties")
    lc = d % 9
    d //= 9
    pb = d // 5
    lp = d % 5
    dict_size = 0
    for i in range(4):
        dict_size |= data[12 + i + 1] << (8 * i)
    if dict_size < LZMA_DIC_MIN:
        dict_size = LZMA_DIC_MIN

    decomp = lzma.LZMADecompressor(lzma.FORMAT_RAW, None, filters=[
        {
            'id': lzma.FILTER_LZMA1,
            'dict_size': dict_size,
            'lc': lc,
            'lp': lp,
            'pb': pb,
        },
    ])
    # This technically leaves the decompressor in an incomplete state, but the
    # stream doesn't contain an EOF marker, so ignore that.
    return decomp.decompress(data[17:])


@attr.define
class Choreo:
    """A choreographed scene."""
    filename: str  # Filename if available.
    checksum: CRC  # CRC hash.
    duration_ms: int  # Duration in milliseconds.
    last_speak_ms: int  # Time at which the last voice line ends.
    sounds: List[str]  # List of sounds it uses.
    _data: bytes

    @property
    def duration(self) -> float:
        """Return the duration in seconds."""
        return self.duration_ms / 1000.0

    @duration.setter
    def duration(self, value: float) -> None:
        """Set the duration (in seconds). This is rounded to the nearest millisecond."""
        self.duration_ms = round(value * 1000.0)

    @property
    def last_speak(self) -> float:
        """Return the last-speak time in seconds."""
        return self.last_speak_ms / 1000.0

    @last_speak.setter
    def last_speak(self, value: float) -> None:
        """Set the last-speak time (in seconds). This is rounded to the nearest millisecond."""
        self.last_speak_ms = round(value * 1000.0)


def parse_scenes_image(file: IO[bytes]) -> Dict[CRC, Choreo]:
    """Parse the scenes.image file, extracting all the choreo data."""
    [
        magic,
        version,
        scene_count,
        string_count,
        scene_off,
    ] = struct_read('<4s4i', file)
    if magic != b'VSIF':
        raise ValueError("Invalid scenes.image!")
    if version not in (2, 3):
        raise ValueError("Unknown version {}!".format(version))

    # Read the indexes in order from the file, then read the null-terminated
    # string from each offset.
    string_pool = [
        read_nullstr(file, off)
        for off in
        struct_read('<' + 'i' * string_count, file)
    ]

    scenes: dict[CRC, Choreo] = {}

    file.seek(scene_off)
    scene_data: List[Tuple[CRC, int, int, int]] = [()] * scene_count
    for i in range(scene_count):
        scene_data[i] = struct_read('<4i', file)

    for (
        crc,
        data_off, data_size,
        summary_off,
    ) in scene_data:
        file.seek(summary_off)
        if version == 3:
            [duration, last_speak, sound_count] = struct_read('<3i', file)
        else:
            [duration, sound_count] = struct_read('<2i', file)
            last_speak = duration - 1  # Assume it's the whole choreo.
        sounds = [
            string_pool[i]
            for i in struct_read('<{}i'.format(sound_count), file)
        ]
        file.seek(data_off)
        data = file.read(data_size)
        if data.startswith(b'LZMA'):
            data = decompress(data)
        scenes[crc] = Choreo(
            '',
            crc,
            duration, last_speak,
            sounds,
            data,
        )
    return scenes
