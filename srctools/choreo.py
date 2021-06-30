"""Parses VCD choreo scenes, as well as data in scenes.image."""
from typing import NamedTuple, List, IO, NewType, Dict, Union, Tuple
import attr

from srctools.binformat import struct_read, read_nullstr


CRC = NewType('CRC', int)


@attr.define
class Choreo:
    """A choreographed scene."""
    filename: str  # Filename if available.
    checksum: CRC  # CRC hash.
    duration: int  # Duration in milliseconds.
    sounds: List[str]  # List of sounds it uses.


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
    if version != 2:
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
    scene_data: List[Tuple[CRC, int]] = [()] * scene_count
    for i in range(scene_count):
        [
            crc,
            data_off,
            data_size,
            summary_off,
        ] = struct_read('<4i', file)
        scene_data[i] = (crc, summary_off)

    for crc, summary_off in scene_data:
        file.seek(summary_off)
        [duration, sound_count] = struct_read('<2i', file)
        sounds = [
            string_pool[i]
            for i in struct_read('<{}i'.format(sound_count), file)
        ]
        scenes[crc] = Choreo('', crc, duration, sounds)
    return scenes
