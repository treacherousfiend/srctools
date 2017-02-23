"""Read and write lumps in Source BSP files.

"""
from enum import Enum, IntFlag
from io import BytesIO
from srctools import AtomicWriter, Vec
import struct

from typing import List, Tuple, Iterator, Union

__all__ = [
    'BSP_LUMPS', 'VERSIONS'
    'BSP', 'Lump', 'StaticProp',
]

BSP_MAGIC = b'VBSP'  # All BSP files start with this


def get_struct(file, format):
    """Get a structure from a file."""
    length = struct.calcsize(format)
    data = file.read(length)
    return struct.unpack_from(format, data)


class VERSIONS(Enum):
    """The BSP version numbers for various games."""
    VER_17 = 17
    VER_18 = 18
    VER_19 = 19
    VER_20 = 20
    VER_21 = 21
    VER_22 = 22
    VER_29 = 29

    HL2 = 19
    CS_SOURCE = 19
    DOF_SOURCE = 19
    HL2_EP1 = 20
    HL2_EP2 = 20
    HL2_LC = 20

    GARYS_MOD = 20
    TF2 = 20
    PORTAL = 20
    L4D = 20
    ZENO_CLASH = 20
    DARK_MESSIAH = 20
    VINDICTUS = 20
    THE_SHIP = 20

    BLOODY_GOOD_TIME = 20
    L4D2 = 21
    ALIEN_SWARM = 21
    PORTAL_2 = 21
    CS_GO = 21
    DEAR_ESTHER = 21
    STANLEY_PARABLE = 21
    DOTA2 = 22
    CONTAGION = 23


class BSP_LUMPS(Enum):
    """All the lumps in a BSP file.

    The values represent the order lumps appear in the index.
    Some indexes were reused, so they have aliases.
    """
    ENTITIES = 0
    PLANES = 1
    TEXDATA = 2
    VERTEXES = 3
    VISIBILITY = 4
    NODES = 5
    TEXINFO = 6
    FACES = 7
    LIGHTING = 8
    OCCLUSION = 9
    LEAFS = 10
    FACEIDS = 11
    EDGES = 12
    SURFEDGES = 13
    MODELS = 14
    WORLDLIGHTS = 15
    LEAFFACES = 16
    LEAFBRUSHES = 17
    BRUSHES = 18
    BRUSHSIDES = 19
    AREAS = 20
    AREAPORTALS = 21

    PORTALS = 22
    UNUSED0 = 22
    PROPCOLLISION = 22

    CLUSTERS = 23
    UNUSED1 = 23
    PROPHULLS = 23

    PORTALVERTS = 24
    UNUSED2 = 24
    PROPHULLVERTS = 24

    CLUSTERPORTALS = 25
    UNUSED3 = 25
    PROPTRIS = 25

    DISPINFO = 26
    ORIGINALFACES = 27
    PHYSDISP = 28
    PHYSCOLLIDE = 29
    VERTNORMALS = 30
    VERTNORMALINDICES = 31
    DISP_LIGHTMAP_ALPHAS = 32
    DISP_VERTS = 33
    DISP_LIGHTMAP_SAMPLE_POSITIONS = 34
    GAME_LUMP = 35
    LEAFWATERDATA = 36
    PRIMITIVES = 37
    PRIMVERTS = 38
    PRIMINDICES = 39
    PAKFILE = 40
    CLIPPORTALVERTS = 41
    CUBEMAPS = 42
    TEXDATA_STRING_DATA = 43
    TEXDATA_STRING_TABLE = 44
    OVERLAYS = 45
    LEAFMINDISTTOWATER = 46
    FACE_MACRO_TEXTURE_INFO = 47
    DISP_TRIS = 48
    PHYSCOLLIDESURFACE = 49
    PROP_BLOB = 49
    WATEROVERLAYS = 50

    LIGHTMAPPAGES = 51
    LEAF_AMBIENT_INDEX_HDR = 51

    LIGHTMAPPAGEINFOS = 52
    LEAF_AMBIENT_INDEX = 52
    LIGHTING_HDR = 53
    WORLDLIGHTS_HDR = 54
    LEAF_AMBIENT_LIGHTING_HDR = 55
    LEAF_AMBIENT_LIGHTING = 56
    XZIPPAKFILE = 57
    FACES_HDR = 58
    MAP_FLAGS = 59
    OVERLAY_FADES = 60
    OVERLAY_SYSTEM_LEVELS = 61
    PHYSLEVEL = 62
    DISP_MULTIBLEND = 63

LUMP_COUNT = max(lump.value for lump in BSP_LUMPS) + 1  # 64 normally


class BRUSH_CONTENTS(IntFlag):
    """Flags for brushes, used to define their collisions."""
    EMPTY         = 0x0
    # an eye is never valid in a solid
    SOLID         = 0x1
    # translucent, but not watery (glass)
    WINDOW        = 0x2
    AUX           = 0x4
    # alpha-tested "grate" textures. Bullets/sight pass through, but solids don't
    GRATE         = 0x8
    # Variant of water
    SLIME         = 0x10
    # Liquid      =
    WATER         = 0x20
    MIST          = 0x40

    # block AI line of sight
    OPAQUE        = 0x80

    # things that cannot be seen through (may be non-solid though)
    TESTFOGVOLUME = 0x100
    # Unused
    UNUSED        = 0x200
    UNUSED6       = 0x400
    # per team contents used to differentiate collisions between players
    # and objects on different teams
    TEAM1         = 0x800
    TEAM2         = 0x1000

    # ignore CONTENTS_OPAQUE on surfaces that have SURF_NODRAW
    IGNORE_NODRAW_OPAQUE = 0x200

    # hits entities which are MOVETYPE_PUSH (doors, plats, etc.)
    MOVEABLE      = 0x4000

    # remaining contents are non-visible, and don't eat brushes
    AREAPORTAL    = 0x8000
    PLAYERCLIP    = 0x10000
    MONSTERCLIP   = 0x20000

    # currents can be added to any other contents, and may be mixed
    CURRENT_0     = 0x40000
    CURRENT_90    = 0x80000
    CURRENT_180   = 0x100000
    CURRENT_270   = 0x200000
    CURRENT_UP    = 0x400000
    CURRENT_DOWN  = 0x800000

    # removed before bsping an entity
    ORIGIN        = 0x1000000
    # should never be on a brush, only in game
    MONSTER       = 0x2000000

    DEBRIS        = 0x4000000
    # func_detail brushes
    DETAIL        = 0x8000000
    TRANSLUCENT   = 0x10000000
    LADDER        = 0x20000000
    HITBOX        = 0x40000000  # use accurate hitboxes on trace


class TexInfoFlag(IntFlag):
    """Flags for individual faces."""
    # Value will hold the light strength
    LIGHT     = 0x1
    # Don't draw, indicates we should skylight + draw 2d sky but not draw the 3D skybox
    SKY2D     = 0x2
    # Don't draw, but add to skybox
    SKY       = 0x4
    # Turbulent water warp
    WARP      = 0x8
    # Texture is translucent
    TRANS     = 0x10
    # The surface can not have a portal placed on it
    NOPORTAL  = 0x20
    # This is an xbox hack to work around elimination of trigger surfaces,
    # which breaks occluders
    TRIGGER   = 0x40
    # Don't bother referencing the texture
    NODRAW    = 0x80
    # Make a primary bsp splitter
    HINT      = 0x100
    # Completely ignore, allowing non-closed brushes
    SKIP      = 0x200
    # Don't calculate light
    NOLIGHT   = 0x400
    # Calculate three lightmaps for the surface for bumpmapping
    BUMPLIGHT = 0x800
    # Don't receive shadows
    NOSHADOWS = 0x1000
    # Don't receive decals
    NODECALS  = 0x2000
    # Don't subdivide patches on this surface
    NOCHOP    = 0x4000
    # Surface is part of a hitbox
    HITBOX    = 0x8000


class BSP:
    """A BSP file."""
    def __init__(self, filename, version=VERSIONS.PORTAL_2):
        self.filename = filename
        self.map_revision = -1  # The map's revision count
        self.lumps = {}
        self.game_lumps = {}
        self.header_off = 0
        self.version = version

    def read_header(self):
        """Read through the BSP header to find the lumps.

        This allows locating any data in the BSP.
        """
        with open(self.filename, mode='br') as file:
            # BSP files start with 'VBSP', then a version number.
            magic_name, bsp_version = get_struct(file, '4si')
            assert magic_name == BSP_MAGIC, 'Not a BSP file!'

            assert bsp_version == self.version.value, 'Different BSP version!'

            # Read the index describing each BSP lump.
            for index in range(LUMP_COUNT):
                lump = Lump.from_bytes(index, file)
                self.lumps[lump.type] = lump

            # Remember how big this is, so we can remake it later when needed.
            self.header_off = file.tell()

    def get_lump(self, lump):
        """Read a lump from the BSP."""
        if isinstance(lump, BSP_LUMPS):
            lump = self.lumps[lump]
        with open(self.filename, 'rb') as file:
            file.seek(lump.offset)
            return file.read(lump.length)

    def replace_lump(self, new_name, lump, new_data: bytes):
        """Write out the BSP file, replacing a lump with the given bytes.

        """
        if isinstance(lump, BSP_LUMPS):
            lump = self.lumps[lump]
        with open(self.filename, 'rb') as file:
            data = file.read()

        before_lump = data[self.header_off:lump.offset]
        after_lump = data[lump.offset + lump.length:]
        del data  # This contains the entire file, we don't want to keep
        # this memory around for long.

        # Adjust the length to match the new data block.
        lump.length = len(new_data)

        with AtomicWriter(new_name, is_bytes=True) as file:
            self.write_header(file)
            file.write(before_lump)
            file.write(new_data)
            file.write(after_lump)

    def write_header(self, file):
        """Write the BSP file header into the given file."""
        file.write(BSP_MAGIC)
        file.write(struct.pack('i', self.version.value))
        for lump_name in BSP_LUMPS:
            # Write each header
            lump = self.lumps[lump_name]
            file.write(lump.as_bytes())
        # The map revision would follow, but we never change that value!

    def read_game_lumps(self):
        """Read in the game-lump's header, so we can get those values."""
        game_lump = BytesIO(self.get_lump(BSP_LUMPS.GAME_LUMP))

        self.game_lumps.clear()
        lump_count = get_struct(game_lump, 'i')[0]

        for _ in range(lump_count):
            (
                lump_id,
                flags,
                version,
                file_off,
                file_len,
            ) = get_struct(game_lump, '<4s HH ii')
            # The lump ID is backward..
            self.game_lumps[lump_id[::-1]] = (flags, version, file_off, file_len)

    def get_game_lump(self, lump_id):
        """Get a given game-lump, given the 4-character byte ID."""
        flags, version, file_off, file_len = self.game_lumps[lump_id]
        with open(self.filename, 'rb') as file:
            file.seek(file_off)
            return file.read(file_len)

    # Lump-specific commands:

    def read_texture_names(self) -> List[str]:
        """Iterate through all brush textures in the map."""
        tex_data = self.get_lump(BSP_LUMPS.TEXDATA_STRING_DATA)
        tex_table = self.get_lump(BSP_LUMPS.TEXDATA_STRING_TABLE)
        # tex_table is an array of int offsets into tex_data. tex_data is a
        # null-terminated block of strings.

        table_offsets = struct.unpack(
            # The number of ints + i, for the repetitions in the struct.
            str(len(tex_table) // struct.calcsize('i')) + 'i',
            tex_table,
        )

        out_table = []

        for off in table_offsets:
            # Look for the NULL at the end - strings are limited to 128 chars.
            str_off = 0
            for str_off in range(off, off + 128):
                if tex_data[str_off] == 0:
                    out_table.append(tex_data[off: str_off].decode('ascii'))
                    break
            else:
                # Reached the 128 char limit without finding a null.
                raise ValueError('Bad string at', off, 'in BSP! ("{}")'.format(
                    tex_data[off:str_off]
                ))

        return out_table

    def read_texinfo(self) -> List['TexInfo']:
        """Read the texinfo blocks, which describe textures."""
        tex_table = self.read_texture_names()

        struct_texdata = struct.Struct('fffiiiii')
        struct_texinfo = struct.Struct('16fii')

        texdata_list = []
        texinfo_list = []

        for (
            ref_x, ref_y, ref_z,
            tex_off,
            height, width,
            view_width, view_height,
        ) in struct_texdata.iter_unpack(self.get_lump(BSP_LUMPS.TEXDATA)):
            texdata_list.append(TexData(
                tex_table[tex_off],
                Vec(ref_x, ref_y, ref_z),
                height, width,
                view_width, view_height,
            ))

        for (
            tex_sx, tex_sy, tex_sz, tex_s_off,
            tex_tx, tex_ty, tex_tz, tex_t_off,
            light_sx, light_sy, light_sz, light_s_off,
            light_tx, light_ty, light_tz, light_t_off,
            flags, texdata_off,
        ) in struct_texinfo.iter_unpack(self.get_lump(BSP_LUMPS.TEXINFO)):
            texinfo_list.append(TexInfo(
                Vec(tex_sx, tex_sy, tex_sz),
                tex_s_off,
                Vec(tex_tx, tex_ty, tex_tz),
                tex_t_off,

                Vec(light_sx, light_sy, light_sz),
                light_s_off,
                Vec(light_tx, light_ty, light_tz),
                light_t_off,

                texdata_list[texdata_off],
                TexInfoFlag(flags),
            ))

        return texinfo_list

    def read_brushes(self) -> List['Brush']:
        """Return the list of brush data (not the faces).

        This returns a list of ([BrushSide], contents) tuples.
        """
        sides_data = self.get_lump(BSP_LUMPS.BRUSHSIDES)
        brush_sides = []
        for plane_num, texinfo, dispinfo, bevel in struct.iter_unpack('Hhhh', sides_data):
            brush_sides.append(BrushSide(plane_num, texinfo, dispinfo, bevel))

        brush_data = self.get_lump(BSP_LUMPS.BRUSHES)

        brushes = []

        for first_side, side_len, contents in struct.iter_unpack('iii', brush_data):
            brushes.append(Brush(
                brush_sides[first_side: first_side + side_len],
                BRUSH_CONTENTS(contents),
            ))

        return brushes

    def write_brushes(self, brushes: List['Brush']):
        """Write the brush and brushsides lumps back into the BSP."""
        brush_data = BytesIO()
        brush_sides = BytesIO()

        cur_off = 0

        struct_brush = struct.Struct('iii')
        struct_sides = struct.Struct('Hhhh')

        for brush in brushes:
            brush_data.write(struct_brush.pack(
                cur_off,
                len(brush.sides),
                int(brush.contents),
            ))
            cur_off += len(brush.sides)

            for side in brush:
                brush_sides.write(struct_sides.pack(
                    side.plane_num,
                    side.texinfo_off,
                    side.dispinfo,
                    side.bevel,
                ))

        self.replace_lump(self.filename, BSP_LUMPS.BRUSHES, brush_data.getvalue())
        self.replace_lump(self.filename, BSP_LUMPS.BRUSHSIDES, brush_sides.getvalue())


    def read_ent_data(self):
        """Iterate through the entities in a map.

        This yields a series of keyvalue dictionaries. The first is WorldSpawn.
        """
        ent_data = self.get_lump(BSP_LUMPS.ENTITIES).decode('ascii')
        cur_dict = None  # None = waiting for '{'

        # This code is similar to property_parser, but simpler since there's
        # no nesting, comments, or whitespace, except between key and value.
        for line in ent_data.splitlines():
            if line == '{':
                cur_dict = {}
            elif line == '}':
                yield cur_dict
                cur_dict = None
            elif line == '\x00':
                return
            else:
                # Line is of the form <"key" "val">
                key, value = line.split('" "')
                cur_dict[key[1:]] = value[:-1]

    @staticmethod
    def write_ent_data(ent_dicts):
        """Generate the entity data lump, given a list of dictionaries."""
        out = BytesIO()
        for keyvals in ent_dicts:
            out.write(b'{\n')
            for key, value in keyvals.items():
                out.write('"{}" "{}"'.format(key, value).encode('ascii'))
            out.write(b'}\n')
        out.write(b'\x00')

        return out.getvalue()

    def read_static_props(self) -> Iterator['StaticProp']:
        """Read in the Static Props lump."""
        # The version of the static prop format - different features.
        version = self.game_lumps[b'sprp'][1]
        if version > 9:
            raise ValueError('Unknown version "{}"!'.format(version))

        static_lump = BytesIO(self.get_game_lump(b'sprp'))
        dict_num = get_struct(static_lump, 'i')[0]

        # Array of model filenames.
        model_dict = []
        for _ in range(dict_num):
            padded_name = get_struct(static_lump, '128s')[0]
            # Strip null chars off the end, and convert to a str.
            model_dict.append(
                padded_name.rstrip(b'\x00').decode('ascii')
            )

        visleaf_count = get_struct(static_lump, 'i')[0]
        visleaf_list = list(get_struct(static_lump, 'H' * visleaf_count))

        prop_count = get_struct(static_lump, 'i')[0]

        print(model_dict)

        print('-' * 30)
        print('props', version, prop_count)
        print('-' * 30)

        pos = static_lump.tell()
        data = static_lump.read()
        static_lump.seek(pos)
        for i in range(12, 200, 12):
            vals = Vec(struct.unpack_from('fff', data, i))
            # if vals: and vals == round(vals):
            print(i, repr(vals))

        print(flush=True)
        for i in range(prop_count):
            origin = Vec(get_struct(static_lump, 'fff'))
            angles = Vec(get_struct(static_lump, 'fff'))
            (
                model_ind,
                first_leaf,
                leaf_count,
                solidity,
                flags,
                skin,
                min_fade,
                max_fade,
            ) = get_struct(static_lump, 'HHHBBiff')

            model_name = model_dict[model_ind]

            visleafs = visleaf_list[first_leaf:first_leaf + leaf_count]
            lighting_origin = Vec(get_struct(static_lump, 'fff'))

            if version >= 5:
                fade_scale = get_struct(static_lump, 'f')[0]
            else:
                fade_scale = 1  # default

            if version in (6, 7):
                min_dx_level, max_dx_level = get_struct(static_lump, 'HH')
            else:
                # Replaced by GPU & CPU in later versions.
                min_dx_level = max_dx_level = 0  # None

            if version >= 8:
                (
                    min_cpu_level,
                    max_cpu_level,
                    min_gpu_level,
                    max_gpu_level,
                ) = get_struct(static_lump, 'BBBB')
            else:
                # None
                min_cpu_level = max_cpu_level = min_gpu_level = max_gpu_level = 0

            if version >= 7:
                r, g, b, a = get_struct(static_lump, 'BBBB')
                # Alpha isn't used.
                tint = Vec(r, g, b)
            else:
                # No tint.
                tint = Vec(255, 255, 255)
            if version >= 9:
                disable_on_xbox = get_struct(static_lump, '?')[0]
            else:
                disable_on_xbox = False

            # Unknown padding...
            static_lump.read(3)

            yield StaticProp(
                model_name,
                origin,
                angles,
                visleafs,
                solidity,
                flags,
                skin,
                min_fade,
                max_fade,
                lighting_origin,
                fade_scale,
                min_dx_level,
                max_dx_level,
                min_cpu_level,
                max_cpu_level,
                min_gpu_level,
                max_gpu_level,
                tint,
                disable_on_xbox,
            )


class Lump:
    """Represents a lump header in a BSP file.

    These indicate the location and size of each component.
    """
    def __init__(self, index, offset, length, version, ident):
        self.type = BSP_LUMPS(index)
        self.offset = offset
        self.length = length
        self.version = version
        self.ident = [int(x) for x in ident]

    @classmethod
    def from_bytes(cls, index, file):
        """Decode this header from the file."""
        offset, length, version, ident = get_struct(
            file,
            # 3 ints and a 4-long char array
            '<3i4s',
        )
        return cls(
            index=index,
            offset=offset,
            length=length,
            version=version,
            ident=ident,
        )

    def as_bytes(self):
        """Get the binary version of this lump header."""
        return struct.pack(
            '<3i4s',
            self.offset,
            self.length,
            self.version,
            bytes(self.ident),
        )

    def __len__(self):
        return self.length

    def __repr__(self):
        return (
            'Lump({s.type}, {s.offset}, '
            '{s.length}, {s.version}, {s.ident})'.format(
                s=self
            )
        )


class BrushSide:
    """Represents a side of a brush in the BSP."""
    __slots__ = ['plane_num', 'texinfo_off', 'dispinfo', 'bevel']

    def __init__(self, plane_num, texinfo_off: int, dispinfo, bevel):
        self.plane_num = plane_num
        self.texinfo_off = texinfo_off
        self.dispinfo = dispinfo
        self.bevel = bevel

    def __repr__(self):
        return 'BrushSide({}, {}, {}, {})'.format(
            self.plane_num,
            self.texinfo_off,
            self.dispinfo,
            self.bevel,
        )


class Brush:
    """Represents a brush in the BSP."""
    __slots__ = ['sides', 'contents']

    def __init__(self, sides: List[BrushSide], contents: BRUSH_CONTENTS):
        self.sides = sides
        self.contents = contents

    def __iter__(self):
        return iter(self.sides)

    def __repr__(self):
        return 'Brush(contents={!r}, sides={!r})'.format(self.contents, self.sides)

class TexData:
    __slots__ = [
        'texture', 'reflection',
        'width', 'height',
        'view_width', 'view_height',
    ]
    def __init__(
        self,
        texture: str,
        reflection: Vec,
        width: float, height: float,
        view_width: float, view_height: float,
    ):
        self.texture = texture
        self.reflection = reflection
        self.width = width
        self.height = height
        self.view_height = view_height
        self.view_width = view_width


class TexInfo:
    """Represents data for a texture - face scaling and lightmap."""
    __slots__ = [
        'tex_s', 'tex_t', 'tex_s_off', 'tex_t_off',
        'light_s', 'light_t', 'light_s_off', 'light_t_off',
        'texdata', 'flags',
    ]
    def __init__(
        self,
        tex_s: Vec,
        tex_s_off: float,
        tex_t: Vec,
        tex_t_off: float,
        light_s: Vec,
        light_s_off: float,
        light_t: Vec,
        light_t_off: float,
        texdata: TexData,
        flags: TexInfoFlag,
    ):
        self.tex_s = tex_s
        self.tex_s_off = tex_s_off
        self.tex_t = tex_t
        self.tex_t_off = tex_t_off

        self.light_s = light_s
        self.light_s_off = light_s_off
        self.light_t = light_t
        self.light_t_off = light_t_off
        self.texdata = texdata
        self.flags = flags


class StaticProp:
    """Represents a prop_static in the BSP.

    Different features were added in different versions.
    v5+ allows fade_scale.
    v6 and v7 allow min/max DXLevel.
    v8+ allows min/max GPU and CPU levels.
    v7+ allows model tinting.
    v9+ allows disabling on XBox 360.
    """
    def __init__(
        self,
        model: str,
        origin: Vec,
        angles: Vec,
        visleafs: List[int],
        solidity: int,
        flags: int,
        skin: int,
        min_fade: float,
        max_fade: float,
        lighting_origin: Vec,
        fade_scale: float,
        min_dx_level: int,
        max_dx_level: int,
        min_cpu_level: int,
        max_cpu_level: int,
        min_gpu_level: int,
        max_gpu_level: int,
        tint: Vec,  # Rendercolor
        disable_on_xbox: bool,
    ):
        self.model = model
        self.origin = origin
        self.angles = angles
        self.visleafs = visleafs
        self.solidity = solidity
        self.flags = flags
        self.skin = skin
        self.min_fade = min_fade
        self.max_fade = max_fade
        self.lighting = lighting_origin
        self.fade_scale = fade_scale
        self.min_dx_level = min_dx_level
        self.max_dx_level = max_dx_level
        self.min_cpu_level = min_cpu_level
        self.max_cpu_level = max_cpu_level
        self.min_gpu_level = min_gpu_level
        self.max_gpu_level = max_gpu_level
        self.tint = tint
        self.disable_on_xbox = disable_on_xbox
