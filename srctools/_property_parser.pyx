# cython: language_level=3, auto_pickle=False
# cython: binding=True
"""Reimplement part of property_parse in C, to speed it up."""
from srctools._tokenizer cimport Tokenizer
cimport cython

cdef object Token, intern
cdef type KeyValError, Property
cdef dict PROP_FLAGS_DEFAULT

from srctools.property_parser import KeyValError, PROP_FLAGS_DEFAULT, Property, Token
from sys import intern

cdef object EOF = Token.EOF
cdef object STRING = Token.STRING
cdef object PROP_FLAG = Token.PROP_FLAG
cdef object NEWLINE = Token.NEWLINE
cdef object BRACE_OPEN = Token.BRACE_OPEN
cdef object BRACE_CLOSE = Token.BRACE_CLOSE


cdef bint _read_flag(flags, str flag_val):
    """Check whether a flag is True or False."""
    cdef bint flag_result
    cdef bint flag_inv = (
        len(flag_val) > 0 and
        flag_val[0] == '!'
    )
    if flag_inv:
        flag_val = flag_val[1:]
    flag_val = flag_val.casefold()
    if flags is not None:
        try:
            flag_result = bool(flags[flag_val])
        except KeyError:
            flag_result = PROP_FLAGS_DEFAULT.get(flag_val, False)
    else:
        flag_result = PROP_FLAGS_DEFAULT.get(flag_val, False)
    # If flag succeeds
    return flag_inv is not flag_result

# Implement Property.parse in C, to attempt to speed it up.
def parse(
    file_contents,
    str filename = '',
    flags = None,
    bint allow_escapes = True,
):
    """Returns a Property tree parsed from given text.

    filename, if set should be the source of the text for debug purposes.
    file_contents should be an iterable of strings or a single string.
    flags should be a mapping for additional flags to accept
    (which overrides defaults).
    character_escapes allows choosing if \\t or similar escapes are parsed.
    """
    # The block we are currently adding to.

    # The special name 'None' marks it as the root property, which
    # just outputs its children when exported. This way we can handle
    # multiple root blocks in the file, while still returning a single
    # Property object which has all the methods.
    # Skip calling __init__ for speed.
    cur_block = Property.__new__(Property)
    cur_block._folded_name = cur_block.real_name = None
    cur_block.value = []

    # A queue of the properties we are currently in (outside to inside).
    cdef list open_properties = [cur_block]

    # Grab a reference to the token values, so we avoid global lookups.

    cdef Tokenizer tokenizer = Tokenizer(
        file_contents,
        filename,
        KeyValError,
        True,  # string_bracket
        allow_escapes,
        False,  # allow /* comments
    )

    # Do we require a block to be opened next? ("name"\n must have { next.)
    cdef bint requires_block = False
    # Are we permitted to replace the last property with a flagged version of the same?
    cdef bint can_flag_replace = False
    cdef str token_value, prop_value

    while True:
        token_type, token_value = tokenizer.next_token()
        if token_type is EOF:
            break
        elif token_type is BRACE_OPEN:  # {
            # Open a new block - make sure the last token was a name..
            if not requires_block:
                raise tokenizer._error(
                    'Property cannot have sub-section if it already '
                    'has an in-line value.\n\n'
                    'A "name" "value" line cannot then open a block.',
                )
            requires_block = can_flag_replace = False
            cur_block = cur_block.value[-1]
            cur_block.value = []
            open_properties.append(cur_block)
            continue
        # Something else, but followed by '{'
        elif requires_block and token_type is not NEWLINE:
            raise tokenizer._error(
                'Block opening ("{{") required!\n\n'
                'A single "name" on a line should next have a open brace '
                'to begin a block.',
            )

        if token_type is NEWLINE:
            continue
        if token_type is STRING:   # "string"
            # Skip calling __init__ for speed. Value needs to be set
            # before using this, since it's unset here.
            keyvalue = Property.__new__(Property)
            with cython.optimize.unpack_method_calls(False):
                keyvalue._folded_name = intern(token_value.casefold())
                keyvalue.real_name = intern(token_value)

            # We need to check the next token to figure out what kind of
            # prop it is.
            prop_type, prop_value = tokenizer.next_token()

            # It's a block followed by flag. ("name" [stuff])
            if prop_type is PROP_FLAG:
                # That must be the end of the line..
                tokenizer._expect(NEWLINE)
                requires_block = True
                if _read_flag(flags, prop_value):
                    keyvalue.value = []

                    # Special function - if the last prop was a
                    # keyvalue with this name, replace it instead.
                    if (
                        can_flag_replace and
                        cur_block.value[-1].real_name == token_value and
                        type(cur_block.value[-1].value) == list
                    ):
                        cur_block.value[-1] = keyvalue
                    else:
                        cur_block.value.append(keyvalue)
                    # Can't do twice in a row
                    can_flag_replace = False

            elif prop_type is STRING:
                # A value.. ("name" "value")
                if requires_block:
                    raise tokenizer._error(
                        'Keyvalue split across lines!\n\n'
                        'A value like "name" "value" must be on the same '
                        'line.'
                    )
                requires_block = False

                keyvalue.value = prop_value

                # Check for flags.
                flag_token, flag_val = tokenizer()
                if flag_token is PROP_FLAG:
                    # Should be the end of the line here.
                    tokenizer._expect(NEWLINE)
                    if _read_flag(flags, flag_val):
                        # Special function - if the last prop was a
                        # keyvalue with this name, replace it instead.
                        if (
                            can_flag_replace and
                            cur_block.value[-1].real_name == token_value and
                            type(cur_block.value[-1].value) == str
                        ):
                            cur_block.value[-1] = keyvalue
                        else:
                            cur_block.value.append(keyvalue)
                        # Can't do twice in a row
                        can_flag_replace = False
                elif flag_token is STRING:
                    # Specifically disallow multiple text on the same line.
                    # ("name" "value" "name2" "value2")
                    raise tokenizer._error(
                        "Cannot have multiple names on the same line!"
                    )
                else:
                    # Otherwise, it's got nothing after.
                    # So insert the keyvalue, and check the token
                    # in the next loop. This allows braces to be
                    # on the same line.
                    cur_block.value.append(keyvalue)
                    can_flag_replace = True
                    tokenizer._push_back(flag_token, flag_val)
                continue
            else:
                # Something else - treat this as a block, and
                # then re-evaluate the token in the next loop.
                keyvalue.value = []

                requires_block = True
                can_flag_replace = False
                cur_block.value.append(keyvalue)
                tokenizer._push_back(prop_type, prop_value)
                continue

        elif token_type is BRACE_CLOSE:  # }
            # Move back a block
            open_properties.pop()
            try:
                cur_block = open_properties[-1]
            except IndexError:
                # It's empty, we've closed one too many properties.
                raise tokenizer._error(
                    'Too many closing brackets.\n\n'
                    'An extra closing bracket was added which would '
                    'close the outermost level.',
                )
            # For replacing the block.
            can_flag_replace = True
        else:
            raise tokenizer._error(token_type)

    # We're out of data, do some final sanity checks.

    # We last had a ("name"\n), so we were expecting a block
    # next.
    if requires_block:
        raise KeyValError(
            'Block opening ("{") required, but hit EOF!\n'
            'A "name" line was located at the end of the file, which needs'
            ' a {} block to follow.',
            tokenizer.filename,
            line=None,
        )

    # All the properties in the file should be closed,
    # so the only thing in open_properties should be the
    # root one we added.

    if len(open_properties) > 1:
        raise KeyValError(
            'End of text reached with remaining open sections.\n\n'
            "File ended with at least one property that didn't "
            'have an ending "}".',
            tokenizer.filename,
            line=None,
        )
    # Return that root property.
    return open_properties[0]

