# cython: language_level=3
cdef class Tokenizer:
    cdef str cur_chunk
    cdef object chunk_iter
    # Class to call when errors occur..
    cdef object error_type

    cdef public str filename

    cdef int char_index # Position inside cur_chunk

    cdef public int line_num
    cdef public bint string_bracket
    cdef public bint allow_escapes
    cdef public bint allow_star_comments

    cdef object pushback_tok
    cdef object pushback_val

    # Private buffer, to hold string parts we're constructing.
    # Tokenizers are expected to be temporary, so we just never shrink.
    cdef Py_ssize_t buf_size  # 2 << x
    cdef unsigned int buf_pos
    cdef Py_UCS4* val_buffer

    cdef inline _error(self, str message)
    cdef inline void buf_reset(self)
    cdef inline void buf_add_char(self, Py_UCS4 uchar)

    cdef object buf_get_text(self)
    cdef Py_UCS4 _next_char(self) except -2
    cdef tuple next_token(self)

    cdef void _push_back(self, object tok, str value=?)

    cpdef peek(self)
    cdef _expect(self, object token, bint skip_newline=?)
