def to_bytes(str_input) -> bytes:
    # Encode to UTF-8 to get binary data.
    if isinstance(str_input, bytes):
        return str_input
    return str_input.encode("utf-8")


def to_string(bytes_input) -> str:
    if isinstance(bytes_input, str):
        return bytes_input
    return bytes_input.decode("utf-8")


def convert_string(bytes_input) -> str:
    try:
        return to_string(bytes_input.decode("utf-8"))
    except AttributeError:  # 'str' object has no attribute 'decode'.
        return str(bytes_input)
    except UnicodeError:
        return str(bytes_input)
