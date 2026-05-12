from .constants import DEFAULT_ENCODING


def ustr(x):
    """Python 3 text helper."""
    if isinstance(x, bytes):
        return x.decode(DEFAULT_ENCODING, "ignore")
    return x
