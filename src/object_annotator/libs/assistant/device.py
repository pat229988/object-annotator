def resolve_device(preferred=None):
    """Resolve training/inference device with safe fallbacks.

    Priority:
    1) user preferred device (if valid and available)
    2) cuda
    3) mps
    4) cpu
    """

    preferred = (preferred or "").strip().lower()

    try:
        import torch
    except Exception:
        return "cpu"

    def cuda_available():
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def mps_available():
        try:
            return bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
        except Exception:
            return False

    if preferred == "cuda" and cuda_available():
        return "cuda"
    if preferred == "mps" and mps_available():
        return "mps"
    if preferred == "cpu":
        return "cpu"

    if cuda_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"
