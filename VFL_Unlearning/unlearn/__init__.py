from .LUV import LUV


def get_unlearn_method(method):
    if method == "LUV":
        return LUV