from .LUV_2 import LUV_2


def get_unlearn_method(method):
    if method == "LUV":
        return LUV_2