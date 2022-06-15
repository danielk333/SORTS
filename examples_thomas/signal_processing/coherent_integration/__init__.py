import os
import ctypes

__cohpath__ = os.path.dirname(__file__)
__libpath__ = __cohpath__ + "/src/libcoh.so"

clibcoh = ctypes.cdll.LoadLibrary(__libpath__)

from . import core
