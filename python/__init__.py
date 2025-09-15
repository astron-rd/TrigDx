from .trigdx import Reference, Lookup16K, Lookup32K, LookupAVX16K, LookupAVX32K

try:
    from .trigdx import MKL
except ImportError:
    pass

try:
    from .trigdx import GPU
except ImportError:
    pass

try:
    from .trigdx import LookupXSIMD16K, LookupXSIMD32K
except ImportError:
    pass
