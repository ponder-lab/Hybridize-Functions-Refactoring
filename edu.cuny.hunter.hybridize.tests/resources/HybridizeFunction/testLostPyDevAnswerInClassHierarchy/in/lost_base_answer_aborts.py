import os
import sys

# PyDev's completion server dies importing this module, so a lookup into it loses its
# answer (#1121). Imported any other way, it's an ordinary module.
if "_pydev_bundle._pydev_imports_tipper" in sys.modules:
    os.abort()


class Base:
    pass
