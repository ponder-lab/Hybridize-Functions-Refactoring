import os
import tempfile

from common import get_model


def test_save():
    get_model().save_weights(os.path.join(tempfile.mkdtemp(), "w.h5"))
