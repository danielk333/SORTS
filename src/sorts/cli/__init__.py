# This is needed so that the registration is performed
from . import spacetrack_download
from . import discos_download

# Then expose the main after registration
from .commands import main
