import logging, typing as t, abc, enum
import numpy as np

logger = logging.getLogger(__name__)


class ControllerProtocol(t.Protocol):
    @abc.abstractmethod
    def close(self) -> None: ...
