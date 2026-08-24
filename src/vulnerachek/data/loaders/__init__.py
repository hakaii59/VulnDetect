from vulnerachek.data.loaders.base import BaseLoader
from vulnerachek.data.loaders.bigvul import BigVulLoader
from vulnerachek.data.loaders.cvefixes import CVEfixesLoader
from vulnerachek.data.loaders.megavul import MegaVulLoader
from vulnerachek.data.loaders.primevul import PrimeVulLoader

LOADERS: dict[str, type[BaseLoader]] = {
    "megavul": MegaVulLoader,
    "bigvul": BigVulLoader,
    "cvefixes": CVEfixesLoader,
    "primevul": PrimeVulLoader,
}

__all__ = [
    "BaseLoader",
    "BigVulLoader",
    "CVEfixesLoader",
    "MegaVulLoader",
    "PrimeVulLoader",
    "LOADERS",
]
