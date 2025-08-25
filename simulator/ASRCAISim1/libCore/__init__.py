# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

# Just for backward compatibility. This file will be removed in the future.

print(
    "Deprecation warning!",
    "  In the current version libCore.so was moved under ASRCAISim1.core.",
    "You should import objects from ASRCAISim1.core instead of ASRCAISim1.libCore.",
    "This backward compatibility for importing something from ASRCAISim1.libCore",
    "package will be removed in the future.",
)
from ASRCAISim1.core import *
