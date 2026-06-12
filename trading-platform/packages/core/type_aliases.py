"""Shared type aliases for the trading platform."""

from datetime import datetime
from decimal import Decimal
from typing import TypeAlias

Symbol: TypeAlias = str
Price: TypeAlias = Decimal
Volume: TypeAlias = int
Timestamp: TypeAlias = datetime
