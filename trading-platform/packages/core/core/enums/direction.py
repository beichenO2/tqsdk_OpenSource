"""交易方向与开平标志."""

from enum import StrEnum


class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class Offset(StrEnum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    CLOSE_TODAY = "CLOSE_TODAY"
