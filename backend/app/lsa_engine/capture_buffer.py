from __future__ import annotations

from dataclasses import dataclass

from app.lsa_engine.captured_item import CapturedFrameItem


@dataclass(frozen=True)
class CaptureBufferState:
    captured_items_count: int
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None
    duration_ms: int | None


class CaptureBuffer:
    """
    Buffer de captura completa.

    A diferencia de un buffer temporal fijo de 20 frames, este buffer guarda
    todos los frames procesados del intento o segmento actual.

    Luego SequenceBuilder se encarga de muestrear 20 frames desde este buffer.
    """

    def __init__(self):
        self._items: list[CapturedFrameItem] = []

    def add(self, item: CapturedFrameItem) -> CaptureBufferState:
        self._items.append(item)
        return self.state

    def reset(self) -> None:
        self._items.clear()

    def get_items(self) -> list[CapturedFrameItem]:
        return list(self._items)

    def to_legacy_captured_items(self) -> list[dict]:
        return [item.to_legacy_dict() for item in self._items]

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def state(self) -> CaptureBufferState:
        if not self._items:
            return CaptureBufferState(
                captured_items_count=0,
                first_timestamp_ms=None,
                last_timestamp_ms=None,
                duration_ms=None,
            )

        first = self._items[0].timestamp_ms
        last = self._items[-1].timestamp_ms

        return CaptureBufferState(
            captured_items_count=len(self._items),
            first_timestamp_ms=first,
            last_timestamp_ms=last,
            duration_ms=max(0, last - first),
        )