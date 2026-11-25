"""Geofence filter for incoming channel messages.

Spam that keeps coming from one area can be recognised by where it enters the
mesh: the first repeaters in a flood packet's path are near the sender. Sender
names on channels are unsigned and the text can be anything, but the path is
added hop by hop by the repeaters that actually forward the packet.

The filter walks the path from the start and takes the first hop it can place
on the map as the message's origin:

- every repeater the hop hash matches (only repeaters with a known location
  count) inside the circle  -> origin inside, the message is dropped
- every match outside the circle -> origin outside, the message is kept
- no located match, or matches both inside and outside -> skip to the next hop

If no hop can be placed the message is kept. Messages that @[mention] this
radio always come through, and so does anything arriving within the reply
window after one of our own messages on that channel, so replies to us from
inside the circle are not lost.

A sender can prepend made-up hops to a path, but cannot remove the real
repeaters that forward the packet afterwards. This filter does not check
for that (yet); see the work brief for the plausibility-check idea.
"""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.models import AppSettings, Contact
from app.path_utils import path_hops
from app.repository import (
    AppSettingsRepository,
    ChannelRepository,
    ContactRepository,
    MessageRepository,
)

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (haversine), same formula as the frontend route map."""
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass(frozen=True)
class PathOrigin:
    """The first hop of a path that could be placed relative to the circle."""

    hop_index: int
    hop: str
    inside: bool
    repeater_names: tuple[str, ...]


def classify_path_origin(
    hops: Sequence[str],
    located_by_hop: dict[str, list[Contact]],
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> PathOrigin | None:
    """Find the first hop whose located candidates are all inside or all outside."""
    for index, hop in enumerate(hops):
        candidates = [c for c in located_by_hop.get(hop.lower(), []) if c.lat is not None]
        if not candidates:
            continue
        inside = [
            distance_km(center_lat, center_lon, c.lat, c.lon) <= radius_km  # type: ignore[arg-type]
            for c in candidates
        ]
        if all(inside) or not any(inside):
            return PathOrigin(
                hop_index=index,
                hop=hop,
                inside=all(inside),
                repeater_names=tuple(c.name or c.public_key[:12] for c in candidates),
            )
    return None


def geofence_active(settings: AppSettings) -> bool:
    return (
        settings.geofence_radius_km > 0
        and settings.geofence_lat is not None
        and settings.geofence_lon is not None
        and bool(settings.geofence_channels)
    )


async def _channel_selected(
    settings: AppSettings, conversation_key: str, channel_name: str | None
) -> tuple[bool, str | None]:
    if channel_name is None:
        channel = await ChannelRepository.get_by_key(conversation_key)
        channel_name = channel.name if channel else None
    key = conversation_key.upper()
    name = (channel_name or "").casefold()
    for entry in settings.geofence_channels:
        if entry.upper() == key or (name and entry.casefold() == name):
            return True, channel_name
    return False, channel_name


def _current_radio_name() -> str:
    from app.services.radio_runtime import radio_runtime

    mc = radio_runtime.meshcore
    return (mc.self_info.get("name") or "") if mc and mc.self_info else ""


async def check_channel_geofence(
    *,
    conversation_key: str,
    channel_name: str | None,
    text: str,
    path: str | None,
    path_len: int | None,
    received_at: int,
) -> PathOrigin | None:
    """Return the origin hop if this channel message should be dropped, else None."""
    settings = await AppSettingsRepository.get()
    if not geofence_active(settings) or not path:
        return None

    selected, channel_name = await _channel_selected(settings, conversation_key, channel_name)
    if not selected:
        return None

    hops = path_hops(path, path_len)
    if not hops:
        return None
    located = await ContactRepository.get_located_repeaters_by_prefixes(hops)
    origin = classify_path_origin(
        hops,
        located,
        settings.geofence_lat,  # type: ignore[arg-type]
        settings.geofence_lon,  # type: ignore[arg-type]
        settings.geofence_radius_km,
    )
    if origin is None or not origin.inside:
        return None

    radio_name = _current_radio_name()
    if radio_name and f"@[{radio_name}]".casefold() in text.casefold():
        logger.info("Geofence: kept message from inside the circle because it mentions us")
        return None

    window = settings.geofence_reply_window_minutes
    if window > 0:
        last_own = await MessageRepository.get_last_outgoing_channel_at(
            conversation_key.upper(), received_at
        )
        if last_own is not None and received_at - last_own <= window * 60:
            logger.info(
                "Geofence: kept message from inside the circle, within %d min of our own post",
                window,
            )
            return None

    return origin
