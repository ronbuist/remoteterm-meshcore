"""Tests for the channel-message geofence filter (geofence_* settings).

A channel message whose first located repeater lies inside the circle is not
stored, broadcast or fanned out, unless it mentions us or arrives within the
reply window after one of our own messages on that channel.
"""

from unittest.mock import patch

import pytest

from app.models import CONTACT_TYPE_REPEATER, ContactUpsert
from app.repository import (
    AppSettingsRepository,
    ContactRepository,
    MessageRepository,
    RawPacketRepository,
)
from app.services.geofence import classify_path_origin, distance_km

CHANNEL_KEY = "8B3387E9C5CDEA6AC9E5EDBAA115CD72"
OTHER_CHANNEL_KEY = "0123456789ABCDEF0123456789ABCDEF"
T0 = 1_700_000_000
MINUTE = 60

AMSTERDAM = (52.37, 4.90)
DIEMEN = (52.34, 4.96)  # ~5 km from the centre
HARDERWIJK = (52.35, 5.62)  # ~49 km
GRONINGEN = (53.22, 6.57)  # ~146 km

AMS_KEY = "77581f" + "11" * 29
GRN_KEY = "c25ff6" + "22" * 29
UNLOCATED_KEY = "e7ce59" + "33" * 29
AMBIG_IN_KEY = "ab" + "01" * 31
AMBIG_OUT_KEY = "ab" + "02" * 31


async def _repeater(public_key: str, name: str, loc: tuple[float, float] | None) -> None:
    await ContactRepository.upsert(
        ContactUpsert(
            public_key=public_key,
            name=name,
            type=CONTACT_TYPE_REPEATER,
            lat=loc[0] if loc else None,
            lon=loc[1] if loc else None,
        )
    )


async def _ingest(
    mock_broadcast,
    *,
    path: str,
    path_len: int,
    text: str = "buy cheap stuff",
    sender: str = "Spammer",
    timestamp: int = T0,
    received_at: int = T0,
    channel_key: str = CHANNEL_KEY,
    channel_name: str | None = "Public",
):
    from app.packet_processor import create_message_from_decrypted

    packet_id, _ = await RawPacketRepository.create(
        f"pkt-{channel_key}-{text}-{timestamp}-{path}".encode(), received_at
    )
    with patch("app.packet_processor.broadcast_event", mock_broadcast):
        msg_id = await create_message_from_decrypted(
            packet_id=packet_id,
            channel_key=channel_key,
            sender=sender,
            message_text=text,
            timestamp=timestamp,
            received_at=received_at,
            path=path,
            path_len=path_len,
            channel_name=channel_name,
        )
    return msg_id, packet_id


async def _rows(channel_key: str = CHANNEL_KEY) -> list:
    return await MessageRepository.get_all(msg_type="CHAN", conversation_key=channel_key)


class TestGeofenceFilter:
    @pytest.fixture(autouse=True)
    async def _setup(self, test_db):
        await AppSettingsRepository.update(
            geofence_lat=AMSTERDAM[0],
            geofence_lon=AMSTERDAM[1],
            geofence_radius_km=25,
            geofence_channels=["public"],
            geofence_reply_window_minutes=30,
        )
        await _repeater(AMS_KEY, "AMS-Diemen", DIEMEN)
        await _repeater(GRN_KEY, "GRN-Stad", GRONINGEN)
        await _repeater(UNLOCATED_KEY, "NoLocation", None)
        await _repeater(AMBIG_IN_KEY, "Ambig-In", DIEMEN)
        await _repeater(AMBIG_OUT_KEY, "Ambig-Out", GRONINGEN)

    @pytest.fixture(autouse=True)
    def _radio_name(self):
        with patch("app.services.geofence._current_radio_name", return_value="TenBoer"):
            yield

    @pytest.mark.asyncio
    async def test_first_hop_inside_is_dropped(self, test_db, captured_broadcasts):
        broadcasts, mock_broadcast = captured_broadcasts
        msg_id, packet_id = await _ingest(mock_broadcast, path="77581fc25ff6", path_len=2)

        assert msg_id is None
        assert await _rows() == []
        assert [b for b in broadcasts if b["type"] == "message"] == []
        packet = await RawPacketRepository.get_by_id(packet_id)
        assert packet is not None and packet[3] is None  # stays undecrypted

    @pytest.mark.asyncio
    async def test_first_hop_outside_is_kept(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        msg_id, _ = await _ingest(mock_broadcast, path="c25ff677581f", path_len=2)
        assert msg_id is not None

    @pytest.mark.asyncio
    async def test_unlocated_first_hop_is_skipped(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        msg_id, _ = await _ingest(mock_broadcast, path="e7ce5977581fc25ff6", path_len=3)
        assert msg_id is None

    @pytest.mark.asyncio
    async def test_ambiguous_hop_is_skipped(self, test_db, captured_broadcasts):
        """1-byte 'ab' matches a repeater inside and one outside; '77' decides."""
        _, mock_broadcast = captured_broadcasts
        msg_id, _ = await _ingest(mock_broadcast, path="ab77", path_len=2)
        assert msg_id is None

    @pytest.mark.asyncio
    async def test_only_ambiguous_or_unknown_hops_keep_message(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        msg_id, _ = await _ingest(mock_broadcast, path="abee", path_len=2)
        assert msg_id is not None

    @pytest.mark.asyncio
    async def test_mention_is_kept(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        msg_id, _ = await _ingest(
            mock_broadcast, path="77581f", path_len=1, text="@[tenboer] hello from Amsterdam"
        )
        assert msg_id is not None

    @pytest.mark.asyncio
    async def test_reply_window(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        await MessageRepository.create(
            msg_type="CHAN",
            text="TenBoer: test from Groningen",
            conversation_key=CHANNEL_KEY,
            sender_timestamp=T0,
            received_at=T0,
            outgoing=True,
        )
        kept, _ = await _ingest(
            mock_broadcast,
            path="77581f",
            path_len=1,
            text="heard you",
            timestamp=T0 + 10 * MINUTE,
            received_at=T0 + 10 * MINUTE,
        )
        dropped, _ = await _ingest(
            mock_broadcast,
            path="77581f",
            path_len=1,
            text="spam later",
            timestamp=T0 + 40 * MINUTE,
            received_at=T0 + 40 * MINUTE,
        )
        assert kept is not None
        assert dropped is None

    @pytest.mark.asyncio
    async def test_reply_window_zero_disables_it(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        await AppSettingsRepository.update(geofence_reply_window_minutes=0)
        await MessageRepository.create(
            msg_type="CHAN",
            text="TenBoer: test",
            conversation_key=CHANNEL_KEY,
            sender_timestamp=T0,
            received_at=T0,
            outgoing=True,
        )
        msg_id, _ = await _ingest(
            mock_broadcast, path="77581f", path_len=1, timestamp=T0 + 60, received_at=T0 + 60
        )
        assert msg_id is None

    @pytest.mark.asyncio
    async def test_unselected_channel_is_not_filtered(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        msg_id, _ = await _ingest(
            mock_broadcast,
            path="77581f",
            path_len=1,
            channel_key=OTHER_CHANNEL_KEY,
            channel_name="#groningen",
        )
        assert msg_id is not None

    @pytest.mark.asyncio
    async def test_channel_selected_by_key(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        await AppSettingsRepository.update(geofence_channels=[OTHER_CHANNEL_KEY.lower()])
        msg_id, _ = await _ingest(
            mock_broadcast,
            path="77581f",
            path_len=1,
            channel_key=OTHER_CHANNEL_KEY,
            channel_name="#whatever",
        )
        assert msg_id is None

    @pytest.mark.asyncio
    async def test_radius_zero_is_off(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        await AppSettingsRepository.update(geofence_radius_km=0)
        msg_id, _ = await _ingest(mock_broadcast, path="77581f", path_len=1)
        assert msg_id is not None

    @pytest.mark.asyncio
    async def test_echo_of_stored_message_still_adds_path(self, test_db, captured_broadcasts):
        """A copy that came in by an allowed route keeps collecting echo paths."""
        _, mock_broadcast = captured_broadcasts
        first, _ = await _ingest(mock_broadcast, path="c25ff6", path_len=1)
        echo, _ = await _ingest(mock_broadcast, path="77581fc25ff6", path_len=2)

        assert first is not None and echo is None
        rows = await _rows()
        assert len(rows) == 1
        assert {p.path for p in rows[0].paths or []} == {"c25ff6", "77581fc25ff6"}

    @pytest.mark.asyncio
    async def test_zero_hop_message_is_kept(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        msg_id, _ = await _ingest(mock_broadcast, path="", path_len=0)
        assert msg_id is not None


class TestClassifyPathOrigin:
    def test_distance(self):
        assert distance_km(*AMSTERDAM, *GRONINGEN) == pytest.approx(146, abs=3)

    def test_no_hops(self):
        assert classify_path_origin([], {}, *AMSTERDAM, 25) is None


class TestGeofenceSettings:
    @pytest.mark.asyncio
    async def test_defaults(self, test_db):
        settings = await AppSettingsRepository.get()
        assert settings.geofence_radius_km == 0
        assert settings.geofence_lat is None
        assert settings.geofence_channels == []
        assert settings.geofence_reply_window_minutes == 30

    @pytest.mark.asyncio
    async def test_router_update_and_validation(self, test_db):
        from pydantic import ValidationError

        from app.routers.settings import AppSettingsUpdate, update_settings

        result = await update_settings(
            AppSettingsUpdate(
                geofence_lat=52.37,
                geofence_lon=4.9,
                geofence_radius_km=20,
                geofence_channels=[" Public ", "public", "", "#test"],
                geofence_reply_window_minutes=15,
            )
        )
        assert result.geofence_lat == 52.37
        assert result.geofence_radius_km == 20
        assert result.geofence_channels == ["Public", "#test"]
        assert result.geofence_reply_window_minutes == 15

        for bad in (
            {"geofence_lat": 91},
            {"geofence_lon": -181},
            {"geofence_radius_km": -1},
            {"geofence_radius_km": 1001},
            {"geofence_reply_window_minutes": 1441},
        ):
            with pytest.raises(ValidationError):
                AppSettingsUpdate(**bad)
