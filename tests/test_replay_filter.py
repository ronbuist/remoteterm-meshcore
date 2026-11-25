"""Tests for the channel replay filter (replay_filter_hours).

A replay is a channel message with the same channel and exact text as one
received recently, but a different sender timestamp. With the filter on it is
not stored, broadcast or fanned out; its raw packet is linked to the original.
"""

from unittest.mock import patch

import pytest

from app.repository import AppSettingsRepository, MessageRepository, RawPacketRepository

CHANNEL_KEY = "ABC123DEF456ABC123DEF456ABC12345"
OTHER_CHANNEL_KEY = "0123456789ABCDEF0123456789ABCDEF"
T0 = 1_700_000_000
HOUR = 3600


async def _ingest(
    mock_broadcast,
    *,
    text: str = "Hello mesh",
    sender: str = "Alice",
    timestamp: int,
    received_at: int,
    channel_key: str = CHANNEL_KEY,
    path: str | None = "aabb",
):
    from app.packet_processor import create_message_from_decrypted

    packet_id, _ = await RawPacketRepository.create(
        f"pkt-{channel_key}-{text}-{timestamp}-{received_at}-{path}".encode(), received_at
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
        )
    return msg_id, packet_id


async def _channel_rows(channel_key: str = CHANNEL_KEY) -> list:
    return await MessageRepository.get_all(msg_type="CHAN", conversation_key=channel_key)


def _new_message_broadcasts(broadcasts) -> list:
    return [b for b in broadcasts if b["type"] == "message"]


class TestReplayFilter:
    @pytest.fixture(autouse=True)
    async def _enable(self, test_db):
        await AppSettingsRepository.update(replay_filter_hours=24)

    @pytest.mark.asyncio
    async def test_replay_with_new_timestamp_is_dropped(self, test_db, captured_broadcasts):
        broadcasts, mock_broadcast = captured_broadcasts
        original_id, _ = await _ingest(mock_broadcast, timestamp=T0, received_at=T0)
        assert original_id is not None

        replay_id, replay_packet = await _ingest(
            mock_broadcast, timestamp=T0 + 5 * HOUR, received_at=T0 + 5 * HOUR
        )

        assert replay_id is None
        assert len(await _channel_rows()) == 1
        assert len(_new_message_broadcasts(broadcasts)) == 1
        packet = await RawPacketRepository.get_by_id(replay_packet)
        assert packet is not None and packet[3] == original_id

    @pytest.mark.asyncio
    async def test_replay_under_another_sender_name_is_stored(self, test_db, captured_broadcasts):
        """The text includes the sender name, so a renamed replay is a new message."""
        _, mock_broadcast = captured_broadcasts
        await _ingest(mock_broadcast, sender="Alice", timestamp=T0, received_at=T0)
        msg_id, _ = await _ingest(
            mock_broadcast, sender="Mallory", timestamp=T0 + 60, received_at=T0 + 60
        )
        assert msg_id is not None
        assert len(await _channel_rows()) == 2

    @pytest.mark.asyncio
    async def test_same_text_after_window_is_stored(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        await _ingest(mock_broadcast, timestamp=T0, received_at=T0)
        msg_id, _ = await _ingest(
            mock_broadcast, timestamp=T0 + 25 * HOUR, received_at=T0 + 25 * HOUR
        )
        assert msg_id is not None
        assert len(await _channel_rows()) == 2

    @pytest.mark.asyncio
    async def test_echo_with_same_timestamp_still_adds_path(self, test_db, captured_broadcasts):
        broadcasts, mock_broadcast = captured_broadcasts
        original_id, _ = await _ingest(mock_broadcast, timestamp=T0, received_at=T0, path="aabb")
        echo_id, _ = await _ingest(mock_broadcast, timestamp=T0, received_at=T0 + 2, path="ccdd")

        assert echo_id is None
        rows = await _channel_rows()
        assert len(rows) == 1
        assert {p.path for p in rows[0].paths or []} == {"aabb", "ccdd"}
        assert original_id == rows[0].id

    @pytest.mark.asyncio
    async def test_other_channel_is_not_affected(self, test_db, captured_broadcasts):
        _, mock_broadcast = captured_broadcasts
        await _ingest(mock_broadcast, timestamp=T0, received_at=T0)
        msg_id, _ = await _ingest(
            mock_broadcast,
            timestamp=T0 + 60,
            received_at=T0 + 60,
            channel_key=OTHER_CHANNEL_KEY,
        )
        assert msg_id is not None

    @pytest.mark.asyncio
    async def test_replay_of_our_own_outgoing_message_is_dropped(
        self, test_db, captured_broadcasts
    ):
        _, mock_broadcast = captured_broadcasts
        await MessageRepository.create(
            msg_type="CHAN",
            text="Ron: Hello mesh",
            conversation_key=CHANNEL_KEY,
            sender_timestamp=T0,
            received_at=T0,
            outgoing=True,
        )
        msg_id, _ = await _ingest(
            mock_broadcast, sender="Ron", timestamp=T0 + 600, received_at=T0 + 600
        )
        assert msg_id is None
        assert len(await _channel_rows()) == 1

    @pytest.mark.asyncio
    async def test_genuine_repeat_by_someone_else_is_dropped_accepted_trade_off(
        self, test_db, captured_broadcasts
    ):
        """Same sender name + same short text within the window is indistinguishable."""
        _, mock_broadcast = captured_broadcasts
        await _ingest(mock_broadcast, text="ok", timestamp=T0, received_at=T0)
        msg_id, _ = await _ingest(
            mock_broadcast, text="ok", timestamp=T0 + 300, received_at=T0 + 300
        )
        assert msg_id is None

    @pytest.mark.asyncio
    async def test_historical_decrypt_out_of_order_keeps_first_seen(
        self, test_db, captured_broadcasts
    ):
        """The window applies either side of received_at, so order doesn't matter."""
        _, mock_broadcast = captured_broadcasts
        await _ingest(mock_broadcast, timestamp=T0 + HOUR, received_at=T0 + HOUR)
        msg_id, _ = await _ingest(mock_broadcast, timestamp=T0, received_at=T0)
        assert msg_id is None
        assert len(await _channel_rows()) == 1

    @pytest.mark.asyncio
    async def test_fallback_path_drops_replay(self, test_db, captured_broadcasts):
        from app.services.messages import create_fallback_channel_message

        broadcasts, mock_broadcast = captured_broadcasts
        await _ingest(mock_broadcast, timestamp=T0, received_at=T0)
        result = await create_fallback_channel_message(
            conversation_key=CHANNEL_KEY,
            message_text="Hello mesh",
            sender_timestamp=T0 + 900,
            received_at=T0 + 900,
            path=None,
            path_len=None,
            txt_type=0,
            sender_name="Alice",
            channel_name="#test",
            broadcast_fn=mock_broadcast,
        )
        assert result is None
        assert len(await _channel_rows()) == 1


class TestReplayFilterOff:
    @pytest.mark.asyncio
    async def test_default_off_stores_replays(self, test_db, captured_broadcasts):
        assert (await AppSettingsRepository.get()).replay_filter_hours == 0
        _, mock_broadcast = captured_broadcasts
        await _ingest(mock_broadcast, timestamp=T0, received_at=T0)
        msg_id, _ = await _ingest(mock_broadcast, timestamp=T0 + 60, received_at=T0 + 60)
        assert msg_id is not None
        assert len(await _channel_rows()) == 2


class TestReplayFilterSetting:
    @pytest.mark.asyncio
    async def test_router_saves_and_validates(self, test_db):
        from pydantic import ValidationError

        from app.routers.settings import AppSettingsUpdate, update_settings

        result = await update_settings(AppSettingsUpdate(replay_filter_hours=24))
        assert result.replay_filter_hours == 24
        result = await update_settings(AppSettingsUpdate(replay_filter_hours=0))
        assert result.replay_filter_hours == 0
        with pytest.raises(ValidationError):
            AppSettingsUpdate(replay_filter_hours=169)
        with pytest.raises(ValidationError):
            AppSettingsUpdate(replay_filter_hours=-1)
