import logging

import aiosqlite

logger = logging.getLogger(__name__)

_COLUMNS = {
    "geofence_lat": "REAL DEFAULT NULL",
    "geofence_lon": "REAL DEFAULT NULL",
    "geofence_radius_km": "REAL DEFAULT 0",
    "geofence_channels": "TEXT DEFAULT '[]'",
    "geofence_reply_window_minutes": "INTEGER DEFAULT 30",
}


async def migrate(conn: aiosqlite.Connection) -> None:
    """Add the geofence filter settings for channel messages (radius 0 = off)."""
    tables_cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    if "app_settings" not in {row[0] for row in await tables_cursor.fetchall()}:
        await conn.commit()
        return
    col_cursor = await conn.execute("PRAGMA table_info(app_settings)")
    columns = {row[1] for row in await col_cursor.fetchall()}
    for name, definition in _COLUMNS.items():
        if name not in columns:
            await conn.execute(f"ALTER TABLE app_settings ADD COLUMN {name} {definition}")
    await conn.commit()
