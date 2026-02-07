from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo import MongoClient


class MongoRepository:
    def __init__(self, mongo_uri: str, db_name: str) -> None:
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.drivers = self.db.drivers
        self.events = self.db.distraction_events
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.drivers.create_index("driver_id", unique=True)
        self.events.create_index([("driver_id", 1), ("timestamp", -1)])

    def upsert_driver(self, driver_doc: Dict[str, Any]) -> None:
        self.drivers.update_one(
            {"driver_id": driver_doc["driver_id"]},
            {"$set": driver_doc},
            upsert=True,
        )

    def get_all_drivers(self) -> list[Dict[str, Any]]:
        return list(self.drivers.find({}, {"_id": 0}))

    def get_driver(self, driver_id: str) -> Optional[Dict[str, Any]]:
        return self.drivers.find_one({"driver_id": driver_id}, {"_id": 0})

    def update_thresholds(self, driver_id: str, ear_threshold: float, mar_threshold: float) -> None:
        self.drivers.update_one(
            {"driver_id": driver_id},
            {
                "$set": {
                    "thresholds.ear": ear_threshold,
                    "thresholds.mar": mar_threshold,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    def log_event(self, event: Dict[str, Any]) -> None:
        event["timestamp"] = datetime.now(timezone.utc)
        self.events.insert_one(event)
