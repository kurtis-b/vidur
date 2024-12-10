from typing import List

from vidur.entities import Batch
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent
        from vidur.events.global_schedule_decode_event import GlobalScheduleDecodeEvent

        self._batch.on_batch_end(self.time)
        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        prefill_complete_requests = replica_scheduler.on_batch_end(self._batch)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )

        with open("events.txt", "a") as f:
            f.write(
                f"BatchEndEvent ({self._id}): time={self.time}, replica_id={self._replica_id}, "
                f"batch_id={self._batch.id}, memory_usage_percent={memory_usage_percent}"
            )
            f.write("\n")
        
        if prefill_complete_requests:
            for request in prefill_complete_requests:
                scheduler.add_request(request)
            if not scheduler.is_empty():
                return [GlobalScheduleDecodeEvent(self.time), ReplicaScheduleEvent(self.time, self._replica_id)]
        return [ReplicaScheduleEvent(self.time, self._replica_id)]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }
