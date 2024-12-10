from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class ReplicaStageScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, stage_id: int):
        super().__init__(time, EventType.REPLICA_STAGE_SCHEDULE)

        self._replica_id = replica_id
        self._stage_id = stage_id

        self._batch = None
        self._batch_stage = None
        self._is_last_stage = None

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_end_event import BatchStageEndEvent

        stage_scheduler = scheduler.get_replica_stage_scheduler(
            self._replica_id, self._stage_id
        )

        self._batch, self._batch_stage, execution_time = stage_scheduler.on_schedule()

        with open("events.txt", "a") as f:
            f.write(
                    f"ReplicaStageScheduleEvent ({self._id}): time={self.time}, "
                    f"replica_id={self._replica_id}, stage_id={self._stage_id}, "
                    f"batch_id={self._batch.id if self._batch else 'None'}, "
                    f"batch_stage_id={self._batch_stage.id if self._batch_stage else 'None'}, "
                    f"is_last_stage={stage_scheduler.is_last_stage}"
                )
        if not (self._batch and self._batch_stage):
            with open("events.txt", "a") as f:
                f.write("\n")
            return []

        self._batch_stage.on_schedule(self.time)
        metrics_store.on_replica_stage_schedule(
            self.time,
            self._replica_id,
            self._stage_id,
            self._batch_stage,
            execution_time,
        )

        self._is_last_stage = stage_scheduler.is_last_stage

        with open("events.txt", "a") as f:
            f.write(
                    f", model_time_ms={execution_time.model_time_ms}, total_time_s={execution_time.total_time}"
                )
            f.write("\n")
        return [
            BatchStageEndEvent(
                self.time + self._batch_stage.execution_time,
                self._replica_id,
                self._stage_id,
                self._is_last_stage,
                self._batch,
                self._batch_stage,
            ),
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id if self._batch else None,
            "batch_stage_id": self._batch_stage.id if self._batch_stage else None,
            "is_last_stage": self._is_last_stage,
        }
