from typing import List, Tuple, Dict

from vidur.config import SimulationConfig
from vidur.entities import Replica, Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.replica_scheduler_type import ReplicaSchedulerType

from math import ceil


class LORSplitwiseeGlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler for Splitwise.
    """
    def __init__(self, config, replicas, is_prefill=None):
        super().__init__(config, replicas)
        self._is_prefill = is_prefill

    @property
    def is_prefill(self):
        return self._is_prefill

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        # keep a map of replica_id -> replica_scheduler
        # this is used to find the replica with the least outstanding requests
        pending_requests_map = {
            replica_scheduler.replica_id: replica_scheduler.num_pending_requests
            for replica_scheduler in self._replica_schedulers.values()
        }

        # using a very simple implementation here, to keep wiring simple
        unfinished_decode_request = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
            pending_requests_map[replica_id] += 1
            request_mapping.append((replica_id, request))
        self._request_queue = unfinished_decode_request + self._request_queue

        return request_mapping
    

class SplitwiseGlobalScheduler(BaseGlobalScheduler):
    """
    Splitwise global scheduler.
    """
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        """
        The Splitwise global scheduler will generate two sub-schedulers based on the other global scheduler types. 
        One sub-scheduler will be used for prefill phase and the other sub-scheduler will be used for decode phase.
        """
        self._config = config
        self._replicas = replicas
        self._num_replicas = len(self._replicas)

        self._pd_node_ratio = self._config.cluster_config.replica_scheduler_config.pd_node_ratio
        assert self._config.cluster_config.replica_scheduler_config.get_type() == ReplicaSchedulerType.SPLITWISE

        self._num_prefill_nodes = ceil(self._num_replicas * self._pd_node_ratio)
        self._num_decode_nodes = self._num_replicas - self._num_prefill_nodes

        assert self._num_prefill_nodes > 0
        assert self._num_decode_nodes > 0

        self._prefill_replicas = {
            replica_id: replica
            for replica_id, replica in self._replicas.items()
            if replica_id < self._num_prefill_nodes
        }

        self._decode_replicas = {
            replica_id: replica
            for replica_id, replica in self._replicas.items()
            if replica_id >= self._num_prefill_nodes
        }

        self._prefill_scheduler = self.get_global_scheduler(self._prefill_replicas, is_prefill=True)
        self._decode_scheduler = self.get_global_scheduler(self._decode_replicas, is_prefill=False)

        for replica_schedulers in self._prefill_scheduler._replica_schedulers.values():
            replica_schedulers.set_prefill(True)
        for replica_schedulers in self._decode_scheduler._replica_schedulers.values():
            replica_schedulers.set_prefill(False)

    def get_global_scheduler(self, replicas: Dict[int, Replica], is_prefill=None):
        return LORSplitwiseeGlobalScheduler(self._config, replicas)

    def get_replica_scheduler(self, replica_id):
        if replica_id < self._num_prefill_nodes:
            return self._prefill_scheduler.get_replica_scheduler(replica_id)
        else:
            return self._decode_scheduler.get_replica_scheduler(replica_id)

    def get_replica_stage_scheduler(self, replica_id, stage_id):
        if replica_id < self._num_prefill_nodes:
            return self._prefill_scheduler.get_replica_stage_scheduler(replica_id, stage_id)
        else:
            return self._decode_scheduler.get_replica_stage_scheduler(replica_id, stage_id)

    def add_request(self, request: Request) -> None:
        if not request._is_prefill_complete:
            self._prefill_scheduler.add_request(request)
        else:
            self._decode_scheduler.add_request(request)

    def is_empty(self) -> bool:
        return self._prefill_scheduler.is_empty() and self._decode_scheduler.is_empty()

    def schedule(self) -> List[Tuple[int, Request]]:
        return self._prefill_scheduler.schedule()
    
    def schedule_decode(self) -> List[Tuple[int, Request]]:
        return self._decode_scheduler.schedule()
