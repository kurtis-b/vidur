from abc import ABC, abstractmethod
from typing import List

from vidur.config import SimulationConfig
from vidur.entities import Batch, Replica, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.logger import init_logger
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler

from math import ceil


logger = init_logger(__name__)


class SplitwiseReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests: List[Request] = []
        self._num_running_batches = 0
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )
        self._is_prefill = None

    @property
    def is_prefill(self) -> bool:
        return self._is_prefill

    def set_prefill(self, is_prefill: bool) -> None:
        self._is_prefill = is_prefill
        for stage_scheduler in self._replica_stage_schedulers.values():
            stage_scheduler.set_prefill(is_prefill)

    def on_batch_end(self, batch: Batch) -> List[Request]:
        self._num_running_batches -= 1
        prefill_complete_requests = []
        unfinished_requests = []

        for request in batch.requests:
            if self.is_prefill:
                prefill_complete_requests = [r for r in batch.requests if r.is_prefill_complete]
                if request.is_prefill_complete:
                    self.free(request.id)
            else:
                if request.completed:
                    self.free(request.id)
                else:
                    unfinished_requests.append(request)
        return prefill_complete_requests

    def _can_allocate_request(self, request: Request) -> bool:
        # if request.id not in self._allocation_map:
        if self.is_prefill:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        # vllm requires at least one block to be available
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        # if request.id not in self._allocation_map:
        if self.is_prefill:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        # num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        # num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)
        # assert (
        #     num_tokens_required == 0 or num_tokens_required == 1
        # ), f"num_tokens_required: {num_tokens_required}"

        # if num_tokens_required == 0:
        #     return

        self.allocate(request.id, 1)

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        num_batch_tokens = 0
        unfinished_decode_requests = []

        if not self.is_prefill:
            for request_id in self._allocation_map.keys():
                if any(request._id == request_id for request in self._request_queue):
                    index = next(i for i, request in enumerate(self._request_queue) if request._id == request_id)
                    request = self._request_queue.pop(index)
                    if request.total_tokens - request.num_processed_tokens > 1: 
                        unfinished_decode_requests.append(request)
                    requests.append(request)
                    next_num_tokens = self._get_request_next_num_tokens(request)
                    num_tokens.append(next_num_tokens)
                    num_batch_tokens += next_num_tokens

        while self._request_queue:
            request = self._request_queue[0]

            next_num_tokens = self._get_request_next_num_tokens(request)

            if not self._can_allocate_request(request):
                break

            new_num_tokens = num_tokens + [next_num_tokens]
            new_num_batch_tokens = len(new_num_tokens) * max(new_num_tokens)
            if new_num_batch_tokens > self._config.max_tokens_in_batch:
                break

            if len(self._allocation_map) == self._config.batch_size_cap:
                break

            if len(requests) == self._max_micro_batch_size:
                break

            request = self._request_queue.pop(0)
            # If this scheduler instance is decode, then check if the request will have 
            # generated all decode tokens after processing this request. If not, then add 
            # it back to the request queue after mapping.
            if not self.is_prefill:
                if request.total_tokens - request.num_processed_tokens > 1: 
                    unfinished_decode_requests.append(request)

            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            num_batch_tokens += next_num_tokens
        self._request_queue = unfinished_decode_requests + self._request_queue

        if requests:
            # if not self.is_prefill:
            #     print("decode return in middle, len(requests): ", len(requests), "num_batch_tokens: ", len(num_tokens) * max(num_tokens))
            return Batch(self._replica_id, requests, num_tokens)

        # Safer to sort preempted_requests to maintain FIFO order
        self._preempted_requests.sort(key=lambda r: r.arrived_at)
        # all preempted_requests will have prefill completed
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.pop(0)

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue = [victim_request] + self._request_queue
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)
                next_num_tokens = self._get_request_next_num_tokens(request)
                requests.append(request)
                num_tokens.append(next_num_tokens)

        if not requests:
            return

        # if not self.is_prefill:
        #     print("decode return at end, len(requests): ", len(requests), "num_batch_tokens: ", len(num_tokens) * max(num_tokens))
        return Batch(self._replica_id, requests, num_tokens)
