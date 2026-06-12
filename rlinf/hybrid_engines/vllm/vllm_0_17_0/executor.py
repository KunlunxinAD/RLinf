# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import queue
import signal
import threading
import weakref
from collections import deque
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Lock as LockType
from typing import Optional

import vllm.envs as envs
from omegaconf import DictConfig
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.tracing import maybe_init_worker_tracer
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_loopback_ip,
    get_open_port,
)
from vllm.utils.system_utils import get_mp_context
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor,
    UnreadyWorkerProcHandle,
    WorkerProc,
    set_multiprocessing_worker_envs,
)
from vllm.v1.worker.worker_base import WorkerWrapperBase

from rlinf.scheduler.manager.worker_manager import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement

logger = init_logger(__name__)


class VLLMExecutor(MultiprocExecutor):
    def __init__(
        self,
        vllm_config: VllmConfig,
        rlinf_config: DictConfig,
        dp_rank: int,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
    ):
        self.rlinf_config = rlinf_config
        self.parent_address = parent_address
        self.placement = placement
        self.dp_rank = dp_rank
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        set_multiprocessing_worker_envs()

        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        if self.parallel_config.node_rank_within_dp == 0:
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            mq_connect_ip = get_ip()
            logger.info(
                "DP group leader: node_rank=%d, node_rank_within_dp=%d, "
                "master_addr=%s, mq_connect_ip=%s (local), "
                "world_size=%d, local_world_size=%d",
                self.parallel_config.node_rank,
                self.parallel_config.node_rank_within_dp,
                self.parallel_config.master_addr,
                mq_connect_ip,
                self.world_size,
                self.local_world_size,
            )
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=mq_connect_ip,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            )
            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                unready_workers.append(
                    VLLMWorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=local_rank,
                        rank=global_rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                        is_driver_worker=is_driver_worker,
                        parent_address=self.parent_address,
                        rlinf_config=self.rlinf_config,
                        placement=self.placement,
                    )
                )

            self.workers = VLLMWorkerProc.wait_for_ready(unready_workers)

            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            if self.parallel_config.node_rank_within_dp == 0:
                for rank in range(self.world_size):
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[
                            rank
                        ]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()

            self.futures_queue = deque()
            self._post_init_executor()
            success = True
        finally:
            if not success:
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()


class VLLMWorkerProc(WorkerProc):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        parent_address: WorkerAddress,
        rlinf_config: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
            "parent_address": parent_address,
            "rlinf_config": rlinf_config,
            "placement": placement,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = threading.Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )

        self._init_message_queues(input_shm_handle, vllm_config)
        is_eep_new_worker = envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH
        if not is_eep_new_worker:
            self.worker.init_device()
            self.setup_proc_title_and_log_prefix(
                enable_ep=vllm_config.parallel_config.enable_expert_parallel
            )
            self.worker.load_model()
            self.worker.use_sharded_weights()

        enable_envs_cache()

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        parent_address: WorkerAddress,
        rlinf_config: DictConfig,
        placement: ModelParallelComponentPlacement,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        reader, writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            "parent_address": parent_address,
            "rlinf_config": rlinf_config,
            "placement": placement,
        }
        proc = context.Process(
            target=VLLMWorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        proc.start()
        writer.close()
        return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)

    @staticmethod
    def worker_main(*args, **kwargs):
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        reader, ready_writer = kwargs.pop("ready_pipe")
        death_pipe: Connection | None = kwargs.pop("death_pipe", None)
        shutdown_event = threading.Event()
        if death_pipe is not None:

            def monitor_parent_death():
                try:
                    death_pipe.recv()
                except EOFError:
                    logger.info_once("Parent process exited, terminating worker")
                    shutdown_event.set()
                except Exception as e:
                    logger.warning("Death monitoring error: %s", e)

            death_monitor = threading.Thread(
                target=monitor_parent_death, daemon=True, name="WorkerDeathMonitor"
            )
            death_monitor.start()

        try:
            reader.close()

            rank = kwargs.get("rank", 0)
            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.worker",
                process_kind="worker",
                process_name=f"Worker_{rank}",
            )

            worker = VLLMWorkerProc(*args, **kwargs)
            assert worker.worker_response_mq is not None

            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                    "peer_response_handles": worker.peer_response_handles,
                }
            )

            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop(cancel=shutdown_event)

        except Exception:
            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_event.is_set():
                logger.info("WorkerProc shutting down.")
            else:
                logger.exception("WorkerProc failed.")

            shutdown_requested = True

        except SystemExit as e:
            logger.warning("WorkerProc was terminated")
            raise e

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            if worker is not None:
                worker.shutdown()
