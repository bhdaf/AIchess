"""
GPU Batch Inference Server

架构说明
--------
* ``run_inference_server()`` 是服务进程的入口函数，独占 GPU，从多个 worker
  通过共享队列收集推理请求，做动态 batch（达到 ``batch_size_max`` 或超过
  ``flush_ms`` 毫秒超时就 flush），一次 forward 完成后按 ``worker_id`` 将
  结果回传给对应的 response queue。

* ``RemoteEvaluator`` 是 worker 端的客户端，实现与 ``ChessModel`` 相同的
  ``predict_with_mask(planes, legal_indices)`` 接口，内部通过 IPC 队列与
  服务通信，可直接作为 ``model`` 参数传给 :class:`~AIchess.mcts.MCTS` 或
  :func:`~AIchess.train.self_play_game`。

* ``ModelUpdateMessage`` 用于主进程向服务进程发送模型权重更新命令。

使用示例
--------
::

    import multiprocessing as mp
    from AIchess.inference_server import RemoteEvaluator, run_inference_server

    ctx = mp.get_context('spawn')
    request_queue    = ctx.Queue(maxsize=512)
    response_queues  = [ctx.Queue(maxsize=128) for _ in range(num_workers)]
    shutdown_event   = ctx.Event()
    model_update_q   = ctx.Queue(maxsize=4)

    server = ctx.Process(
        target=run_inference_server,
        args=(model_path, request_queue, response_queues,
              shutdown_event, model_update_q),
        kwargs=dict(batch_size_max=64, flush_ms=5.0, precision='bf16'),
        daemon=True,
    )
    server.start()

    # 在 worker 进程中
    evaluator = RemoteEvaluator(worker_id, request_queue, response_queues[worker_id])
    from AIchess.train import self_play_game
    data, winner, moves, reason = self_play_game(evaluator, num_simulations=200)
"""

import time
import queue
import logging
import numpy as np
import torch
import torch.nn.functional as F

from .game import NUM_ACTIONS

logger = logging.getLogger(__name__)

# ── 公开的协议常量（主进程通过 model_update_queue 使用）────────────────────────

#: 发送到 model_update_queue / request_queue 时表示请求关闭
SHUTDOWN_MSG = "__SHUTDOWN__"

#: 发送到 model_update_queue 时表示请求推理服务重新从文件加载权重
RELOAD_MODEL_MSG = "__RELOAD_MODEL__"

# ── 内部协议常量 ───────────────────────────────────────────────────────────────

# 发送到 response_queue 时通知 worker 服务已关闭
_SERVER_GONE = "__SERVER_GONE__"

# 请求队列非阻塞轮询间隔（秒）
_POLL_INTERVAL = 0.001  # 1 ms

# 有效精度集合
_VALID_PRECISIONS = {"bf16", "fp16", "fp32"}


# ── 自定义异常 ────────────────────────────────────────────────────────────────

class InferenceServerShutdownError(RuntimeError):
    """推理服务已关闭，worker 应停止运行。"""


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _precision_to_dtype(precision: str):
    """
    将精度字符串转换为 torch autocast 的 dtype。

    Args:
        precision: 'bf16', 'fp16', 或 'fp32'

    Returns:
        对应的 torch dtype，fp32 时返回 None（不使用 autocast）

    Raises:
        ValueError: 传入无效精度字符串时
    """
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return None
    raise ValueError(
        f"无效的推理精度 '{precision}'，有效值为: {sorted(_VALID_PRECISIONS)}"
    )


def _process_batch(requests, model, response_queues, device, autocast_dtype):
    """
    对一批推理请求执行一次 GPU forward，并将结果分发回各 worker。

    Args:
        requests:        list of ``(worker_id, request_id, planes_np, legal_indices)``
        model:           已加载到 ``device`` 的 ``ChessModel``
        response_queues: per-worker response queue 列表
        device:          ``torch.device``
        autocast_dtype:  autocast dtype，或 ``None``
    """
    B = len(requests)

    # ── 组 batch ─────────────────────────────────────────────────────────────
    planes_np = np.stack([r[2] for r in requests])          # (B, 14, 10, 9)
    tensor = torch.from_numpy(planes_np).float().to(device, non_blocking=True)

    # 构造批量掩码 (B, NUM_ACTIONS)：合法走法位置为 0，其余为 -1e9
    mask = torch.full((B, NUM_ACTIONS), -1e9, device=device)
    for i, req in enumerate(requests):
        legal = req[3]
        if legal:
            mask[i, legal] = 0.0

    # ── Forward ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        if autocast_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits, values = model.model(tensor)
        else:
            logits, values = model.model(tensor)

    # ── Post-process（用 float32 保证数值稳定）────────────────────────────────
    logits = logits.float() + mask                          # (B, NUM_ACTIONS)
    policies = F.softmax(logits, dim=1).cpu().numpy()       # (B, NUM_ACTIONS)
    values_np = values.float().squeeze(-1).cpu().numpy()    # (B,)

    # ── 分发结果 ──────────────────────────────────────────────────────────────
    for i, req in enumerate(requests):
        worker_id, request_id = req[0], req[1]
        try:
            response_queues[worker_id].put(
                (request_id, policies[i], float(values_np[i])),
                timeout=5.0,
            )
        except queue.Full:
            logger.warning(
                "Worker %d 的 response queue 已满，丢弃 request_id=%d",
                worker_id, request_id,
            )


# ── 服务进程入口 ───────────────────────────────────────────────────────────────

def run_inference_server(
    model_path,
    request_queue,
    response_queues,
    shutdown_event,
    model_update_queue,
    batch_size_max=64,
    flush_ms=5.0,
    precision="fp32",
    use_tf32=False,
):
    """
    推理服务进程入口函数。

    此函数独占 GPU，循环收集 worker 发来的推理请求，做动态 batching 后一次
    完成 forward，再将结果路由回各 worker 的 response queue。

    Args:
        model_path:        模型权重文件路径（首次从此加载；更新时也从此重载）
        request_queue:     所有 worker 共用的请求队列
        response_queues:   per-worker response queue 列表（按 worker_id 索引）
        shutdown_event:    主进程设置此事件即触发优雅关闭
        model_update_queue: 主进程发送 ``RELOAD_MODEL_MSG`` 或 ``SHUTDOWN_MSG`` 消息
        batch_size_max:    最大 batch 大小（达到后立即 flush）
        flush_ms:          最长等待时间（ms），超时后强制 flush 已积累的请求
        precision:         推理精度：``'bf16'`` / ``'fp16'`` / ``'fp32'``
        use_tf32:          是否为 matmul/cudnn 启用 TF32
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [InferenceServer] %(levelname)s %(message)s",
    )

    # ── TF32 ──────────────────────────────────────────────────────────────────
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── 设备 ──────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA 不可用，推理服务将在 CPU 上运行")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    from .model import ChessModel
    model = ChessModel()
    if not model.load(model_path):
        logger.info("在 %s 未找到权重文件，构建全新模型", model_path)
        model.build()
    model.model = model.model.to(device)
    model.device = device
    model.model.eval()

    autocast_dtype = _precision_to_dtype(precision)

    logger.info(
        "推理服务就绪：device=%s  precision=%s  batch_max=%d  flush_ms=%.1f  "
        "num_workers=%d",
        device, precision, batch_size_max, flush_ms, len(response_queues),
    )

    flush_interval = flush_ms / 1000.0
    pending = []          # [(worker_id, request_id, planes_np, legal_indices), ...]
    last_flush = time.monotonic()

    # ── 统计 ──────────────────────────────────────────────────────────────────
    stat_req = 0
    stat_batches = 0
    stat_batch_total = 0
    stat_gpu_s = 0.0
    stat_window_start = time.monotonic()

    # ── 主循环 ────────────────────────────────────────────────────────────────
    while not shutdown_event.is_set():

        # 检查模型更新消息（非阻塞）
        try:
            msg = model_update_queue.get_nowait()
            if msg == SHUTDOWN_MSG:
                logger.info("收到关闭信号")
                break
            if msg == RELOAD_MODEL_MSG:
                logger.info("重新从 %s 加载模型权重", model_path)
                try:
                    model.load(model_path)
                    model.model = model.model.to(device)
                    model.device = device
                    model.model.eval()
                    logger.info("模型权重已更新")
                except Exception as exc:
                    logger.error("模型重载失败: %s", exc)
        except queue.Empty:
            pass

        # 收集推理请求
        try:
            req = request_queue.get(timeout=_POLL_INTERVAL)
            if req == SHUTDOWN_MSG:
                break
            pending.append(req)
        except queue.Empty:
            pass

        # 判断是否需要 flush
        now = time.monotonic()
        if pending and (
            len(pending) >= batch_size_max
            or (now - last_flush) >= flush_interval
        ):
            t0 = time.monotonic()
            _process_batch(pending, model, response_queues, device, autocast_dtype)
            dt = time.monotonic() - t0

            stat_batches += 1
            stat_batch_total += len(pending)
            stat_req += len(pending)
            stat_gpu_s += dt
            pending = []
            last_flush = now

        # 定期打印统计（每 10 秒）
        if now - stat_window_start >= 10.0:
            elapsed = now - stat_window_start
            if stat_batches > 0:
                avg_bs = stat_batch_total / stat_batches
                avg_gpu_ms = stat_gpu_s / stat_batches * 1000.0
                rps = stat_req / elapsed
                logger.info(
                    "统计：req/s=%.1f  avg_batch=%.1f  avg_gpu_ms=%.2f  "
                    "batches=%d  total_req=%d",
                    rps, avg_bs, avg_gpu_ms, stat_batches, stat_req,
                )
            # 重置窗口统计
            stat_req = 0
            stat_batches = 0
            stat_batch_total = 0
            stat_gpu_s = 0.0
            stat_window_start = now

    # ── 处理剩余请求 ──────────────────────────────────────────────────────────
    if pending:
        _process_batch(pending, model, response_queues, device, autocast_dtype)

    # ── 通知各 worker 服务已关闭 ─────────────────────────────────────────────
    for rq in response_queues:
        try:
            rq.put(_SERVER_GONE, timeout=1.0)
        except (queue.Full, Exception):
            pass

    logger.info("推理服务进程已退出")


# ── 客户端（worker 端）────────────────────────────────────────────────────────

class RemoteEvaluator:
    """
    Worker 端推理客户端。

    实现与 :class:`~AIchess.model.ChessModel` 相同的
    ``predict_with_mask(planes, legal_indices)`` 接口，可以直接作为
    ``model`` 参数传入 :class:`~AIchess.mcts.MCTS` 或
    :func:`~AIchess.train.self_play_game`，无需修改 MCTS/训练逻辑。

    内部通过 multiprocessing Queue 将请求发到推理服务进程，同步等待结果。

    Args:
        worker_id:      本 worker 的编号（用于服务端路由）
        request_queue:  所有 worker 共用的请求队列
        response_queue: 本 worker 专属的响应队列
    """

    def __init__(self, worker_id: int, request_queue, response_queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._counter = 0
        self._server_gone = False

    def predict_with_mask(self, planes, legal_indices):
        """
        向推理服务发送请求，阻塞等待结果。

        Args:
            planes:        numpy array ``(14, 10, 9)``，局面特征平面
            legal_indices: list[int]，合法走法在策略向量中的索引

        Returns:
            policy: numpy array ``(NUM_ACTIONS,)``，仅合法走法有概率质量
            value:  float，局面评估 [-1, 1]

        Raises:
            InferenceServerShutdownError: 推理服务已关闭
            RuntimeError: 30 秒内未收到回复或请求队列已满
        """
        if self._server_gone:
            raise InferenceServerShutdownError("推理服务已关闭")

        if not legal_indices:
            raise ValueError("predict_with_mask: no legal moves (empty legal_indices)")

        req_id = self._counter
        self._counter += 1

        # 发送请求
        try:
            self.request_queue.put(
                (self.worker_id, req_id, planes, list(legal_indices)),
                timeout=30.0,
            )
        except queue.Full:
            raise RuntimeError("推理请求队列已满（服务负载过高）")

        # 等待匹配的响应
        while True:
            try:
                resp = self.response_queue.get(timeout=30.0)
            except queue.Empty:
                raise RuntimeError(
                    f"等待推理结果超时（request_id={req_id}，worker={self.worker_id}）"
                )

            if resp == _SERVER_GONE:
                self._server_gone = True
                raise InferenceServerShutdownError("推理服务已关闭")

            resp_id, policy, value = resp
            if resp_id == req_id:
                return policy, value

            # 收到了不匹配的响应（极罕见；一般不会发生于阻塞单步模式）
            logger.debug(
                "Worker %d: 收到无序响应 resp_id=%d，期望 %d，忽略",
                self.worker_id, resp_id, req_id,
            )
