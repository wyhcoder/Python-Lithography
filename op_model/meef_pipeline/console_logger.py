"""控制台输出双写工具 (tee).

提供 _TeeStream + tee_console_to_file 上下文管理器, 把 sys.stdout/stderr
的内容同时写到屏幕和日志文件. 与 op_model/demo_sraf.py 内部的同名实现
逻辑一致, 这里抽出来作为 meef_pipeline 的公共工具, 避免反向依赖.

用法:
    from op_model.meef_pipeline.console_logger import tee_console_to_file
    with tee_console_to_file(Path(filepath) / "run.log"):
        print("xxx")           # 屏幕 + 日志文件 各一份
        # 任何子函数里的 print / 异常 traceback 也都会被 tee 到文件
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Union


class _TeeStream:
    """同时写到多个底层 stream 的轻量封装 (用于把控制台输出复制到文件).

    与直接 reassign sys.stdout 相比, 用 _TeeStream 的好处:
      - 屏幕仍正常显示, 不影响交互体验;
      - 同时把每一行 print 的内容追加到日志文件;
      - 文件内容用 flush() 立即落盘, 避免长任务异常退出后丢日志.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                # 单个 stream 失败不应影响其它 (例如文件系统满)
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    # 让 isatty 等查询沿用第一个 stream (通常是真实 terminal),
    # 避免某些库把 tee 误识别成非交互式后改变行为.
    def isatty(self) -> bool:
        first = self._streams[0] if self._streams else None
        return bool(first and getattr(first, "isatty", lambda: False)())

    def fileno(self) -> int:
        # 部分库 (如 matplotlib) 可能调用 fileno; 用第一个真实 stream 的
        first = self._streams[0]
        return first.fileno()


@contextmanager
def tee_console_to_file(log_path: Union[str, Path], header: str = "MEEF Pipeline Run Log"):
    """上下文管理器: 进入时把 sys.stdout/sys.stderr 复制一份写到 log_path.

    Args:
        log_path: 日志文件路径 (相对/绝对均可). 父目录会自动创建.
        header:   写入文件首行的标题, 便于区分不同任务.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 'w' 覆盖写, 一次 run 一份日志 (如需追加改 'a' 即可).
    f = open(log_path, "w", encoding="utf-8", buffering=1)  # 行缓冲
    f.write(f"# {header}\n")
    f.write(f"# Started at: {datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"# Working dir: {Path.cwd()}\n")
    f.write("# " + "-" * 60 + "\n")
    f.flush()

    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(real_stdout, f)
    sys.stderr = _TeeStream(real_stderr, f)
    try:
        yield log_path
    finally:
        # 复位 + 收尾
        sys.stdout = real_stdout
        sys.stderr = real_stderr
        f.write("# " + "-" * 60 + "\n")
        f.write(f"# Finished at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.flush()
        f.close()
