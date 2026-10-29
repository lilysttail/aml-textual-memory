"""核心的配置：调用方构造好传进来，核心自己不读环境变量。

每个开关都只说「检索怎么做」，不说「数据从哪来」：隔离维叫 space_id、
幂等键叫 batch_id，平台的名字由壳做映射。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple


@dataclass
class CoreConfig:
    # --- 存哪 ---
    db_path: Optional[str] = None
    db: Optional[Any] = None              # 注入已有的 Database（壳共用写锁）
    busy_timeout_ms: int = 5000
    write_lock_timeout_ms: int = 10000

    # --- 检索怎么做 ---
    paths: Tuple[str, ...] = ("keyword",)  # 开哪几路（keyword / dense）
    rerank_enabled: bool = False
    rerank_n: int = 200
    top_k: int = 10
    context_before: int = 0                # 邻居条数（比赛各 1；别的使用方可以多取几条）
    context_after: int = 0
    group_by_session: bool = True          # 同会话内按 ts 升序
    dense_exclude_lines: Tuple[str, ...] = ("terminal",)
                                             # 这些 line 只进关键词索引，不进向量

    # --- 插头 ---
    tokenizer: Optional[Callable[[str], list]] = None
                                             # 切词器插头（返回词列表）；不给就用内置的
                                             # 英文切词（中文切词是 2b 的事）
    tokenizer_key: Optional[str] = None
                                             # 切词器的版本标识（写进 core_write_settings）：
                                             # 换切词器 = 换 key，旧索引对不上就拒启动
    stem_english: bool = False               # 英文内置切词的词干还原（单复数/时态）开关；
                                             # 只影响 builtin 英文分词，自定义 tokenizer（如
                                             # 中文 2b）不经过它。改它 = 改索引格式，旧库拒启动
                                             # 需重建索引（见 core.engine.rebuild_fts_index）
    embedder: Optional[Any] = None         # embed(texts) -> vectors
    vector_cache: Optional[Any] = None     # 注入壳的向量缓存（共用一套缓存）

    # --- 向量缓存自身的闸（核心自带缓存时用） ---
    embed_max_concurrency: int = 2
    embed_batch: int = 32
    embed_backfill_wait_s: float = 60.0
    embed_acquire_s: float = 30.0

    # --- 搜索路径的限量补算---
    dense_backfill_m: int = 8      # 每次搜索最多补几条
    dense_backfill_t: float = 2.0  # 补算的总时间上限（秒），从拿到 query 向量之后算
    dense_query_t: float = 3.0     # query 向量自己的期限（秒），从要它那一刻算、等闸位也在里面

    # --- 内存里的空间索引（core/dense_index.py） ---
    search_heavy_concurrency: int = 0         # 建索引 / 算相似度同时最多几个；0 = 本进程能用的 CPU 数
    search_heavy_wait_s: float = 15.0         # 等不到名额多久就放弃（抛 SearchBusy，服务回 503）
    search_cache_bytes: int = 640 * 2 ** 20   # 共享向量池 + 各空间索引的字节上限
