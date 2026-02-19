"""
MnemosyneManual - Mnemosyne 手动记忆注入伴生插件
直接向 Mnemosyne 的 Milvus 向量数据库中插入手动编写的记忆条目。
插入的记录与 Mnemosyne 自动生成的记录格式完全一致，
使 Mnemosyne 在检索时能正常 fetch 到手动插入的记忆。

核心策略：不自建 Milvus 连接，直接借用 Mnemosyne 已建立的连接。
两个插件运行在同一个 AstrBot 进程中，共享 pymilvus 连接池。
"""

import asyncio
import time
from typing import Any, cast

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.provider.provider import EmbeddingProvider

from pymilvus import (
    Collection,
    MilvusException,
    connections,
    utility,
)

# ---------------------------------------------------------------------------
# Mnemosyne 兼容常量（必须与 Mnemosyne core/constants.py 完全一致）
# ---------------------------------------------------------------------------
VECTOR_FIELD_NAME = "embedding"
PRIMARY_FIELD_NAME = "memory_id"
DEFAULT_PERSONA_ON_NONE = "UNKNOWN_PERSONA"

# Mnemosyne 的 MilvusManager 默认使用 "default" 作为连接别名
MNEMOSYNE_CONNECTION_ALIAS = "default"


@register(
    "MnemosyneManual",
    "FelisAbyssalis",
    "Mnemosyne 手动记忆注入插件 - 向 Mnemosyne 的向量数据库手动插入记忆条目",
    "1.0.0",
    "",
)
class MnemosyneManual(Star):
    """
    AstrBot 伴生插件：为 Mnemosyne 提供手动记忆插入能力。

    核心设计：不自建 Milvus 连接，直接借用同进程中 Mnemosyne 插件
    已经建立的 pymilvus 连接（别名 "default"）。这样可以：
    1. 避免 Milvus Lite 多连接隔离问题
    2. 保证读写同一个数据库实例
    3. 简化配置（不需要再配置数据库路径）
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.context = context

        # --- 组件状态 ---
        self.collection_name: str = "default"
        self.embedding_provider: EmbeddingProvider | None = None
        self._embedding_provider_ready = False

        logger.info("开始初始化 MnemosyneManual 插件...")

        # 读取集合名称
        self.collection_name = self.config.get("collection_name", "default")
        logger.info(f"MnemosyneManual 将写入集合: '{self.collection_name}'")
        logger.info("MnemosyneManual 插件基础初始化完成（将在 AstrBot 加载后连接 Milvus）")

    # -----------------------------------------------------------------------
    # Milvus 连接（借用 Mnemosyne 的）
    # -----------------------------------------------------------------------

    def _check_mnemosyne_connection(self) -> bool:
        """
        检查 Mnemosyne 的 pymilvus 连接是否可用。
        Mnemosyne 使用别名 "default" 建立连接。
        """
        try:
            # 检查 pymilvus 连接池中是否存在 Mnemosyne 的连接
            existing = connections.list_connections()
            for alias, _ in existing:
                if alias == MNEMOSYNE_CONNECTION_ALIAS:
                    # 验证连接是否真的活跃
                    try:
                        # 尝试列出集合来验证连接可用性
                        utility.list_collections(using=MNEMOSYNE_CONNECTION_ALIAS)
                        return True
                    except Exception:
                        logger.warning(
                            f"连接别名 '{MNEMOSYNE_CONNECTION_ALIAS}' 存在但不可用"
                        )
                        return False

            logger.warning(
                "未找到 Mnemosyne 的 Milvus 连接。"
                "请确保 Mnemosyne 插件已安装并启用，且 Milvus 已初始化"
            )
            return False

        except Exception as e:
            logger.error(f"检查 Mnemosyne 连接时出错: {e}")
            return False

    # -----------------------------------------------------------------------
    # Embedding Provider
    # -----------------------------------------------------------------------

    def _get_embedding_provider(self) -> EmbeddingProvider | None:
        """
        获取 Embedding Provider，策略与 Mnemosyne 一致：
        1. 从配置指定的 Provider ID 获取
        2. 使用框架默认的第一个 Embedding Provider
        """
        try:
            # 优先级 1: 配置指定的 Provider ID
            emb_id = self.config.get("embedding_provider_id", "")
            if emb_id:
                provider = None
                try:
                    provider_manager = getattr(self.context, "provider_manager", None)
                    inst_map = getattr(provider_manager, "inst_map", None)
                    if isinstance(inst_map, dict):
                        provider = inst_map.get(emb_id)
                except (AttributeError, TypeError):
                    pass

                if provider is None:
                    try:
                        provider = self.context.get_provider_by_id(emb_id)
                    except Exception:
                        pass

                if provider and (
                    callable(getattr(provider, "embed_texts", None))
                    or callable(getattr(provider, "get_embedding", None))
                ):
                    logger.info(
                        f"MnemosyneManual 成功加载 Embedding Provider: {emb_id}"
                    )
                    return cast(EmbeddingProvider, provider)

            # 优先级 2: 框架默认
            try:
                embedding_providers = self.context.get_all_embedding_providers()
                if embedding_providers and len(embedding_providers) > 0:
                    provider = embedding_providers[0]
                    provider_id = getattr(provider, "provider_config", {}).get(
                        "id", "unknown"
                    )
                    logger.info(
                        f"MnemosyneManual 使用默认 Embedding Provider: {provider_id}"
                    )
                    return cast(EmbeddingProvider, provider)
            except Exception as e:
                logger.debug(f"获取默认 Embedding Provider 失败: {e}")

            logger.warning("MnemosyneManual: 没有可用的 Embedding Provider")
            return None

        except Exception as e:
            logger.error(f"MnemosyneManual 获取 Embedding Provider 失败: {e}")
            return None

    def _ensure_embedding_provider(self) -> EmbeddingProvider | None:
        """确保 Embedding Provider 可用。"""
        if not self._embedding_provider_ready or not self.embedding_provider:
            self.embedding_provider = self._get_embedding_provider()
            if self.embedding_provider:
                self._embedding_provider_ready = True
        return self.embedding_provider

    # -----------------------------------------------------------------------
    # 核心功能: insert_memory
    # -----------------------------------------------------------------------

    async def insert_memory(
        self,
        text: str,
        session_id: str,
        persona_id: str | None = None,
    ) -> dict[str, Any]:
        """
        向 Mnemosyne 的 Milvus 集合中插入一条手动记忆。

        数据格式与 Mnemosyne 的 _store_summary_to_milvus() 完全一致。

        Args:
            text: 记忆文本内容
            session_id: 会话 ID（来自 event.unified_msg_origin）
            persona_id: 人格 ID（可选，默认使用配置值）

        Returns:
            包含 success, message, memory_id 字段的结果字典
        """
        # --- 前置检查 ---
        if not text or not text.strip():
            return {"success": False, "message": "记忆文本不能为空"}

        text = text.strip()
        if len(text) > 4096:
            return {
                "success": False,
                "message": f"记忆文本过长 ({len(text)} 字符)，最大 4096 字符",
            }

        # 检查 Mnemosyne 的连接是否可用
        if not self._check_mnemosyne_connection():
            return {
                "success": False,
                "message": (
                    "无法连接到 Milvus 数据库。"
                    "请确保 Mnemosyne 插件已启用并且 Milvus 已初始化 (/memory init)"
                ),
            }

        # 确保集合存在
        try:
            has_collection = utility.has_collection(
                self.collection_name, using=MNEMOSYNE_CONNECTION_ALIAS
            )
            if not has_collection:
                # 列出所有可用集合帮助调试
                all_collections = utility.list_collections(
                    using=MNEMOSYNE_CONNECTION_ALIAS
                )
                return {
                    "success": False,
                    "message": (
                        f"集合 '{self.collection_name}' 不存在。"
                        f"当前可用集合: {all_collections}。"
                        "请先在 Mnemosyne 中执行 /memory init 创建集合，"
                        "或检查集合名称配置是否与 Mnemosyne 一致"
                    ),
                }
        except Exception as e:
            return {"success": False, "message": f"检查集合状态失败: {e}"}

        # 确保 Embedding Provider 可用
        provider = self._ensure_embedding_provider()
        if not provider:
            return {
                "success": False,
                "message": (
                    "Embedding Provider 不可用，无法向量化记忆文本。"
                    "请在 AstrBot 中配置 Embedding Provider"
                ),
            }

        # --- 获取 Embedding ---
        try:
            embedding_vector = await provider.get_embedding(text)
            if not embedding_vector:
                return {
                    "success": False,
                    "message": "获取文本 Embedding 失败（返回空向量）",
                }
        except Exception as e:
            return {"success": False, "message": f"获取文本 Embedding 失败: {e}"}

        # --- 构建插入数据（与 Mnemosyne 完全一致的格式）---
        effective_persona_id = (
            persona_id
            if persona_id
            else self.config.get("default_persona_id", DEFAULT_PERSONA_ON_NONE)
        )

        data_to_insert = [
            {
                "personality_id": effective_persona_id,
                "session_id": session_id,
                "content": text,
                VECTOR_FIELD_NAME: embedding_vector,
                "create_time": int(time.time()),
            }
        ]

        # --- 插入 Milvus（使用 Mnemosyne 的连接）---
        try:
            collection = Collection(
                name=self.collection_name, using=MNEMOSYNE_CONNECTION_ALIAS
            )
            collection.load()

            loop = asyncio.get_event_loop()

            def _do_insert():
                return collection.insert(data_to_insert)

            mutation_result = await loop.run_in_executor(None, _do_insert)

            if mutation_result and mutation_result.insert_count > 0:
                inserted_ids = mutation_result.primary_keys
                logger.info(
                    f"MnemosyneManual 成功插入记忆 "
                    f"(ID: {inserted_ids}, Session: {session_id[:16]}...)"
                )

                # Flush 确保立即可用
                try:
                    def _do_flush():
                        collection.flush()

                    await loop.run_in_executor(None, _do_flush)
                    logger.debug(
                        f"集合 '{self.collection_name}' 已 flush"
                    )
                except Exception as flush_err:
                    logger.warning(f"Flush 集合时出错 (记忆已插入): {flush_err}")

                return {
                    "success": True,
                    "message": "记忆插入成功",
                    "memory_id": inserted_ids[0] if inserted_ids else None,
                }
            else:
                return {
                    "success": False,
                    "message": f"Milvus 插入失败。结果: {mutation_result}",
                }

        except MilvusException as e:
            logger.error(f"MnemosyneManual Milvus 插入出错: {e}", exc_info=True)
            return {"success": False, "message": f"Milvus 插入出错: {e}"}
        except Exception as e:
            logger.error(f"MnemosyneManual 插入记忆时发生未知错误: {e}", exc_info=True)
            return {"success": False, "message": f"插入记忆时发生错误: {e}"}

    # -----------------------------------------------------------------------
    # 事件钩子
    # -----------------------------------------------------------------------

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """AstrBot 初始化完成后，尝试加载 Embedding Provider。"""
        try:
            logger.info("MnemosyneManual: AstrBot 已加载，开始初始化组件...")

            # 加载 Embedding Provider
            self.embedding_provider = self._get_embedding_provider()
            if self.embedding_provider:
                self._embedding_provider_ready = True
                logger.info("MnemosyneManual: Embedding Provider 已就绪")
            else:
                logger.warning(
                    "MnemosyneManual: Embedding Provider 尚未就绪，"
                    "将在首次使用时重试"
                )

            # 验证 Mnemosyne 连接
            if self._check_mnemosyne_connection():
                logger.info("MnemosyneManual: 成功借用 Mnemosyne 的 Milvus 连接")
                # 列出集合验证
                collections = utility.list_collections(
                    using=MNEMOSYNE_CONNECTION_ALIAS
                )
                logger.info(f"MnemosyneManual: 可用集合: {collections}")
            else:
                logger.warning(
                    "MnemosyneManual: Mnemosyne 的 Milvus 连接尚不可用，"
                    "将在首次使用 /madd 时重试"
                )

        except Exception as e:
            logger.error(
                f"MnemosyneManual on_astrbot_loaded 出错: {e}", exc_info=True
            )

    # -----------------------------------------------------------------------
    # 命令: /madd
    # -----------------------------------------------------------------------

    @filter.command("madd")
    async def madd_cmd(self, event: AstrMessageEvent, text: str):
        """手动向 Mnemosyne 的记忆数据库中插入一条记忆。

        使用示例: /madd 这是一条手动记忆内容
        """
        if not text or not text.strip():
            yield event.plain_result(
                "用法: /madd <记忆文本>\n"
                "示例: /madd Felis Abyssalis 喜欢在深夜调试代码"
            )
            return

        # 获取当前会话 ID（与 Mnemosyne 的 session_id 来源一致）
        session_id = event.unified_msg_origin
        if not session_id:
            yield event.plain_result("无法获取当前会话 ID，请稍后重试")
            return

        yield event.plain_result("正在插入记忆...")

        result = await self.insert_memory(
            text=text.strip(),
            session_id=session_id,
        )

        if result["success"]:
            memory_id = result.get("memory_id", "N/A")
            yield event.plain_result(
                f"记忆插入成功\n"
                f"ID: {memory_id}\n"
                f"内容: {text.strip()[:100]}{'...' if len(text.strip()) > 100 else ''}"
            )
        else:
            yield event.plain_result(f"记忆插入失败: {result['message']}")

    # -----------------------------------------------------------------------
    # 生命周期
    # -----------------------------------------------------------------------

    async def terminate(self):
        """插件停止时清理资源。"""
        # 我们不需要清理任何连接，因为连接属于 Mnemosyne
        logger.info("MnemosyneManual 插件已停止")
