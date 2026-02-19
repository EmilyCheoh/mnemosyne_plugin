"""
MnemosyneManual - Mnemosyne 手动记忆注入伴生插件
直接向 Mnemosyne 的 Milvus 向量数据库中插入手动编写的记忆条目。
插入的记录与 Mnemosyne 自动生成的记录格式完全一致，
使 Mnemosyne 在检索时能正常 fetch 到手动插入的记忆。
"""

import asyncio
import os
import platform
import time
from typing import Any, cast
from urllib.parse import urlparse

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.provider.provider import EmbeddingProvider

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

# ---------------------------------------------------------------------------
# Mnemosyne 兼容常量（必须与 Mnemosyne core/constants.py 完全一致）
# ---------------------------------------------------------------------------
DEFAULT_COLLECTION_NAME = "mnemosyne_default"
DEFAULT_EMBEDDING_DIM = 1024
VECTOR_FIELD_NAME = "embedding"
PRIMARY_FIELD_NAME = "memory_id"
DEFAULT_PERSONA_ON_NONE = "UNKNOWN_PERSONA"


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
    共享 Mnemosyne 的 Milvus 数据库、集合和 Embedding 模型，
    以完全兼容的数据格式写入记忆记录。
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.context = context

        # --- 组件状态 ---
        self.collection_name: str = DEFAULT_COLLECTION_NAME
        self.embedding_provider: EmbeddingProvider | None = None
        self._embedding_provider_ready = False
        self._warned_missing_provider_ids: set[str] = set()
        self._milvus_connected = False
        self._milvus_alias: str = "mnemosyne_manual"
        self._milvus_uri: str | None = None
        self._is_lite: bool = False
        self.plugin_data_dir: str | None = None

        logger.info("开始初始化 MnemosyneManual 插件...")
        asyncio.create_task(self._initialize_plugin_async())

    # -----------------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------------

    async def _initialize_plugin_async(self):
        """非阻塞异步初始化流程。"""
        try:
            # 1. 获取插件数据目录
            try:
                from astrbot.api.star import StarTools
                plugin_data_dir = StarTools.get_data_dir()
                self.plugin_data_dir = str(plugin_data_dir) if plugin_data_dir else None
                logger.info(f"MnemosyneManual 数据目录: {self.plugin_data_dir}")
            except Exception as e:
                logger.warning(f"无法获取插件数据目录: {e}")
                self.plugin_data_dir = None

            # 2. 读取集合名称
            self.collection_name = self.config.get(
                "collection_name", DEFAULT_COLLECTION_NAME
            )
            logger.info(
                f"MnemosyneManual 将写入集合: '{self.collection_name}'"
            )

            # 3. 配置 Milvus 连接（延迟连接）
            self._configure_milvus()

            # 4. Embedding Provider 延迟初始化
            logger.info("Embedding Provider 将在首次使用时加载")

            logger.info("MnemosyneManual 插件基础初始化完成")

        except Exception as e:
            logger.error(
                f"MnemosyneManual 初始化失败: {e}", exc_info=True
            )

    def _configure_milvus(self):
        """根据配置确定 Milvus 连接参数（不立即连接）。"""
        is_windows = platform.system() == "Windows"
        lite_path = self.config.get("milvus_lite_path", "") if not is_windows else ""
        milvus_address = self.config.get("address", "")

        if lite_path:
            # Milvus Lite 模式
            self._is_lite = True
            # 解析路径
            if self.plugin_data_dir and not os.path.isabs(lite_path):
                full_path = os.path.join(self.plugin_data_dir, lite_path)
            else:
                full_path = lite_path
            if not full_path.endswith(".db"):
                full_path = os.path.join(full_path, "mnemosyne_lite.db")
            # 确保目录存在
            db_dir = os.path.dirname(full_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            self._milvus_uri = full_path
            logger.info(f"MnemosyneManual 将使用 Milvus Lite: {full_path}")

        elif milvus_address:
            # 标准 Milvus
            self._is_lite = False
            if milvus_address.startswith(("http://", "https://")):
                self._milvus_uri = milvus_address
            else:
                self._milvus_uri = f"http://{milvus_address}"
            logger.info(
                f"MnemosyneManual 将使用标准 Milvus: {self._milvus_uri}"
            )

        else:
            # 默认 Milvus Lite
            self._is_lite = True
            if self.plugin_data_dir:
                # 使用与 Mnemosyne 相同的默认路径逻辑
                # Mnemosyne 的默认路径是 plugin_data_dir/mnemosyne_lite.db
                # 但由于我们是不同的插件，数据目录不同
                # 用户必须显式配置 milvus_lite_path 来指向 Mnemosyne 的数据库
                logger.warning(
                    "未配置 milvus_lite_path 和 address，"
                    "请在插件配置中填写与 Mnemosyne 相同的 milvus_lite_path，"
                    "否则将无法访问 Mnemosyne 的记忆数据库"
                )
                default_path = os.path.join(
                    self.plugin_data_dir, "mnemosyne_lite.db"
                )
                self._milvus_uri = default_path
            else:
                logger.error("无法确定 Milvus 数据库路径")
                self._milvus_uri = None

    def _connect_milvus(self) -> bool:
        """建立 Milvus 连接。返回是否成功。"""
        if self._milvus_connected:
            return True

        if not self._milvus_uri:
            logger.error("Milvus URI 未配置，无法连接")
            return False

        connect_params: dict[str, Any] = {"uri": self._milvus_uri}

        # 添加认证信息（仅标准 Milvus）
        if not self._is_lite:
            auth_config = self.config.get("authentication", {})
            if isinstance(auth_config, dict):
                if auth_config.get("token"):
                    connect_params["token"] = auth_config["token"]
                elif auth_config.get("user") and auth_config.get("password"):
                    connect_params["user"] = auth_config["user"]
                    connect_params["password"] = auth_config["password"]

        mode_name = "Milvus Lite" if self._is_lite else "标准 Milvus"
        logger.info(
            f"MnemosyneManual 正在连接 {mode_name} "
            f"(别名: {self._milvus_alias})..."
        )
        try:
            connections.connect(alias=self._milvus_alias, **connect_params)
            self._milvus_connected = True
            logger.info(f"MnemosyneManual 成功连接到 {mode_name}")
            return True
        except Exception as e:
            logger.error(f"MnemosyneManual 连接 {mode_name} 失败: {e}")
            self._milvus_connected = False
            return False

    def _ensure_connected(self) -> bool:
        """确保 Milvus 已连接。"""
        if self._milvus_connected:
            return True
        return self._connect_milvus()

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

                if provider is None and not hasattr(self.context, "provider_manager"):
                    provider = self.context.get_provider_by_id(emb_id)

                if provider and (
                    callable(getattr(provider, "embed_texts", None))
                    or callable(getattr(provider, "get_embedding", None))
                ):
                    logger.info(
                        f"MnemosyneManual 成功加载 Embedding Provider: {emb_id}"
                    )
                    return cast(EmbeddingProvider, provider)

            # 优先级 2: 框架默认
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

        # 确保 Milvus 连接
        if not self._ensure_connected():
            return {"success": False, "message": "Milvus 数据库连接失败"}

        # 确保集合存在
        try:
            has_collection = utility.has_collection(
                self.collection_name, using=self._milvus_alias
            )
            if not has_collection:
                return {
                    "success": False,
                    "message": (
                        f"集合 '{self.collection_name}' 不存在。"
                        "请先在 Mnemosyne 中执行 /memory init 创建集合"
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

        # --- 插入 Milvus ---
        try:
            collection = Collection(
                name=self.collection_name, using=self._milvus_alias
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
        """AstrBot 初始化完成后，尝试加载 Embedding Provider 并连接 Milvus。"""
        try:
            logger.info("MnemosyneManual: AstrBot 已加载，开始初始化组件...")
            self.embedding_provider = self._get_embedding_provider()
            if self.embedding_provider:
                self._embedding_provider_ready = True
                logger.info("MnemosyneManual: Embedding Provider 已就绪")
            else:
                logger.warning(
                    "MnemosyneManual: Embedding Provider 尚未就绪，"
                    "将在首次使用时重试"
                )

            # 尝试建立 Milvus 连接
            self._connect_milvus()

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
        logger.info("MnemosyneManual 正在停止...")
        if self._milvus_connected:
            try:
                connections.disconnect(self._milvus_alias)
                self._milvus_connected = False
                logger.info("MnemosyneManual: Milvus 连接已断开")
            except Exception as e:
                logger.error(f"MnemosyneManual 断开 Milvus 连接时出错: {e}")
        logger.info("MnemosyneManual 插件已停止")
