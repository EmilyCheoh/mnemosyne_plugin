## 工作原理

```mermaid
graph LR
    A["/madd 文本"] --> B["EmbeddingProvider.get_embedding()"]
    B --> C["构建 Mnemosyne 兼容的 insert dict"]
    C --> D["Collection.insert()"]
    D --> E["Milvus Lite DB<br/>(与 Mnemosyne 共享)"]
    E --> F["Mnemosyne 正常检索到"]
```

插入的每条记录包含：`personality_id`, `session_id`, `content`, [embedding], `create_time` — 与 Mnemosyne 自动生成的记录格式完全一致。

## 部署步骤

1. 将 `astrbot_plugin_mnemosyne_manual` 目录上传到云服务器的 AstrBot 插件目录
2. 在 AstrBot WebUI 中启用插件
3. **关键配置**：确保以下三项与 Mnemosyne 完全一致：
   - `milvus_lite_path`
   - `collection_name`
   - `embedding_provider_id`（或都留空使用默认）
4. 在 QQ 中测试：`/madd 这是一条手动插入的记忆`

## 使用方式

```
/madd <记忆文本>
```

示例：
```
/madd Noir是用户的赛博小猫，有蓝色的眼睛
```
