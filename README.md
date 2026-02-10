# ChatDB

基于 LLM 多智能体的自然语言数据库查询系统。通过自然语言与数据库交互，自动生成 SQL 查询并返回结果和智能总结。

## ✨ 特性

- 🗣️ **自然语言查询**: 使用自然语言描述查询需求，无需手写 SQL
- 🤖 **多智能体协作**: 
  - SQL 生成智能体 - 将自然语言转换为 SQL
  - SQL 验证智能体 - 验证 SQL 安全性和正确性
  - 结果总结智能体 - 对查询结果进行分析总结
- 🔄 **多数据库支持**: PostgreSQL、MySQL、SQLite
- 🔌 **多 LLM 支持**: OpenAI GPT、Anthropic Claude
- 🔒 **安全查询**: 仅支持 SELECT 查询，防止数据篡改
- 🌐 **RESTful API**: 完整的 API 接口，易于集成

## 📁 项目结构

```
ChatDB/
├── src/
│   └── chatdb/
│       ├── __init__.py
│       ├── cli.py                 # 命令行入口
│       ├── core/                  # 核心模块
│       │   ├── config.py          # 配置管理
│       │   ├── exceptions.py      # 自定义异常
│       │   └── logger.py          # 日志配置
│       ├── database/              # 数据库模块
│       │   ├── connector.py       # 数据库连接器
│       │   └── schema.py          # Schema 检查器
│       ├── llm/                   # LLM 模块
│       │   ├── base.py            # LLM 抽象基类
│       │   ├── openai_llm.py      # OpenAI 实现
│       │   ├── anthropic_llm.py   # Anthropic 实现
│       │   └── factory.py         # LLM 工厂
│       ├── agents/                # 智能体模块
│       │   ├── base.py            # 智能体基类
│       │   ├── prompts.py         # Prompt 模板
│       │   ├── sql_generator.py   # SQL 生成智能体
│       │   ├── sql_validator.py   # SQL 验证智能体
│       │   ├── summarizer.py      # 结果总结智能体
│       │   └── orchestrator.py    # 智能体协调器
│       └── api/                   # API 模块
│           ├── app.py             # FastAPI 应用
│           ├── schemas.py         # 数据模型
│           ├── dependencies.py    # 依赖注入
│           └── routes/            # API 路由
│               ├── query.py       # 查询接口
│               ├── database.py    # 数据库接口
│               └── health.py      # 健康检查
├── tests/                         # 测试目录
├── pyproject.toml                 # 项目配置
├── .env.example                   # 环境变量示例
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/chatdb.git
cd chatdb

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装项目
pip install -e .

# 或安装开发依赖
pip install -e ".[dev]"
```

### 2. 配置环境变量

```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env 文件，配置以下内容：
# - LLM API Key (OpenAI 或 Anthropic)
# - 数据库连接信息
```

### 3. 启动服务

```bash
# 启动 API 服务器
chatdb serve

# 或使用 uvicorn 直接启动
uvicorn chatdb.api.app:app --reload
```

### 4. 访问 API

- API 文档: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📖 API 使用示例

### 自然语言查询

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "查询销售额最高的前10个产品"}'
```

响应示例：

```json
{
  "success": true,
  "query": "查询销售额最高的前10个产品",
  "sql": "SELECT product_name, SUM(amount) as total_sales FROM orders GROUP BY product_name ORDER BY total_sales DESC LIMIT 10;",
  "result": [
    {"product_name": "产品A", "total_sales": 50000},
    {"product_name": "产品B", "total_sales": 45000}
  ],
  "row_count": 10,
  "summary": "查询结果显示，销售额最高的产品是\"产品A\"，总销售额达到50000元..."
}
```

### 获取数据库 Schema

```bash
curl "http://localhost:8000/database/schema"
```

### 仅生成 SQL

```bash
curl -X POST "http://localhost:8000/query/sql-only" \
  -H "Content-Type: application/json" \
  -d '{"query": "统计每个部门的员工数量"}'
```

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `OPENAI_MODEL` | OpenAI 模型名称 | gpt-4-turbo-preview |
| `ANTHROPIC_API_KEY` | Anthropic API Key | - |
| `DEFAULT_LLM_PROVIDER` | 默认 LLM 提供商 | openai |
| `POSTGRES_HOST` | PostgreSQL 主机 | localhost |
| `POSTGRES_PORT` | PostgreSQL 端口 | 5432 |
| `DEFAULT_DB_TYPE` | 默认数据库类型 | postgresql |

## 🏗️ 架构设计

### 多智能体协作流程

```
用户查询
    ↓
┌─────────────────┐
│ SQL 生成智能体   │  ← 将自然语言转换为 SQL
└────────┬────────┘
         ↓
┌─────────────────┐
│ SQL 验证智能体   │  ← 验证 SQL 安全性和正确性
└────────┬────────┘
         ↓
┌─────────────────┐
│   执行 SQL 查询   │  ← 在数据库中执行查询
└────────┬────────┘
         ↓
┌─────────────────┐
│ 结果总结智能体   │  ← 分析结果并生成总结
└────────┬────────┘
         ↓
    返回结果
```

### 核心组件

1. **AgentOrchestrator**: 智能体协调器，管理多智能体的执行流程
2. **DatabaseConnector**: 数据库连接器，支持多种数据库类型
3. **SchemaInspector**: Schema 检查器，提取数据库元数据
4. **LLMFactory**: LLM 工厂，统一创建 LLM 实例

## 🧪 测试

```bash
# 运行测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=chatdb
```

## 📝 开发

```bash
# 代码格式化
black src/

# 代码检查
ruff check src/

# 类型检查
mypy src/
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License


