#!/usr/bin/env python3
"""
Excel 自动注册到数据源服务
支持自动缓存和增量导入
"""
# ruff: noqa: E501

import hashlib
import json
import logging
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl
import pandas as pd

from chatdb.utils.exceptions import ConnectionError
from chatdb.utils.logger import logger
from chatdb.database.base import BaseDatabaseConnector
from chatdb.database.excel.data_processor import DataProcessor
from chatdb.database.schema import SchemaGenerator
from chatdb.storage import DataCacheManager


# ExcelCacheManager 已迁移到 chatdb.storage.meta_data.DataCacheManager
# 保留别名以保持向后兼容
ExcelCacheManager = DataCacheManager


class ExcelConnector(BaseDatabaseConnector):
    """Excel 连接器和自动注册服务（合并类）"""

    _instance = None
    _lock = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保只有一个实例（仅用于自动注册服务）"""
        # 如果提供了 file_path，说明是作为连接器使用，不使用单例
        if "file_path" in kwargs:
            return super().__new__(cls)
        
        # 否则作为自动注册服务使用，使用单例
        if cls._instance is None:
            if cls._lock is None:
                import threading
                cls._lock = threading.Lock()

            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        file_path: str | None = None,
        sheet_name: str | None = None,
        llm_client=None,
        model_name=None,
        connection_url: str | None = None,
    ):
        """
        初始化 Excel 连接器/服务
        
        Args:
            file_path: Excel 文件路径（如果提供，作为连接器使用）
            sheet_name: 工作表名称（连接器模式）
            llm_client: LLM 客户端（自动注册服务模式）
            model_name: 模型名称（自动注册服务模式）
            connection_url: 连接 URL（连接器模式，通常不需要）
        """
        # 连接器模式：有 file_path
        if file_path:
            self._file_path = os.path.abspath(file_path)
            self._sheet_name = sheet_name
            self._table_name = sheet_name or "excel_data"
            self._temp_file: str | None = None

            # 创建临时 DuckDB 文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".duckdb")
            temp_file.close()
            self._temp_file = temp_file.name

            # Excel 通过 DuckDB 读取，使用临时文件数据库
            super().__init__(f"duckdb:///{self._temp_file}")
            return

        # 自动注册服务模式：没有 file_path，需要初始化基类
        super().__init__(connection_url)
        
        if not hasattr(self, "_initialized"):
            self.cache_manager = DataCacheManager()
            self.llm_client = llm_client
            self.model_name = model_name

            # 数据库文件存储在 data/duckdb 目录下
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent.parent
            self.db_storage_dir = project_root / "data" / "duckdb"
            self.db_storage_dir.mkdir(parents=True, exist_ok=True)

            # 初始化通用处理器
            self.data_processor = DataProcessor()
            # 确保在创建 SchemaGenerator 时传递正确的 llm_client
            self.schema_generator = SchemaGenerator(connector=None, llm_client=llm_client, model_name=model_name)

            self._initialized = True
        
        # 每次都更新 llm_client 和 model_name，确保使用最新的值
        self.llm_client = llm_client
        self.model_name = model_name
        # 更新通用处理器的 LLM 客户端
        self.schema_generator.llm_client = llm_client
        self.schema_generator.model_name = model_name



    def _detect_id_columns_with_llm(self, df: pd.DataFrame, table_name: str) -> List[str]:
        """使用 LLM 识别 ID 列（委托给 SchemaGenerator）"""
        return self.schema_generator.detect_id_columns_with_llm(df, table_name)

    def get_table_preview_data(
        self,
        sheets_data: List[Tuple[str, pd.DataFrame]],
        source_column_name: str = "数据类型",
    ) -> pd.DataFrame:
        """
        合并多个sheet的数据，添加来源标识列

        Args:
            sheets_data: [(sheet_name, dataframe), ...] 列表
            source_column_name: 来源列的列名，默认为"数据类型"

        Returns:
            合并后的DataFrame
        """
        if not sheets_data:
            raise ValueError("sheets_data不能为空")

        if len(sheets_data) == 1:
            # 只有一个sheet，直接添加来源列
            sheet_name, df = sheets_data[0]
            df_copy = df.copy()
            df_copy[source_column_name] = sheet_name
            return df_copy

        # 多个sheet的情况
        merged_dfs = []

        # 收集所有列名（按出现顺序）
        all_columns = []
        seen_columns = set()
        for sheet_name, df in sheets_data:
            for col in df.columns:
                if col not in seen_columns:
                    all_columns.append(col)
                    seen_columns.add(col)

        logger.info(f"合并{len(sheets_data)}个sheet，共{len(all_columns)}个唯一列")

        # 对每个sheet进行列对齐
        for sheet_name, df in sheets_data:
            df_copy = df.copy()

            # 添加缺失的列（填充为None）
            for col in all_columns:
                if col not in df_copy.columns:
                    df_copy[col] = None

            # 按统一的列顺序重新排列
            df_copy = df_copy[all_columns]

            # 添加来源列
            df_copy[source_column_name] = sheet_name

            merged_dfs.append(df_copy)
            logger.debug(f"Sheet '{sheet_name}': {len(df)}行 -> 对齐后{len(df_copy)}行")

        # 合并所有DataFrame
        merged_df = pd.concat(merged_dfs, ignore_index=True)
        logger.info(f"合并完成：总行数 {len(merged_df)}")

        return merged_df

    def get_table_preview_data(
        self, db_path: str, table_name: str, limit: int = None, file_name: str = None
    ) -> Dict:
        """从数据库中获取表格预览数据（委托给 DataProcessor）"""
        return self.data_processor.get_table_preview_data(db_path, table_name, limit, file_name)

    def _generate_create_table_sql(self, db_path: str, table_name: str) -> str:
        """生成建表SQL语句（委托给 DataProcessor）"""
        return self.data_processor.generate_create_table_sql(db_path, table_name)

    async def process_excel_multi_tables(
        self,
        excel_file_path: str,
        force_reimport: bool = False,
        original_filename: str = None,
        conv_uid: str = None,
        sheet_names: List[str] = None,
        preview_limit: int = None,
    ) -> Dict:
        """处理Excel文件，将每个sheet存为独立的表（多表模式）

        Args:
            excel_file_path: Excel文件路径
            force_reimport: 是否强制重新导入
            original_filename: 原始文件名（可选）
            conv_uid: 会话ID（可选）
            sheet_names: 要处理的sheet名称列表，如果为None则处理所有sheet
            preview_limit: 预览数据行数限制，None表示不限制
            
        Returns:
            包含多表信息的字典，结构如下：
            {
                "status": "imported" | "cached",
                "message": "...",
                "file_hash": "文件哈希",
                "db_name": "数据库名",
                "db_path": "数据库路径",
                "tables": [
                    {
                        "sheet_name": "sheet名称",
                        "table_name": "表名",
                        "table_hash": "表哈希",
                        "row_count": 行数,
                        "column_count": 列数,
                        "columns_info": [...],
                        "data_schema_json": "...",
                        "create_table_sql": "建表SQL",
                        ...
                    },
                    ...
                ],
                "conv_uid": "会话ID"
            }
        """
        if original_filename is None:
            original_filename = Path(excel_file_path).name

        # 读取Excel获取sheet信息
        file_ext = Path(excel_file_path).suffix.lower()
        if file_ext == '.xls':
            excel_file = pd.ExcelFile(excel_file_path, engine='xlrd')
        else:
            excel_file = pd.ExcelFile(excel_file_path)
        all_sheet_names = excel_file.sheet_names

        # 确定要处理的sheet
        if sheet_names is None:
            target_sheets = all_sheet_names
        else:
            target_sheets = []
            for name in sheet_names:
                if name in all_sheet_names:
                    target_sheets.append(name)
                else:
                    logger.warning(f"Sheet '{name}' 不存在，跳过")

            if not target_sheets:
                raise ValueError(f"指定的sheet都不存在。可用的sheet: {all_sheet_names}")

        # 计算文件级别的哈希
        file_hash = DataCacheManager.calculate_file_hash(excel_file_path, target_sheets)
        
        # 去除 Excel 文件的筛选状态
        excel_file_path = self.data_processor.remove_excel_filters(excel_file_path)

        # 检查缓存（多表模式）
        if not force_reimport:
            cached_tables = self.cache_manager.get_tables_by_file_hash(file_hash, source_type="excel")
            if cached_tables:
                # 检查所有缓存的表是否都存在
                first_table = cached_tables[0]
                if os.path.exists(first_table["db_path"]):
                    logger.info(f"多表缓存命中: {original_filename}, {len(cached_tables)}个表")
                    
                    # 获取每个表的预览数据
                    for table_info in cached_tables:
                        table_info["preview_data"] = self.get_table_preview_data(
                            table_info["db_path"],
                            table_info["table_name"],
                            preview_limit,
                            original_filename
                        )
                    
                    return {
                        "status": "cached",
                        "message": f"使用缓存数据，共{len(cached_tables)}个表",
                        "file_hash": file_hash,
                        "db_name": first_table["db_name"],
                        "db_path": first_table["db_path"],
                        "tables": cached_tables,
                        "conv_uid": conv_uid,
                    }

        # 没有缓存或强制重新导入
        logger.info(f"处理Excel（多表模式）: {original_filename}, {len(target_sheets)}个sheet", center=True, symbol="=")

        # 获取LLM客户端（如果没有提供，使用默认配置）
        if self.llm_client is None or self.model_name is None:
            try:
                from chatdb.utils.config import settings
                from chatdb.llm.factory import LLMFactory
                if self.llm_client is None:
                    self.llm_client = LLMFactory.create()
                if self.model_name is None:
                    self.model_name = self.llm_client.model if self.llm_client else None
            except Exception as e:
                logger.warning(f"无法创建默认LLM客户端: {e}")

        # 创建数据库
        db_name = f"excel_{file_hash[:8]}"
        db_filename = f"{db_name}.duckdb"
        db_path = str(self.db_storage_dir / db_filename)

        import duckdb
        
        # 删除已存在的数据库文件（如果强制重新导入）
        if force_reimport and os.path.exists(db_path):
            os.remove(db_path)
            # 同时删除缓存记录
            self.cache_manager.delete_tables_by_file_hash(file_hash, source_type="excel")

        tables_info = []
        tables_basic_info = []  # 存储每个表的基础信息（用于统一生成schema）
        
        # 第一阶段：处理所有sheet的数据，生成基础信息
        from chatdb.utils.logger import log_step
        logger.info(f"第一阶段：处理{len(target_sheets)}个sheet的数据", center=True, symbol="=")
        for idx, sheet_name in enumerate(target_sheets):
            logger.info(f"Sheet {idx+1}/{len(target_sheets)}: {sheet_name}", center=True, symbol="-")
            
            try:
                # 步骤1：Excel 前处理（去除筛选、读取、检测表头、处理多级表头）
                log_step(logger, "步骤1：表头识别", f"Sheet '{sheet_name}'")
                df = await self.data_processor.process_excel_file_with_header_detection(
                    excel_file_path, sheet_name, self.llm_client, self.model_name
                )
                
                # 生成表名
                safe_sheet_name = "".join(
                    c if c.isalnum() or c == "_" else "_" for c in sheet_name
                )
                if safe_sheet_name and safe_sheet_name[0].isdigit():
                    safe_sheet_name = f"tbl_{safe_sheet_name}"
                if not safe_sheet_name or len(safe_sheet_name) < 2:
                    safe_sheet_name = f"sheet_{len(tables_info)}"
                table_name = safe_sheet_name
                
                # 计算表哈希（file_hash + sheet_name）
                table_hash = hashlib.sha256(
                    f"{file_hash}_{sheet_name}".encode("utf-8")
                ).hexdigest()
                
                # 步骤2：识别ID列（需要LLM）
                id_columns = []
                try:
                    log_step(logger, "步骤2：ID列识别", f"Sheet '{sheet_name}'")
                    id_columns = self._detect_id_columns_with_llm(df, table_name)
                except Exception as e:
                    logger.warning(f"  ID列识别失败: {e}")
                
                # 步骤3：数据清洗和转换
                log_step(logger, "步骤3：数据清洗", f"Sheet '{sheet_name}'")
                df = self.data_processor.process_excel_sheet(df, id_columns)
                logger.info(f"  清洗完成，数据行数: {len(df)}, 列数: {len(df.columns)}")
                
                # 步骤4：导入到 DuckDB
                log_step(logger, "步骤4：导入DuckDB", f"Sheet '{sheet_name}'")
                self.data_processor.import_to_duckdb(df, db_path, table_name, id_columns)
                logger.info(f"  导入完成，表名: {table_name}")
                
                # 生成建表SQL
                create_table_sql = self._generate_create_table_sql(db_path, table_name)
                
                # 存储基础信息，稍后统一生成表描述信息
                tables_basic_info.append({
                    "sheet_name": sheet_name,
                    "table_name": table_name,
                    "table_hash": table_hash,
                    "df": df,
                    "id_columns": id_columns,
                    "create_table_sql": create_table_sql,
                    "db_path": db_path,
                })
                
            except Exception as e:
                logger.error(f"  处理Sheet '{sheet_name}' 失败: {e}")
                # 继续处理其他sheet
                continue

        if not tables_basic_info:
            raise ValueError("没有成功处理任何sheet")
        
        # 第二阶段：生成所有表的描述信息
        logger.info(f"第二阶段：生成{len(tables_basic_info)}个表的描述信息", center=True, symbol="=")
        try:
            all_schemas = await self.schema_generator.generate_multi_table_schemas_with_llm(
                tables_basic_info, original_filename, self.data_processor
            )
        except Exception as e:
            logger.warning(f"  LLM生成表描述信息失败，使用基础描述: {e}")
            # 回退到单独生成
            all_schemas = {}
            for table_info in tables_basic_info:
                table_name = table_info["table_name"]
                df = table_info["df"]
                id_columns = table_info["id_columns"]
                all_schemas[table_name] = self.schema_generator.generate_basic_schema_json(df, table_name, id_columns)
        
        # 第三阶段：保存所有表的信息到缓存
        logger.info(f"第三阶段：保存{len(tables_basic_info)}个表的信息到缓存", center=True, symbol="=")
        for table_info in tables_basic_info:
            sheet_name = table_info["sheet_name"]
            table_name = table_info["table_name"]
            table_hash = table_info["table_hash"]
            df = table_info["df"]
            id_columns = table_info["id_columns"]
            create_table_sql = table_info["create_table_sql"]
            db_path = table_info["db_path"]
            
            # 获取该表的schema
            schema_json = all_schemas.get(table_name, self.schema_generator.generate_basic_schema_json(df, table_name, id_columns))
            summary_prompt = self.schema_generator.format_schema_as_prompt(schema_json, df, table_name)
            
            # 获取列信息
            conn = duckdb.connect(db_path, read_only=True)
            columns_result = conn.execute(f'DESCRIBE "{table_name}"').fetchall()
            columns_info = [
                {"name": col[0], "type": col[1], "dtype": str(df[col[0]].dtype)}
                for col in columns_result
            ]
            conn.close()
            
            # 保存到缓存
            self.cache_manager.save_table_cache_info(
                file_hash=file_hash,
                sheet_name=sheet_name,
                table_hash=table_hash,
                original_filename=original_filename,
                table_name=table_name,
                db_name=db_name,
                db_path=db_path,
                df=df,
                source_type="excel",
                summary_prompt=summary_prompt,
                data_schema_json=schema_json,
                id_columns=id_columns,
                create_table_sql=create_table_sql,
            )
            
            # 获取预览数据
            preview_data = self.get_table_preview_data(
                db_path, table_name, preview_limit, original_filename
            )
            
            tables_info.append({
                "sheet_name": sheet_name,
                "table_name": table_name,
                "table_hash": table_hash,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns_info": columns_info,
                "summary_prompt": summary_prompt,
                "data_schema_json": schema_json,
                "id_columns": id_columns,
                "create_table_sql": create_table_sql,
                "preview_data": preview_data,
            })


        return {
            "status": "imported",
            "message": f"成功导入{len(tables_info)}个表",
            "file_hash": file_hash,
            "db_name": db_name,
            "db_path": db_path,
            "tables": tables_info,
            "conv_uid": conv_uid,
        }


    def _get_sample_rows(self, df: pd.DataFrame, n: int = 2) -> list:
        """获取n行样本数据（委托给 DataProcessor）"""
        return self.data_processor.get_sample_rows(df, n)

    def update_summary_prompt(self, content_hash: str, summary_prompt: str):
        """
        更新数据理解提示词

        Args:
            content_hash: 内容哈希值
            summary_prompt: 新的数据理解提示词
        """
        self.cache_manager.update_summary_prompt(content_hash, summary_prompt, source_type="excel")

    def get_excel_info(self, content_hash: str) -> Optional[Dict]:
        """
        获取 Excel 信息

        Args:
            content_hash: 内容哈希值

        Returns:
            Excel 信息字典
        """
        return self.cache_manager.get_cached_info(content_hash, source_type="excel")

    def get_tables_info_for_selection(self, file_hash: str) -> List[Dict]:
        """
        获取多表信息用于表选择
        
        Args:
            file_hash: 文件哈希值
            
        Returns:
            表信息列表，每个元素包含表选择所需的信息
        """
        tables = self.cache_manager.get_tables_by_file_hash(file_hash, source_type="excel")
        
        result = []
        for table in tables:
            result.append({
                "table_name": table.get("table_name"),
                "sheet_name": table.get("sheet_name"),
                "table_hash": table.get("table_hash"),
                "create_table_sql": table.get("create_table_sql"),
                "data_schema_json": table.get("data_schema_json"),
                "row_count": table.get("row_count"),
                "column_count": table.get("column_count"),
                "db_path": table.get("db_path"),
            })
        
        return result

    def get_table_schema_by_name(
        self, file_hash: str, table_name: str
    ) -> Optional[Dict]:
        """
        根据文件哈希和表名获取单个表的完整信息
        
        Args:
            file_hash: 文件哈希值
            table_name: 表名
            
        Returns:
            表的完整信息字典
        """
        tables = self.cache_manager.get_tables_by_file_hash(file_hash, source_type="excel")
        
        for table in tables:
            if table.get("table_name") == table_name:
                return table
        
        return None

    def get_combined_schema_for_tables(
        self, file_hash: str, table_names: List[str]
    ) -> Dict:
        """
        获取多个表的组合Schema信息，用于query改写
        
        Args:
            file_hash: 文件哈希值
            table_names: 要组合的表名列表
            
        Returns:
            组合后的Schema信息
        """
        tables = self.cache_manager.get_tables_by_file_hash(file_hash, source_type="excel")
        
        selected_tables = []
        for table in tables:
            if table.get("table_name") in table_names:
                selected_tables.append(table)
        
        if not selected_tables:
            return {}
        
        # 如果只有一个表，直接返回其schema
        if len(selected_tables) == 1:
            table = selected_tables[0]
            return {
                "table_name": table.get("table_name"),
                "data_schema_json": table.get("data_schema_json"),
                "create_table_sql": table.get("create_table_sql"),
                "summary_prompt": table.get("summary_prompt"),
                "db_path": table.get("db_path"),
            }
        
        # 多个表的情况，组合schema
        combined_columns = []
        table_descriptions = []
        create_sqls = []
        
        for table in selected_tables:
            table_name = table.get("table_name")
            schema_json = table.get("data_schema_json")
            
            if schema_json:
                try:
                    schema = json.loads(schema_json) if isinstance(schema_json, str) else schema_json
                    
                    # 为每个列添加表名前缀
                    for col in schema.get("columns", []):
                        col_copy = col.copy()
                        col_copy["table_name"] = table_name
                        col_copy["full_column_name"] = f"{table_name}.{col.get('column_name', '')}"
                        combined_columns.append(col_copy)
                    
                    if schema.get("table_description"):
                        table_descriptions.append(f"【{table_name}】: {schema['table_description']}")
                except Exception as e:
                    logger.warning(f"解析表 {table_name} 的schema失败: {e}")
            
            if table.get("create_table_sql"):
                create_sqls.append(table.get("create_table_sql"))
        
        combined_schema = {
            "tables": [t.get("table_name") for t in selected_tables],
            "table_descriptions": "\n".join(table_descriptions),
            "columns": combined_columns,
            "create_table_sqls": "\n\n".join(create_sqls),
        }
        
        return {
            "is_multi_table": True,
            "table_names": [t.get("table_name") for t in selected_tables],
            "data_schema_json": json.dumps(combined_schema, ensure_ascii=False),
            "create_table_sql": "\n\n".join(create_sqls),
            "summary_prompt": "\n".join(table_descriptions),
            "db_path": selected_tables[0].get("db_path"),  # 所有表在同一个数据库
        }


    # ==================== 连接器模式方法 ====================

    def _get_default_connection_url(self) -> str:
        """获取默认连接 URL（连接器模式）"""
        if hasattr(self, "_temp_file") and self._temp_file:
            return f"duckdb:///{self._temp_file}"
        # 自动注册服务模式不需要连接 URL
        return "duckdb:///:memory:"

    async def connect(self) -> None:
        """建立连接并加载 Excel 数据（连接器模式）"""
        if not hasattr(self, "_file_path") or not self._file_path:
            # 自动注册服务模式不需要连接
            return

        try:
            import pandas as pd
            import duckdb

            # 读取 Excel 文件（使用基类的公共方法）
            df = BaseDatabaseConnector.read_excel_file(
                self._file_path, self._sheet_name
            )

            # 使用 DuckDB 将 DataFrame 写入临时数据库
            con = duckdb.connect(self._temp_file)
            con.execute(f"CREATE TABLE {self._table_name} AS SELECT * FROM df")
            con.close()

            # 调用父类连接方法
            await super().connect()

            logger.info(
                f"[{self.__class__.__name__}] Excel 文件加载成功: {self._file_path}, "
                f"表名: {self._table_name}, 行数: {len(df)}"
            )

        except ImportError:
            raise ConnectionError(
                "需要安装 pandas、openpyxl 和 duckdb: pip install pandas openpyxl duckdb"
            )
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Excel 文件加载失败: {e}")
            raise ConnectionError(f"Excel 文件加载失败: {e}")

    async def disconnect(self) -> None:
        """关闭连接并清理临时文件（连接器模式）"""
        if hasattr(self, "_file_path") and self._file_path:
            await super().disconnect()

            # 清理临时文件
            if hasattr(self, "_temp_file") and self._temp_file:
                try:
                    if os.path.exists(self._temp_file):
                        os.unlink(self._temp_file)
                        logger.debug(
                            f"[{self.__class__.__name__}] 临时文件已删除: {self._temp_file}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[{self.__class__.__name__}] 删除临时文件失败: {e}"
                    )

    def _get_pool_size(self) -> int:
        """Excel 连接池大小"""
        return 1

    def _get_max_overflow(self) -> int:
        """Excel 最大溢出连接数"""
        return 0


# 向后兼容别名
ExcelAutoRegisterService = ExcelConnector
