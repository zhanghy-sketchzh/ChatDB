"""
ExploreByDimensionTool - 维度探索工具

按指定维度拆解已有查询结果，用于分析数据来源/贡献。

适用场景：
- 已有基础聚合结果，需要分析来源
- 回答"流水增长来自哪里"类问题
- 需要按维度拆解数据

输入：
- dimension: 要拆解的维度列名
- base_sql: 基础查询 SQL（提取 WHERE 条件）
- table_name: 表名

输出：
- analysis_result: 拆解结果
- rows: 每个维度值的统计
"""

from typing import Any, TYPE_CHECKING

from chatdb.tools.base import BaseTool, ToolMetadata, ToolParameter, ToolResult
from chatdb.database.base import BaseDatabaseConnector
from chatdb.utils.logger import get_component_logger

if TYPE_CHECKING:
    from chatdb.core.react_state import ReActState
    from chatdb.agents.base import AgentContext


class ExploreByDimensionTool(BaseTool):
    """
    维度探索工具
    
    核心能力：
    - 按维度拆解指标
    - 分析贡献来源
    - 支持多步分析
    """
    
    def __init__(self, db_connector: BaseDatabaseConnector):
        metadata = ToolMetadata(
            name="explore_by_dimension",
            description="按指定维度拆解已有结果，分析数据来源、结构、贡献者",
            category="analysis",
            inputs={
                "dimension": {"type": "str", "description": "要拆解的维度列名"},
                "metric_column": {"type": "str", "description": "要聚合的指标列"},
                "table_name": {"type": "str", "description": "表名"},
                "where_clause": {"type": "str", "description": "WHERE 条件（可选）"},
            },
            outputs={
                "rows": {"type": "list", "description": "每个维度值的统计结果"},
                "row_count": {"type": "int", "description": "维度取值个数"},
                "dimension": {"type": "str", "description": "使用的维度名"},
            },
            subtools=[
                "explore_by_dimension.auto_select",      # 自动选择合适的维度
                "explore_by_dimension.build_groupby_sql", # 构造 GROUP BY SQL
                "explore_by_dimension.get_top_contributors", # 获取主要贡献者
            ],
            is_core=False,
            dependencies=["execute_and_evaluate"],
        )
        super().__init__(metadata)
        
        self.db_connector = db_connector
        self._log = get_component_logger("ExploreDimensionTool")
    
    @property
    def name(self) -> str:
        return "explore_by_dimension"
    
    @property
    def description(self) -> str:
        return """维度探索工具：按指定维度拆解已有结果。

使用场景：
- 已有基础聚合结果（如总流水）
- 需要分析该结果在某个维度下的构成
- 回答"来自哪里""主要贡献"类问题

示例：
- 按「投资公司标签」拆解流水
- 按「产品大类」分析利润构成"""
    
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="dimension", type="string", description="维度列名"),
            ToolParameter(name="metric_column", type="string", description="指标列名"),
            ToolParameter(name="table_name", type="string", description="表名"),
            ToolParameter(name="where_clause", type="string", description="WHERE条件", required=False),
        ]
    
    async def execute(
        self,
        dimension: str,
        metric_column: str,
        table_name: str,
        where_clause: str = "",
        limit: int = 20,
        **kwargs: Any,
    ) -> ToolResult:
        """执行维度探索"""
        self._log.info(f"探索维度: {dimension}")
        
        try:
            # 构建 GROUP BY SQL
            where_part = f" WHERE {where_clause}" if where_clause else ""
            sql = f'''
SELECT "{dimension}", SUM("{metric_column}") AS total 
FROM "{table_name}"{where_part} 
GROUP BY "{dimension}" 
ORDER BY total DESC 
LIMIT {limit}
'''
            
            # 执行查询
            rows = await self.db_connector.execute_query(sql)
            
            return ToolResult.ok(
                data={
                    "rows": rows,
                    "row_count": len(rows),
                    "dimension": dimension,
                    "metric_column": metric_column,
                    "sql": sql.strip(),
                },
                message=f"按「{dimension}」拆解，共 {len(rows)} 类",
            )
        
        except Exception as e:
            self._log.error(f"探索失败: {e}")
            return ToolResult.fail(str(e))
    
    async def __call__(
        self,
        state: "ReActState",
        context: "AgentContext",
        dimension: str | None = None,
        **kwargs: Any,
    ) -> None:
        """ReAct 模式执行：直接修改 state"""
        from chatdb.core.react_state import ReActPhase, AnalysisPhase
        
        state.phase = ReActPhase.CRITIQUE
        state.think("按维度拆解，探索分析")
        
        table = state.table_name
        if not table:
            state.reflect("缺少表名，无法探索")
            return
        
        # 自动选择维度和指标
        dim_col, metric_col = self._pick_dimension_and_metric(state, dimension)
        if not dim_col or not metric_col:
            state.reflect("无法确定维度列或指标列，跳过探索")
            state.need_more_analysis = False
            return
        
        # 检查是否已探索过
        if dim_col in state.explored_dimensions:
            unexplored = state.get_unexplored_dimensions(
                [c.get("name") or c.get("column_name") for c in state.available_columns]
            )
            if unexplored:
                dim_col = unexplored[0]
            else:
                state.reflect("所有维度已探索完毕")
                state.need_more_analysis = False
                return
        
        # 提取 WHERE 子句
        where_clause = self._extract_where_clause(state.current_sql or state.final_sql or "")
        
        result = await self.execute(
            dimension=dim_col,
            metric_column=metric_col,
            table_name=table,
            where_clause=where_clause,
        )
        
        if result.success:
            rows = result.data.get("rows", [])
            
            # 使用新的 AnalysisSlice
            state.add_analysis_slice(
                phase=AnalysisPhase.EXPLORE,
                sql=result.data.get("sql", ""),
                dimension=dim_col,
                filters={"where_clause": where_clause} if where_clause else {},
                rows=rows[:20],
                row_count=len(rows),
                metric_col=metric_col,
            )
            
            state.need_more_analysis = False
            state.reflect(f"已按「{dim_col}」拆解，共 {len(rows)} 类")
        else:
            state.reflect(f"探索执行失败: {result.error}")
            state.need_more_analysis = False
    
    def _pick_dimension_and_metric(
        self,
        state: "ReActState",
        preferred_dimension: str | None = None,
    ) -> tuple[str | None, str | None]:
        """选择维度和指标列"""
        cols = state.available_columns or []
        if not cols:
            return None, None
        
        dim_col = preferred_dimension
        metric_col = None
        
        for c in cols:
            name = c.get("name") or c.get("column_name") or ""
            typ = (c.get("type") or c.get("column_type") or "").upper()
            
            # 维度：字符串类型
            if not dim_col and ("VARCHAR" in typ or "TEXT" in typ or "STRING" in typ):
                dim_col = name
            
            # 指标：数值类型
            if not metric_col and ("DOUBLE" in typ or "DECIMAL" in typ or "INT" in typ or "FLOAT" in typ):
                metric_col = name
            
            if dim_col and metric_col:
                break
        
        # Fallback
        if not dim_col and cols:
            dim_col = cols[0].get("name") or cols[0].get("column_name")
        
        return dim_col, metric_col
    
    def _extract_where_clause(self, sql: str) -> str:
        """从 SQL 中提取 WHERE 子句"""
        if not sql or "WHERE" not in sql.upper():
            return ""
        
        try:
            start = sql.upper().find("WHERE") + 5
            end = len(sql)
            for sep in (" GROUP BY", " ORDER BY", " LIMIT", ";"):
                i = sql.upper().find(sep, start)
                if i >= 0:
                    end = min(end, i)
            return sql[start:end].strip()
        except Exception:
            return ""
