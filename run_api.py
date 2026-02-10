#!/usr/bin/env python3
"""启动 ChatDB API 服务"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "chatdb.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
