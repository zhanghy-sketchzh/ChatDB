#!/usr/bin/env python3
"""
混元 API 连通性诊断脚本

逐层检查：DNS → TCP → HTTP → API 响应，定位连接失败的具体原因。

用法：
    python examples/test_hunyuan_api.py
"""

from __future__ import annotations

import asyncio
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_W = 60


def _ok(msg: str):
    print(f"  ✓ {msg}")


def _fail(msg: str):
    print(f"  ✗ {msg}")


def _info(msg: str):
    print(f"  ℹ {msg}")


# ─── 1. 加载配置 ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    """从项目配置中读取混元 API 参数"""
    try:
        from chatdb.utils.config import settings
        return {
            "api_base": settings.llm.hunyuan_api_base,
            "api_key": settings.llm.hunyuan_api_key,
            "model": settings.llm.hunyuan_model,
        }
    except Exception as e:
        _fail(f"加载配置失败: {e}")
        # 回退：直接读 config.toml
        try:
            import tomllib
            with open("config.toml", "rb") as f:
                cfg = tomllib.load(f)
            llm = cfg.get("llm", {})
            return {
                "api_base": llm.get("hunyuan_api_base", ""),
                "api_key": llm.get("hunyuan_api_key", ""),
                "model": llm.get("hunyuan_model", "hunyuan-t1-latest"),
            }
        except Exception as e2:
            _fail(f"回退读取 config.toml 也失败: {e2}")
            return {}


# ─── 2. DNS 解析 ──────────────────────────────────────────────────────────────

def check_dns(host: str) -> bool:
    print(f"\n{'─' * _W}\n  步骤 1: DNS 解析 → {host}\n{'─' * _W}")
    try:
        addrs = socket.getaddrinfo(host, None)
        ips = sorted(set(a[4][0] for a in addrs))
        _ok(f"解析成功: {', '.join(ips)}")
        return True
    except socket.gaierror as e:
        _fail(f"DNS 解析失败: {e}")
        _info("可能原因:")
        _info("  1. 不在内网环境（混元 API 是内网服务）")
        _info("  2. DNS 服务器未配置内网域名")
        _info("  3. VPN 未连接或已断开")
        return False


# ─── 3. TCP 连接 ──────────────────────────────────────────────────────────────

def check_tcp(host: str, port: int) -> bool:
    print(f"\n{'─' * _W}\n  步骤 2: TCP 连接 → {host}:{port}\n{'─' * _W}")
    try:
        t0 = time.time()
        sock = socket.create_connection((host, port), timeout=5)
        latency = (time.time() - t0) * 1000
        sock.close()
        _ok(f"连接成功 (延迟 {latency:.0f}ms)")
        return True
    except socket.timeout:
        _fail("连接超时 (>5s)")
        _info("可能原因: 防火墙/安全组阻断、端口未开放")
        return False
    except ConnectionRefusedError:
        _fail("连接被拒绝")
        _info("可能原因: 服务未启动、端口错误")
        return False
    except OSError as e:
        _fail(f"连接失败: {e}")
        _info("可能原因: 网络不通、路由不可达")
        return False


# ─── 4. HTTP 请求（不带 body）──────────────────────────────────────────────────

async def check_http(url: str, api_key: str) -> bool:
    import httpx

    print(f"\n{'─' * _W}\n  步骤 3: HTTP 连通性 → {url}\n{'─' * _W}")
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            # 先用 GET 或空 POST 探测，看服务端是否响应
            t0 = time.time()
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": "test", "messages": []},  # 空请求，测试服务端是否活着
            )
            latency = (time.time() - t0) * 1000
            _ok(f"HTTP 响应: {resp.status_code} (延迟 {latency:.0f}ms)")
            if resp.status_code == 403:
                _fail("认证失败 (403)")
                try:
                    body = resp.json()
                    _info(f"  错误详情: {body.get('error', {}).get('message', resp.text[:200])}")
                except Exception:
                    _info(f"  响应体: {resp.text[:200]}")
                _info("可能原因:")
                _info("  1. API Key 无效或过期")
                _info("  2. API Key 无权限访问该模型")
                return False
            elif resp.status_code == 429:
                _fail("限流 (429)")
                _info("服务端可用，但当前被限流。请稍后重试。")
                return True  # 服务本身是通的
            elif resp.status_code >= 500:
                _fail(f"服务端错误 ({resp.status_code})")
                _info(f"  响应: {resp.text[:200]}")
                return False
            return True
    except httpx.ConnectError as e:
        _fail(f"HTTP 连接失败: {e}")
        _info("TCP 层可能通了但 HTTP 层握手失败")
        return False
    except httpx.ReadTimeout:
        _fail("HTTP 读取超时")
        return False
    except Exception as e:
        _fail(f"HTTP 请求异常: {type(e).__name__}: {e}")
        return False


# ─── 5. 完整 API 调用 ─────────────────────────────────────────────────────────

async def check_api_call(api_base: str, api_key: str, model: str) -> bool:
    import httpx

    print(f"\n{'─' * _W}\n  步骤 4: 完整 API 调用 → model={model}\n{'─' * _W}")
    try:
        json_data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "只输出一个字"},
                {"role": "user", "content": "你好"},
            ],
            "temperature": 0.0,
            "max_tokens": 10,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        t0 = time.time()
        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            resp = await client.post(api_base, headers=headers, json=json_data)
            latency = (time.time() - t0) * 1000

        if resp.status_code != 200:
            _fail(f"API 返回 {resp.status_code}")
            try:
                body = resp.json()
                err = body.get("error", {})
                _info(f"  error.type: {err.get('type', 'N/A')}")
                _info(f"  error.message: {err.get('message', resp.text[:200])}")
            except Exception:
                _info(f"  响应: {resp.text[:300]}")
            return False

        body = resp.json()
        choices = body.get("choices", [])
        if not choices:
            _fail("响应中无 choices")
            _info(f"  响应体: {body}")
            return False

        content = choices[0].get("message", {}).get("content", "")
        usage = body.get("usage", {})
        _ok(f"API 调用成功! (延迟 {latency:.0f}ms)")
        _ok(f"模型回复: {content}")
        _ok(f"Token 用量: prompt={usage.get('prompt_tokens', '?')}, "
            f"completion={usage.get('completion_tokens', '?')}")
        return True

    except httpx.ReadTimeout:
        _fail("API 调用超时 (>60s)")
        _info("可能原因:")
        _info("  1. 模型服务负载高")
        _info("  2. 网络延迟过大")
        _info("  3. 使用思考模型(如 t1)时推理时间较长，尝试简化 prompt")
        return False
    except httpx.ConnectError as e:
        _fail(f"连接失败: {e}")
        return False
    except Exception as e:
        _fail(f"异常: {type(e).__name__}: {e}")
        return False


# ─── 6. 同时测试 Venus ────────────────────────────────────────────────────────

async def check_venus() -> bool:
    print(f"\n{'─' * _W}\n  [对照] Venus API 连通性\n{'─' * _W}")
    try:
        from chatdb.utils.config import settings
        api_base = settings.llm.venus_api_base
        api_key = settings.llm.venus_api_key
        model = settings.llm.venus_model
    except Exception:
        _info("无法加载 Venus 配置，跳过")
        return False

    parsed = urlparse(api_base)
    host = parsed.hostname or ""
    port = parsed.port or 80

    if not check_dns(host):
        return False
    if not check_tcp(host, port):
        return False

    return await check_api_call(api_base, api_key, model)


# ─── 主入口 ───────────────────────────────────────────────────────────────────

async def main():
    print(f"{'=' * _W}\n  混元 API 连通性诊断\n{'=' * _W}")

    cfg = load_config()
    if not cfg:
        print("\n配置加载失败，无法继续。")
        return

    api_base = cfg["api_base"]
    api_key = cfg["api_key"]
    model = cfg["model"]

    _info(f"API Base: {api_base}")
    _info(f"Model:    {model}")
    _info(f"API Key:  {api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else f"API Key: {api_key}")

    parsed = urlparse(api_base)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    # 逐层检查
    results = {}
    results["dns"] = check_dns(host)
    results["tcp"] = check_tcp(host, port) if results["dns"] else False
    results["http"] = await check_http(api_base, api_key) if results["tcp"] else False
    results["api"] = await check_api_call(api_base, api_key, model) if results["http"] else False

    # 对照：同时测试 Venus
    results["venus"] = await check_venus()

    # 汇总
    print(f"\n{'=' * _W}\n  诊断汇总\n{'=' * _W}")
    labels = {
        "dns": "DNS 解析", "tcp": "TCP 连接",
        "http": "HTTP 请求", "api": "API 调用",
        "venus": "Venus 对照",
    }
    for k, v in results.items():
        icon = "✓" if v else "✗"
        print(f"  {icon} {labels[k]}")

    # 定位问题
    if all(results.values()):
        print(f"\n  ✓ 混元 API 一切正常！")
    elif not results["dns"]:
        print(f"\n  诊断结论: DNS 解析失败")
        print(f"  → 最可能原因: 不在腾讯内网环境，或 VPN 未连接")
        print(f"  → 解决方案: 连接内网 VPN 或在办公网络环境下运行")
    elif not results["tcp"]:
        print(f"\n  诊断结论: TCP 连接失败")
        print(f"  → 最可能原因: 防火墙阻断、服务未启动")
    elif not results["http"]:
        print(f"\n  诊断结论: HTTP 层失败")
        print(f"  → 检查 API Key 和服务端状态")
    elif not results["api"]:
        print(f"\n  诊断结论: API 调用失败（HTTP 层正常）")
        print(f"  → 检查模型名称 '{model}' 是否正确、API Key 权限")
        if results["venus"]:
            print(f"  → Venus 正常，建议暂时切换到 Venus: default_llm_provider = \"venus\"")

    print()


if __name__ == "__main__":
    asyncio.run(main())
