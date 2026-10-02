"""Single-operator access protection; not multi-tenant authorization."""
import hashlib
import ipaddress
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import Field, SecretStr
from starlette.middleware.trustedhost import TrustedHostMiddleware

from valuationagent.schemas.models import ApiModel


COOKIE = "valuation_operator"
SESSION_SECONDS = 8 * 60 * 60


def is_loopback(host):
    if host in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def access_token():
    token = os.getenv("VALUATION_ACCESS_TOKEN", "")
    if token and (len(token) < 32 or len(token) > 512 or token.strip() != token):
        raise ValueError("VALUATION_ACCESS_TOKEN must be 32..512 characters without surrounding whitespace")
    return token


def validate_bind(host):
    if not is_loopback(host) and not access_token():
        raise ValueError("非本机监听必须配置VALUATION_ACCESS_TOKEN（至少32字符）及HTTPS反向代理；当前不是多租户服务。")


class AccessInput(ApiModel):
    token: SecretStr = Field(min_length=1, max_length=512)


def register_access(app, origins):
    token = access_token()
    sessions, attempts = {}, defaultdict(deque)
    lock = threading.Lock()
    hosts = [item.strip() for item in os.getenv("VALUATION_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",") if item.strip()]
    if not hosts or "*" in hosts:
        raise ValueError("VALUATION_ALLOWED_HOSTS must contain explicit host names, not '*'")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts, www_redirect=False)

    def client_host(request):
        return request.client.host if request.client else ""

    def authenticated(request):
        bearer = request.headers.get("authorization", "")
        if bearer.startswith("Bearer ") and secrets.compare_digest(bearer[7:].encode(), token.encode()):
            return True
        cookie = request.cookies.get(COOKIE, "")
        if not cookie or len(cookie) > 128:
            return False
        digest = hashlib.sha256(cookie.encode()).hexdigest()
        with lock:
            return sessions.get(digest, 0) > time.monotonic()

    def safe_origin(request):
        origin = request.headers.get("origin")
        if not origin:
            return True
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        return origin in origins or (parsed.scheme == request.url.scheme and parsed.netloc == request.headers.get("host")
                                     and not parsed.path and not parsed.query and not parsed.fragment)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        local = is_loopback(client_host(request))
        path = request.url.path
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not safe_origin(request):
            response = JSONResponse({"detail": "拒绝跨站写入请求"}, status_code=403)
        elif path != "/health" and not token and not local:
            response = JSONResponse({"detail": "未配置访问凭证，仅允许本机访问"}, status_code=403)
        elif path != "/health" and token and not local and request.url.scheme != "https":
            response = JSONResponse({"detail": "远程访问必须通过HTTPS"}, status_code=403)
        elif token and path not in {"/health", "/api/access/login"} and not authenticated(request):
            if path == "/" and request.method == "GET" and "text/html" in request.headers.get("accept", ""):
                response = login_page()
            else:
                response = JSONResponse({"detail": "需要工作区访问凭证；请在首页登录"}, status_code=401)
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if response.status_code >= 400 or path.startswith("/api/") or path == "/":
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/access/login")
    def login(request: Request, body: AccessInput):
        if not token:
            raise HTTPException(400, "本机模式未启用访问凭证")
        now = time.monotonic()
        host = client_host(request)
        with lock:
            for address in list(attempts):
                while attempts[address] and attempts[address][0] < now - 60:
                    attempts[address].popleft()
                if not attempts[address]:
                    del attempts[address]
            if len(attempts) >= 1000 and host not in attempts or len(attempts[host]) >= 5:
                raise HTTPException(429, "登录过于频繁，请稍后重试", headers={"Retry-After": "60"})
            attempts[host].append(now)
            if not secrets.compare_digest(body.token.get_secret_value().encode(), token.encode()):
                raise HTTPException(401, "访问凭证无效")
            for key in [key for key, expiry in sessions.items() if expiry <= now]:
                del sessions[key]
            if len(sessions) >= 100:
                raise HTTPException(429, "活动登录会话已达上限")
            cookie = secrets.token_urlsafe(32)
            sessions[hashlib.sha256(cookie.encode()).hexdigest()] = now + SESSION_SECONDS
        response = JSONResponse({"status": "authenticated", "mode": "single_operator", "expires_in": SESSION_SECONDS})
        response.set_cookie(COOKIE, cookie, max_age=SESSION_SECONDS, httponly=True,
                            secure=request.url.scheme == "https", samesite="strict", path="/")
        return response

    @app.post("/api/access/logout", status_code=204)
    def logout(request: Request):
        digest = hashlib.sha256(request.cookies.get(COOKIE, "").encode()).hexdigest()
        with lock:
            sessions.pop(digest, None)
        response = Response(status_code=204)
        response.delete_cookie(COOKIE, path="/", httponly=True, samesite="strict")
        return response

    @app.get("/api/access/status")
    def status():
        return {"authentication_required": bool(token), "mode": "single_operator" if token else "local_only"}


def login_page():
    nonce = secrets.token_urlsafe(18)
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ValuationAgent · 登录</title><style nonce="NONCE">
body{margin:0;background:#101b2a;color:#e9edf5;font:16px system-ui;display:grid;place-items:center;min-height:100vh}
main{max-width:420px;padding:36px}h1{font-size:28px}p{line-height:1.7;color:#b7c6dc}input,button{box-sizing:border-box;width:100%;padding:13px;border-radius:8px;margin-top:14px;font:inherit}
input{background:#17273a;color:#fff;border:1px solid #637b97}button{background:#5ad1c3;color:#10232a;border:0;cursor:pointer}#status{min-height:28px;color:#ffbb9e}
</style><main><h1>ValuationAgent</h1><p>输入部署管理员提供的工作区访问凭证。不是模型 API Key。此部署为单操作者空间，登录者共享全部研究资料。</p>
<form id="login"><label for="token">访问凭证</label><input id="token" type="password" autocomplete="current-password" maxlength="512" required><button>进入工作区</button></form><p id="status" role="status"></p></main>
<script nonce="NONCE">document.querySelector('#login').addEventListener('submit',async event=>{event.preventDefault();const input=document.querySelector('#token');const button=document.querySelector('button');const token=input.value;input.value='';button.disabled=true;try{const result=await fetch('/api/access/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});if(!result.ok)throw new Error(result.status===429?'尝试过多，请稍后重试':'登录失败，请检查访问凭证');location.replace('/')}catch(error){document.querySelector('#status').textContent=error.message}finally{button.disabled=false}})</script></html>"""
    return HTMLResponse(html.replace("NONCE", nonce), status_code=401, headers={
        "Content-Security-Policy": f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'; connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
        "Cache-Control": "no-store"})
