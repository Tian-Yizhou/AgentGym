from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .environment import EnvError, ServerConfig, TextWorldExpressServer
from .model import CloseRequestBody, ResetRequestBody, StepRequestBody


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The JVMs start when the app starts (not at import time), so importing this package is cheap.
    app.state.server = TextWorldExpressServer(ServerConfig.from_env())
    try:
        yield
    finally:
        app.state.server.shutdown()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(EnvError)
async def env_error_handler(request: Request, exc: EnvError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


def _server(request: Request) -> TextWorldExpressServer:
    return request.app.state.server


@app.get("/")
def hello():
    return "This is environment TextWorldExpress."


@app.post("/create")
def create(request: Request):
    return _server(request).create()


@app.post("/reset")
def reset(body: ResetRequestBody, request: Request):
    return _server(request).reset(body.id, body.data_idx)


@app.post("/step")
def step(body: StepRequestBody, request: Request):
    return _server(request).step(body.id, body.action)


@app.get("/observation")
def get_observation(id: int, request: Request):
    return _server(request).observation(id)


@app.get("/detail")
def get_detail(id: int, request: Request):
    return _server(request).detail(id)


@app.get("/gold_path")
def get_gold_path(id: int, request: Request):
    return _server(request).gold_path(id)


@app.post("/close")
def close(body: CloseRequestBody, request: Request):
    return _server(request).close(body.id)
