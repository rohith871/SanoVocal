"""
SanoVocal Production FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_health import router as health_router
from app.api.routes_chat import router as chat_router
from app.api.routes_voice import router as voice_router
from app.storage.database import init_db

app = FastAPI(
    title="SanoVocal API",
    description="Voice-first, safety-grounded Indian health & welfare scheme navigator backend.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(voice_router)

@app.on_event("startup")
def on_startup():
    print("[SanoVocal API] Initializing databases and resources...")
    init_db()
    print("[SanoVocal API] Ready.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)
