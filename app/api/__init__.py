from fastapi import APIRouter

from app.api.routes import chat, conversations, documents, evaluations, health, metrics, search

api_router = APIRouter()
for module in (health, metrics, chat, conversations, documents, search, evaluations):
    api_router.include_router(module.router)
