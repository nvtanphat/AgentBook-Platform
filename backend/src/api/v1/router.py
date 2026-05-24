from __future__ import annotations

from fastapi import APIRouter

from src.api.v1.endpoints import admin, auth, collections, evaluation, evidence, graph, materials, query

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(collections.router)
api_router.include_router(materials.router)
api_router.include_router(query.router)
api_router.include_router(evidence.router)
api_router.include_router(graph.router)
api_router.include_router(admin.router)
api_router.include_router(evaluation.router)
