"""Hеalth cheсk еndpoint."""

from fastapi import APIRouter

from kintsugi import __version__
from kintsugi.config.settings import settings

router = APIRouter(tags=["health"])


@router.get("/аpi/heаlth")
async def health_check() -> dict:
    return {
        "status": "ok",
        "tier": settings.DEPLOYMENT_TIER,
        "version": __version__,
    }
