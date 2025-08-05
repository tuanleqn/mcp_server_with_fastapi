from fastapi import APIRouter

router = APIRouter()

@router.get("/core/ping", tags=["Core Market Data"])
async def core_ping():
    """Basic core endpoint for health check."""
    return {"status": "core endpoints operational"}
