import logging
import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from api.auth import verify_api_key
from api.dependencies import get_pipeline
from api.limiter import limiter
from api.sanitization import sanitize_filename
from orchestration.pipeline import KnowledgeGraphPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"], dependencies=[Depends(verify_api_key)])

_MAX_PDF_BYTES = 50 * 1024 * 1024   # 50 MB
_MAX_CSV_BYTES = 10 * 1024 * 1024   # 10 MB


class IngestResponse(BaseModel):
    status: str
    chunks_processed: int
    entities_extracted: int


@router.post("/pdf", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest_pdf(
    request: Request,
    file: UploadFile = File(...),
    pipeline: KnowledgeGraphPipeline = Depends(get_pipeline),
) -> IngestResponse:
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Validate and sanitize filename
    try:
        sanitize_filename(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {str(e)}")

    content = await file.read()

    if len(content) > _MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds maximum size of {_MAX_PDF_BYTES // (1024 * 1024)} MB",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.ingest(pdf_path=tmp_path)

        chunks_processed = len(result.get("chunks", []))
        knowledge_graphs = result.get("knowledge_graphs", [])
        entities_extracted = sum(len(kg.entities) for kg in knowledge_graphs)

        logger.info(
            f"Ingested PDF: {file.filename} — "
            f"{chunks_processed} chunks, {entities_extracted} entities"
        )
        return IngestResponse(
            status="success",
            chunks_processed=chunks_processed,
            entities_extracted=entities_extracted,
        )
    except Exception as e:
        logger.error(f"Error ingesting PDF: {e}")
        raise HTTPException(status_code=500, detail="PDF ingestion failed")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/csv", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest_csv(
    request: Request,
    file: UploadFile = File(...),
    pipeline: KnowledgeGraphPipeline = Depends(get_pipeline),
) -> IngestResponse:
  
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Validate and sanitize filename
    try:
        sanitize_filename(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {str(e)}")

    content = await file.read()

    if len(content) > _MAX_CSV_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds maximum size of {_MAX_CSV_BYTES // (1024 * 1024)} MB",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.ingest(csv_path=tmp_path)

        triples = result.get("triples", [])
        chunks_processed = len(triples) if isinstance(triples, list) else 0
        graph_stats = result.get("graph_stats", {})
        entities_extracted = graph_stats.get("node_count", 0)

        logger.info(
            f"Ingested CSV: {file.filename} — "
            f"{chunks_processed} records, {entities_extracted} entities"
        )
        return IngestResponse(
            status="success",
            chunks_processed=chunks_processed,
            entities_extracted=entities_extracted,
        )
    except Exception as e:
        logger.error(f"Error ingesting CSV: {e}")
        raise HTTPException(status_code=500, detail="CSV ingestion failed")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
