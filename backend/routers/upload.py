import os
import tempfile
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models.models import Document, Chunk
from backend.services.ingestion import process_pdf
from backend.services.embedding import generate_embedding

router = APIRouter()

@router.post("/api/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 1. Store Document Metadata
        db_doc = Document(filename=file.filename)
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)

        # 2. Process PDF into chunks
        chunks_data = process_pdf(tmp_path)
        
        # 3. Embed text and create Chunk records
        chunk_objects = []
        for data in chunks_data:
            text = data["text"]
            embedding = generate_embedding(text)
            
            chunk_obj = Chunk(
                document_id=db_doc.id,
                text_content=text,
                page_number=data["page_number"],
                embedding=embedding
            )
            chunk_objects.append(chunk_obj)
            
        # 4. Save to PostgreSQL
        db.add_all(chunk_objects)
        db.commit()

        # Clean up temp file
        os.remove(tmp_path)

        return {
            "status": "success",
            "message": f"Successfully processed {file.filename}",
            "document_id": db_doc.id,
            "chunks_created": len(chunk_objects)
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
