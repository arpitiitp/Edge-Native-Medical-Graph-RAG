from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database import engine, Base
from backend.routers import upload, graph, query

# Create the database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Graph RAG Edge-Native API", version="1.0.0")

# CORS setup for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(graph.router)
app.include_router(query.router)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Graph RAG API is running"}

