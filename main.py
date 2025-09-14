from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID, uuid4

from dotenv import load_dotenv
load_dotenv()  # read .env into environment early

from fastapi import Depends, FastAPI, HTTPException, Query, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field as PydField
from sqlalchemy import func
from sqlmodel import Field, SQLModel, Session, create_engine, select

# -----------------------------------------------------------------------------
# Database configuration
# -----------------------------------------------------------------------------
# For SQLAlchemy + Psycopg3, use "postgresql+psycopg://...?...sslmode=require"
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set. Put it in your environment or .env")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,  # avoid stale connections after free-tier sleeps
)

def create_db_and_tables() -> None:
    # Creates all declared tables if they don't exist yet
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class TaskBase(SQLModel):
    title: str = Field(index=True, min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=500)
    due_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(days=1)
    )

class Task(TaskBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    completed: bool = Field(default=False, index=True)

class TaskCreate(SQLModel):
    title: str
    description: Optional[str] = None
    due_at: Optional[datetime] = None

class TaskUpdate(SQLModel):
    title: Optional[str] = None
    description: Optional[str] = None
    due_at: Optional[datetime] = None
    completed: Optional[bool] = None

class PaginatedTasks(BaseModel):
    total: int = PydField(ge=0)
    items: List[Task]

# -----------------------------------------------------------------------------
# Health router
# -----------------------------------------------------------------------------
health = APIRouter(prefix="/health", tags=["health"])

@health.get("/db")
def health_db(session: Session = Depends(get_session)):
    total = session.exec(select(func.count()).select_from(Task)).one()
    return {"database": "up", "tasks_count": total}

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Todo API (Postgres + SQLModel)",
    description="A minimal CRUD API using FastAPI + SQLModel + Postgres.",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health)

# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "Todo API (Postgres) running."}

@app.get("/tasks", response_model=PaginatedTasks, tags=["tasks"])
def list_tasks(
    completed: Optional[bool] = Query(default=None, description="Filter by completion"),
    q: Optional[str] = Query(default=None, description="Search in title/description"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
):
    # Base query with filters
    stmt = select(Task)
    count_stmt = select(func.count()).select_from(Task)

    if completed is not None:
        stmt = stmt.where(Task.completed == completed)
        count_stmt = count_stmt.where(Task.completed == completed)

    if q:
        pattern = f"%{q}%"
        cond = (Task.title.ilike(pattern)) | (Task.description.ilike(pattern))
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    total = session.exec(count_stmt).one()
    items = session.exec(stmt.offset(offset).limit(limit)).all()
    return PaginatedTasks(total=total, items=items)

@app.get("/tasks/{task_id}", response_model=Task, tags=["tasks"])
def get_task(task_id: UUID, session: Session = Depends(get_session)):
    task = session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/tasks", response_model=Task, status_code=status.HTTP_201_CREATED, tags=["tasks"])
def create_task(payload: TaskCreate, session: Session = Depends(get_session)):
    task = Task(**payload.model_dump(), id=uuid4(), completed=False)
    session.add(task)
    session.commit()
    session.refresh(task)
    return task

@app.put("/tasks/{task_id}", response_model=Task, tags=["tasks"])
def update_task(task_id: UUID, payload: TaskUpdate, session: Session = Depends(get_session)):
    task = session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(task, k, v)

    session.add(task)
    session.commit()
    session.refresh(task)
    return task

@app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["tasks"])
def delete_task(task_id: UUID, session: Session = Depends(get_session)):
    task = session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    session.delete(task)
    session.commit()
    return None
