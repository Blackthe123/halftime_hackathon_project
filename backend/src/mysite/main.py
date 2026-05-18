import os
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy import select, delete as sa_delete

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Todo(Base):
    __tablename__ = "todos"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    content = Column(Text, nullable=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Dispose of engine on shutdown
    await engine.dispose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TodoItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str

    model_config = {"from_attributes": True}


class TodoItemCreate(BaseModel):
    content: str


@app.post("/todos", response_model=TodoItem)
async def create_todo(item: TodoItemCreate):
    new_todo = Todo(id=uuid4(), content=item.content)
    async with async_session() as session:
        session.add(new_todo)
        await session.commit()
        await session.refresh(new_todo)
    return new_todo


@app.get("/todos", response_model=list[TodoItem])
async def read_todos():
    async with async_session() as session:
        result = await session.execute(select(Todo))
        return result.scalars().all()


@app.delete("/todos/{todo_id}")
async def delete_todo(todo_id: UUID):
    async with async_session() as session:
        result = await session.execute(
            sa_delete(Todo).where(Todo.id == todo_id)
        )
        await session.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Todo not found")
    return {"message": "Todo deleted successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
