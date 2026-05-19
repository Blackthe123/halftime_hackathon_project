import os
import random
import string
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Column, ForeignKey, String, Text, Table, DateTime, Boolean, delete, PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, selectinload, sessionmaker
from sqlalchemy import select, func

# ── Config ────────────────────────────────────────────────────────────────────

DATABASE_URL = os.environ["DATABASE_URL"]
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ── Models ────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass

# Association table for group membership
group_members = Table(
    "group_members",
    Base.metadata,
    Column("user_id", PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("group_id", PG_UUID(as_uuid=True), ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True),
)

class User(Base):
    __tablename__ = "users"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)

    groups = relationship("Group", secondary=group_members, back_populates="members")

class Group(Base):
    __tablename__ = "groups"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False)
    invite_code = Column(String(8), unique=True, nullable=False, index=True)
    owner_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    members = relationship("User", secondary=group_members, back_populates="groups")
    messages = relationship("Message", back_populates="group", cascade="all, delete-orphan")
    timetable = relationship(
        "GroupTimetable", back_populates="group",
        uselist=False, cascade="all, delete-orphan"
    )

class Message(Base):
    __tablename__ = "messages"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    group_id = Column(PG_UUID(as_uuid=True), ForeignKey("groups.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    group = relationship("Group", back_populates="messages")
    user = relationship("User")

class UserGroupCourse(Base):
    """Stores which courses a user has selected for a specific group."""
    __tablename__ = "user_group_courses"

    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    group_id = Column(PG_UUID(as_uuid=True), ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True)
    course_code = Column(String(20), primary_key=True)

class GroupTimetable(Base):
    """Stores the latest computed optimised timetable for a group."""
    __tablename__ = "group_timetables"

    group_id = Column(PG_UUID(as_uuid=True), ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True)
    result_json = Column(Text, nullable=False)
    computed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    invalidated = Column(Boolean, nullable=False, default=False)

    group = relationship("Group", back_populates="timetable")

class EventVote(Base):
    """Stores which member voted for which campus event inside a specific group."""
    __tablename__ = "event_votes"

    group_id = Column(PG_UUID(as_uuid=True), ForeignKey("groups.id", ondelete="CASCADE"))
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    event_id = Column(String(100), nullable=False)
    
    __table_args__ = (PrimaryKeyConstraint('group_id', 'user_id', 'event_id'),)
    user = relationship("User")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan, title="InSync API")
ALLOWED_ORIGINS = {"http://localhost", "http://localhost:3000"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    origin = request.headers.get("origin", "")
    headers = {}
    if origin in ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=headers)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str: return pwd_context.hash(password)
def verify_password(plain: str, hashed: str) -> bool: return pwd_context.verify(plain, hashed)
def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
def generate_invite_code(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None: raise credentials_exception
    except JWTError:
        raise credentials_exception

    async with async_session() as session:
        result = await session.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()
        if user is None: raise credentials_exception
        return user

async def assert_member(session: AsyncSession, user_id: UUID, group_id: UUID) -> None:
    membership = await session.execute(
        select(group_members).where(group_members.c.user_id == user_id, group_members.c.group_id == group_id)
    )
    if not membership.first(): raise HTTPException(status_code=403, detail="Not a member of this group")

# ── Schemas ───────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=6)

class UserOut(BaseModel):
    id: UUID
    email: str
    display_name: str
    model_config = {"from_attributes": True}

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserOut

class GroupCreate(BaseModel): name: str = Field(min_length=1, max_length=100)
class GroupJoin(BaseModel): invite_code: str

class GroupOut(BaseModel):
    id: UUID
    name: str
    invite_code: str
    owner_id: UUID
    member_count: int
    model_config = {"from_attributes": True}

class GroupDetail(GroupOut): members: list[UserOut]
class MessageCreate(BaseModel): content: str = Field(min_length=1, max_length=2000)

class MessageOut(BaseModel):
    id: UUID
    group_id: UUID
    user_id: UUID
    display_name: str
    content: str
    created_at: datetime
    model_config = {"from_attributes": True}

class CourseSelection(BaseModel): course_codes: list[str] = Field(default_factory=list)
class MemberCoursesOut(BaseModel):
    user_id: UUID
    display_name: str
    course_codes: list[str]
class GroupCoursesOut(BaseModel):
    members: list[MemberCoursesOut]
    ready_count: int
    total_count: int
    is_ready: bool

class TimetableSave(BaseModel): result_json: str
class TimetableOut(BaseModel):
    group_id: UUID
    result_json: str
    computed_at: datetime
    invalidated: bool
    model_config = {"from_attributes": True}

# Voting Schemas
class EventVoteCreate(BaseModel):
    event_id: str = Field(min_length=1, max_length=100)

class EventVoteOut(BaseModel):
    event_id: str
    user_id: UUID
    display_name: str
    model_config = {"from_attributes": True}

# ── Auth Routes ───────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=Token, status_code=201)
async def register(body: UserRegister):
    async with async_session() as session:
        existing = await session.execute(select(User).where(User.email == body.email))
        if existing.scalar_one_or_none(): raise HTTPException(status_code=400, detail="Email already registered")
        user = User(id=uuid4(), email=body.email, display_name=body.display_name, hashed_password=hash_password(body.password))
        session.add(user)
        await session.commit()
        await session.refresh(user)
    token = create_access_token({"sub": str(user.id)})
    return Token(access_token=token, token_type="bearer", user=UserOut.model_validate(user))

@app.post("/auth/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    async with async_session() as session:
        result = await session.execute(select(User).where(User.email == form.username))
        user = result.scalar_one_or_none()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token({"sub": str(user.id)})
    return Token(access_token=token, token_type="bearer", user=UserOut.model_validate(user))

@app.get("/auth/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    return current_user

# ── Group Routes ──────────────────────────────────────────────────────────────

@app.get("/groups/me", response_model=list[GroupOut])
async def my_groups(current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        result = await session.execute(
            select(Group).options(selectinload(Group.members))
            .join(group_members, Group.id == group_members.c.group_id)
            .where(group_members.c.user_id == current_user.id)
        )
        return [GroupOut(id=g.id, name=g.name, invite_code=g.invite_code, owner_id=g.owner_id, member_count=len(g.members)) for g in result.scalars().all()]

@app.post("/groups", response_model=GroupOut, status_code=201)
async def create_group(body: GroupCreate, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        while True:
            code = generate_invite_code()
            if not (await session.execute(select(Group).where(Group.invite_code == code))).scalar_one_or_none(): break
        group = Group(id=uuid4(), name=body.name, invite_code=code, owner_id=current_user.id)
        session.add(group)
        await session.flush()
        await session.execute(group_members.insert().values(user_id=current_user.id, group_id=group.id))
        await session.commit()
        result = await session.execute(select(Group).options(selectinload(Group.members)).where(Group.id == group.id))
        group = result.scalar_one()
    return GroupOut(id=group.id, name=group.name, invite_code=group.invite_code, owner_id=group.owner_id, member_count=len(group.members))

@app.post("/groups/join", response_model=GroupOut)
async def join_group(body: GroupJoin, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        result = await session.execute(select(Group).options(selectinload(Group.members)).where(Group.invite_code == body.invite_code.upper()))
        group = result.scalar_one_or_none()
        if not group: raise HTTPException(status_code=404, detail="Invalid invite code")
        if (await session.execute(select(group_members).where(group_members.c.user_id == current_user.id, group_members.c.group_id == group.id))).first():
            raise HTTPException(status_code=400, detail="Already a member of this group")
        await session.execute(group_members.insert().values(user_id=current_user.id, group_id=group.id))
        await session.commit()
        result = await session.execute(select(Group).options(selectinload(Group.members)).where(Group.id == group.id))
        group = result.scalar_one()
    return GroupOut(id=group.id, name=group.name, invite_code=group.invite_code, owner_id=group.owner_id, member_count=len(group.members))

@app.get("/groups/{group_id}", response_model=GroupDetail)
async def get_group(group_id: UUID, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        result = await session.execute(select(Group).options(selectinload(Group.members)).where(Group.id == group_id))
        group = result.scalar_one_or_none()
        if not group: raise HTTPException(status_code=404, detail="Group not found")
        await assert_member(session, current_user.id, group_id)
        members = [UserOut.model_validate(m) for m in group.members]
    return GroupDetail(id=group.id, name=group.name, invite_code=group.invite_code, owner_id=group.owner_id, member_count=len(members), members=members)

@app.delete("/groups/{group_id}/leave", status_code=204)
async def leave_group(group_id: UUID, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        group = (await session.execute(select(Group).where(Group.id == group_id))).scalar_one_or_none()
        if not group: raise HTTPException(status_code=404, detail="Group not found")
        await assert_member(session, current_user.id, group_id)
        if group.owner_id == current_user.id: raise HTTPException(status_code=400, detail="Owner must transfer ownership or delete group")
        await session.execute(group_members.delete().where(group_members.c.user_id == current_user.id, group_members.c.group_id == group_id))
        await session.commit()

@app.delete("/groups/{group_id}", status_code=204)
async def delete_group(group_id: UUID, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        group = (await session.execute(select(Group).where(Group.id == group_id))).scalar_one_or_none()
        if not group: raise HTTPException(status_code=404, detail="Group not found")
        if group.owner_id != current_user.id: raise HTTPException(status_code=403, detail="Only owner can delete group")
        await session.delete(group)
        await session.commit()

@app.delete("/groups/{group_id}/members/{user_id}", status_code=204)
async def remove_member(group_id: UUID, user_id: UUID, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        group = (await session.execute(select(Group).where(Group.id == group_id))).scalar_one_or_none()
        if not group: raise HTTPException(status_code=404, detail="Group not found")
        if group.owner_id != current_user.id: raise HTTPException(status_code=403, detail="Only owner can remove members")
        if group.owner_id == user_id: raise HTTPException(status_code=400, detail="Cannot remove the owner")
        res = await session.execute(group_members.delete().where(group_members.c.user_id == user_id, group_members.c.group_id == group_id))
        if res.rowcount == 0: raise HTTPException(status_code=404, detail="Member not found")
        await session.execute(delete(UserGroupCourse).where(UserGroupCourse.user_id == user_id, UserGroupCourse.group_id == group_id))
        await session.execute(delete(EventVote).where(EventVote.user_id == user_id, EventVote.group_id == group_id))
        tt = (await session.execute(select(GroupTimetable).where(GroupTimetable.group_id == group_id))).scalar_one_or_none()
        if tt: tt.invalidated = True
        await session.commit()

# ── Chat, Courses & Timetable Routes ──────────────────────────────────────────

@app.get("/groups/{group_id}/messages", response_model=list[MessageOut])
async def get_messages(group_id: UUID, limit: int = 50, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        await assert_member(session, current_user.id, group_id)
        messages = (await session.execute(select(Message).options(selectinload(Message.user)).where(Message.group_id == group_id).order_by(Message.created_at.desc()).limit(limit))).scalars().all()
    return [MessageOut(id=m.id, group_id=m.group_id, user_id=m.user_id, display_name=m.user.display_name, content=m.content, created_at=m.created_at) for m in reversed(messages)]

@app.post("/groups/{group_id}/messages", response_model=MessageOut, status_code=201)
async def send_message(group_id: UUID, body: MessageCreate, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        await assert_member(session, current_user.id, group_id)
        message = Message(id=uuid4(), group_id=group_id, user_id=current_user.id, content=body.content)
        session.add(message)
        await session.commit()
        message = (await session.execute(select(Message).options(selectinload(Message.user)).where(Message.id == message.id))).scalar_one()
    return MessageOut(id=message.id, group_id=message.group_id, user_id=message.user_id, display_name=message.user.display_name, content=message.content, created_at=message.created_at)

@app.get("/groups/{group_id}/courses", response_model=GroupCoursesOut)
async def get_group_courses(group_id: UUID, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        group = (await session.execute(select(Group).options(selectinload(Group.members)).where(Group.id == group_id))).scalar_one_or_none()
        if not group: raise HTTPException(status_code=404)
        await assert_member(session, current_user.id, group_id)
        all_selections = (await session.execute(select(UserGroupCourse).where(UserGroupCourse.group_id == group_id))).scalars().all()
        courses_map = {}
        for sel in all_selections: courses_map.setdefault(sel.user_id, []).append(sel.course_code)
        members_out = [MemberCoursesOut(user_id=m.id, display_name=m.display_name, course_codes=sorted(courses_map.get(m.id, []))) for m in group.members]
        ready_count = sum(1 for m in members_out if m.course_codes)
    return GroupCoursesOut(members=members_out, ready_count=ready_count, total_count=len(members_out), is_ready=ready_count >= 2)

@app.put("/groups/{group_id}/my-courses", status_code=204)
async def update_my_courses(group_id: UUID, body: CourseSelection, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        await assert_member(session, current_user.id, group_id)
        await session.execute(delete(UserGroupCourse).where(UserGroupCourse.user_id == current_user.id, UserGroupCourse.group_id == group_id))
        seen = set()
        for code in body.course_codes:
            n = code.strip().upper()
            if n and n not in seen:
                seen.add(n)
                session.add(UserGroupCourse(user_id=current_user.id, group_id=group_id, course_code=n))
        tt = (await session.execute(select(GroupTimetable).where(GroupTimetable.group_id == group_id))).scalar_one_or_none()
        if tt: tt.invalidated = True
        await session.commit()

@app.get("/groups/{group_id}/timetable", response_model=TimetableOut)
async def get_timetable(group_id: UUID, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        await assert_member(session, current_user.id, group_id)
        tt = (await session.execute(select(GroupTimetable).where(GroupTimetable.group_id == group_id))).scalar_one_or_none()
        if not tt: raise HTTPException(status_code=404)
    return TimetableOut.model_validate(tt)

@app.post("/groups/{group_id}/timetable", response_model=TimetableOut, status_code=201)
async def save_timetable(group_id: UUID, body: TimetableSave, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        await assert_member(session, current_user.id, group_id)
        tt = (await session.execute(select(GroupTimetable).where(GroupTimetable.group_id == group_id))).scalar_one_or_none()
        now = datetime.now(timezone.utc)
        if tt:
            tt.result_json, tt.computed_at, tt.invalidated = body.result_json, now, False
        else:
            tt = GroupTimetable(group_id=group_id, result_json=body.result_json, computed_at=now, invalidated=False)
            session.add(tt)
        await session.commit()
        await session.refresh(tt)
    return TimetableOut.model_validate(tt)

# ── Event Voting Routes ───────────────────────────────────────────────────────

@app.get("/groups/{group_id}/votes", response_model=list[EventVoteOut])
async def get_event_votes(group_id: UUID, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        await assert_member(session, current_user.id, group_id)
        votes = (await session.execute(
            select(EventVote).options(selectinload(EventVote.user)).where(EventVote.group_id == group_id)
        )).scalars().all()
    return [EventVoteOut(event_id=v.event_id, user_id=v.user_id, display_name=v.user.display_name) for v in votes]

@app.post("/groups/{group_id}/votes", status_code=201)
async def add_event_vote(group_id: UUID, body: EventVoteCreate, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        await assert_member(session, current_user.id, group_id)
        existing = (await session.execute(
            select(EventVote).where(EventVote.group_id==group_id, EventVote.user_id==current_user.id, EventVote.event_id==body.event_id)
        )).first()
        if not existing:
            session.add(EventVote(group_id=group_id, user_id=current_user.id, event_id=body.event_id))
            await session.commit()

@app.delete("/groups/{group_id}/votes/{event_id}", status_code=204)
async def remove_event_vote(group_id: UUID, event_id: str, current_user: User = Depends(get_current_user)):
    async with async_session() as session:
        await assert_member(session, current_user.id, group_id)
        await session.execute(
            delete(EventVote).where(EventVote.group_id==group_id, EventVote.user_id==current_user.id, EventVote.event_id==event_id)
        )
        await session.commit()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)