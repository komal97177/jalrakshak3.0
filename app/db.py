import os, datetime as dt
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

url = os.getenv("DATABASE_URL", "sqlite:///./jalrakshak.db")
if url.startswith("postgres://"): url = url.replace("postgres://", "postgresql+psycopg2://", 1)
elif url.startswith("postgresql://"): url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
engine = create_engine(url, pool_pre_ping=True, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
Session = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()
now = lambda: dt.datetime.utcnow()


class User(Base):            # privacy: email stored as keyed hash (lookup) + encrypted copy (only for sending alerts)
    __tablename__ = "users"
    id = Column(Integer, primary_key=True); email_hash = Column(String(64), unique=True, index=True)
    email_enc = Column(Text); name = Column(String(80)); block_id = Column(String(8)); lang = Column(String(2), default="en")
    subscribed = Column(Boolean, default=True); consent_at = Column(DateTime, default=now); created_at = Column(DateTime, default=now)
    last_alert_at = Column(DateTime, nullable=True)


class Otp(Base):
    __tablename__ = "otps"
    id = Column(Integer, primary_key=True); email_hash = Column(String(64), index=True); code_hash = Column(String(64))
    expires = Column(DateTime); attempts = Column(Integer, default=0); pending_name = Column(String(80), nullable=True)


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True); text = Column(Text); place = Column(String(40)); lat = Column(Float); lon = Column(Float)
    flood_prob = Column(Float); severity = Column(String(10)); needs = Column(String(60), default=""); lang = Column(String(2))
    score = Column(Float); status = Column(String(12)); dups = Column(Integer, default=0); scenario = Column(Boolean, default=False)
    created_at = Column(DateTime, default=now)


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True); level = Column(String(6)); message = Column(Text); recipients = Column(Integer, default=0)
    mode = Column(String(10)); created_at = Column(DateTime, default=now)


class Snapshot(Base):        # data collection: every live fetch is stored -> future training data (rain + river + risk)
    __tablename__ = "snapshots"
    id = Column(Integer, primary_key=True); ts = Column(DateTime, default=now); rain24_mean = Column(Float); rain24_max = Column(Float)
    river_index = Column(Float); peak_risk = Column(Float); level = Column(String(6))


class Audit(Base):
    __tablename__ = "audit"
    id = Column(Integer, primary_key=True); ts = Column(DateTime, default=now); actor = Column(String(16)); action = Column(String(60))


def init(): Base.metadata.create_all(engine)
