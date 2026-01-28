from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables (from the .env file)
load_dotenv()
SQLALCHEMY_DATABASE_URL = os.environ.get("SQLALCHEMY_DATABASE_URL", "sqlite:///./social_media_ai.db")

# Setup SQLAlchemy
Engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=Engine)
Base = declarative_base()

# --- Database Models ---

class ScheduledPost(Base):
    """Model for a post scheduled for future publication."""
    __tablename__ = "scheduled_posts"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    schedule_time = Column(DateTime, nullable=False)
    platform = Column(String, nullable=False)
    status = Column(String, default="PENDING") # Status: PENDING, SENT, FAILED
    is_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ReviewQueue(Base):
    """Model for content that requires manual human review."""
    __tablename__ = "review_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    message_text = Column(Text, nullable=False)
    source_platform = Column(String, nullable=False)
    trigger_reason = Column(String, nullable=False) # e.g., "Very Negative Sentiment", "Suspicious Link"
    url_to_comment = Column(String) # Mock URL to the original content
    created_at = Column(DateTime, default=datetime.utcnow)

# --- Helper Functions ---

def init_db():
    """Creates all defined tables in the database."""
    print("Database Initialized: Creating tables...")
    Base.metadata.create_all(bind=Engine)

def get_db():
    """
    Provides a database session for FastAPI dependencies.
    Crucially, this uses the try/finally block to ensure the session is always closed.
    """
    db = SessionLocal()
    try:
        # Yield the session to the FastAPI route handler
        yield db
    finally:
        # Ensure the session is closed after the request is processed
        db.close()

def update_post_status(db: Session, post_id: int, status: str):
    """Updates the status of a ScheduledPost."""
    post = db.query(ScheduledPost).filter(ScheduledPost.id == post_id).first()
    if post:
        post.status = status
        # If the status is SENT, we also set is_sent to True
        if status == "SENT":
            post.is_sent = True
        db.commit()
        return True
    return False

def add_to_review_queue(db: Session, message_text: str, source_platform: str, trigger_reason: str, url_to_comment: str):
    """
    Adds a new item to the manual review queue.
    """
    review_item = ReviewQueue(
        message_text=message_text,
        source_platform=source_platform,
        trigger_reason=trigger_reason,
        url_to_comment=url_to_comment
    )
    db.add(review_item)
    db.commit()
    db.refresh(review_item)
    return review_item