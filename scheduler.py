from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import os
from dotenv import load_dotenv

# --- IMPORTANT FIX: Explicitly import datetime for use in scheduler job ---
from datetime import datetime 

from sqlalchemy.orm import Session 

# Import database models and helper functions
from database import get_db, ScheduledPost, ReviewQueue, update_post_status 
from ai_pipelines import spam_filter, sentiment_analysis

# --- Load Config ---
load_dotenv()
SCHEDULER_INTERVAL_POSTS = int(os.environ.get("SCHEDULER_INTERVAL_POSTS", 60)) # seconds
SCHEDULER_INTERVAL_MODERATE = int(os.environ.get("SCHEDULER_INTERVAL_MODERATE", 5)) # minutes

# --- Placeholder Functions for Social Media API Interaction (Objective 2/4) ---

def social_media_api_send_post(content: str, platform: str):
    """Mocks sending a post via the social media API library."""
    print(f"--- POST SENT: [{platform}] '{content[:30]}...' ---")
    return True # Return success status

def social_media_api_fetch_comments():
    """Mocks fetching new comments for moderation."""
    print("--- FETCHING NEW COMMENTS FOR MODERATION ---")
    # Mock data to simulate fetching:
    return [
        {"id": 101, "text": "This product is beyond horrible, I demand a refund.", "source": "X"},
        {"id": 102, "text": "Click this link now for free money and a guaranteed win!", "source": "Facebook"},
        {"id": 103, "text": "That's a nice design. I like it.", "source": "Instagram"}
    ]

def social_media_api_delete_comment(comment_id):
    """Mocks deleting/hiding a comment."""
    print(f"--- COMMENT DELETED/HIDDEN: ID {comment_id} ---")

# --- Scheduler Tasks ---

def check_for_scheduled_posts():
    """Objective 2: Checks DB for posts due to be sent."""
    # get_db is a generator, so we iterate to get a session
    for db in get_db():
        now = datetime.utcnow()
        posts = db.query(ScheduledPost).filter(
            ScheduledPost.schedule_time <= now,
            ScheduledPost.is_sent == False
        ).all()
        
        for post in posts:
            try:
                # Use the local mock function defined in this file
                success = social_media_api_send_post(post.content, post.platform)
                if success:
                    # Use the helper function imported from database.py
                    update_post_status(db, post.id, "SENT")
                    print(f"✅ Posted scheduled item: ID {post.id}")
            except Exception as e:
                # Use the helper function to mark as FAILED
                update_post_status(db, post.id, "FAILED")
                print(f"❌ Failed to send post ID {post.id}: {e}")

        # Note: update_post_status calls db.commit(), but we commit again to be safe
        db.commit()


def monitor_and_moderate_content():
    """Objective 4: Triage new content using AI models."""
    new_comments = social_media_api_fetch_comments()
    
    for db in get_db():
        for comment in new_comments:
            text = comment['text']
            
            # Step 1: Spam Filter
            spam_result = spam_filter.predict(text)
            if spam_result == "Spam":
                social_media_api_delete_comment(comment['id'])
                continue 

            # Step 2: Sentiment Review Check (High Priority)
            sentiment_result = sentiment_analysis.predict(text)
            if sentiment_result in ["Very Negative", "Negative"]:
                # Add to ReviewQueue for human action
                review_item = ReviewQueue(
                    message_text=text,
                    source_platform=comment['source'],
                    trigger_reason=f"{sentiment_result} Sentiment",
                    url_to_comment=f"http://link.to.comment/{comment['id']}" # Mock URL
                )
                db.add(review_item)
                print(f"🚨 Added to Review Queue ({sentiment_result}): {text[:50]}...")
        
        db.commit()

# --- Scheduler Setup ---

scheduler = BackgroundScheduler()

# Add job 1: Scheduled Posts
scheduler.add_job(
    check_for_scheduled_posts, 
    trigger=IntervalTrigger(seconds=SCHEDULER_INTERVAL_POSTS),
    id='scheduled_poster',
    name='Check and Send Scheduled Posts',
    replace_existing=True
)

# Add job 2: Moderation Check
scheduler.add_job(
    monitor_and_moderate_content, 
    trigger=IntervalTrigger(minutes=SCHEDULER_INTERVAL_MODERATE),
    id='moderation_checker',
    name='Monitor and Moderate New Content',
    replace_existing=True
)

def start_scheduler():
    """Starts the background scheduler."""
    if not scheduler.running:
        scheduler.start()
        print(f"⏱️ Scheduler started: Posts check every {SCHEDULER_INTERVAL_POSTS}s, Moderation check every {SCHEDULER_INTERVAL_MODERATE}m.")