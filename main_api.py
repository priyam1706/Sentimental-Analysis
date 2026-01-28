# # # import time
# # # import asyncio
# # # import json
# # # import logging
# # # import random
# # # from typing import List, Optional, Union
# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from pydantic import BaseModel, Field

# # # # NEW REQUIREMENT: We need 'requests' to talk to the local Ollama API
# # # import requests 

# # # # Attempt to import transformers pipeline, handle failure if not installed
# # # try:
# # #     from transformers import pipeline
# # # except ImportError:
# # #     pipeline = None
# # #     logging.warning("⚠️ 'transformers' library not found. Moderation will use the MOCK pipeline.")

# # # # --- Configuration ---

# # # # Configure logging to show INFO messages
# # # logging.basicConfig(level=logging.INFO, format='INFO: %(message)s')

# # # # --- Global State and Initialization ---

# # # # --- IMPORTANT LLM CONFIGURATION ---
# # # OLLAMA_MODEL = "mistral" # CHANGE THIS if you pulled a different model (e.g., 'gemma:2b')
# # # OLLAMA_API_URL = "http://localhost:11434/api/generate"
# # # # ---

# # # app = FastAPI(title="Social Media AI Backend (OLLAMA INTEGRATION)", version="0.1.0")

# # # # 1. AI Model Initialization (Moderation Model - Updated Mock for precision)
# # # sentiment_pipeline = None
# # # if pipeline:
# # #     try:
# # #         # Note: We stick to distilbert-base-uncased-finetuned-sst-2-english which detects POSITIVE/NEGATIVE
# # #         # We use the score and label to infer the "Very" sentiments later.
# # #         sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# # #         logging.info("✅ DistilBERT Sentiment Model Loaded for Moderation.")
# # #     except Exception as e:
# # #         logging.error(f"❌ Failed to load AI model: {e}")
# # #         pipeline = None 
        
# # # if not pipeline:
# # #     # MOCK implementation for 5-class sentiment and spam detection (HIGH PRECISION MOCK)
# # #     def mock_sentiment_analysis(text):
# # #         text_lower = text.lower()
        
# # #         # High Confidence Very Negative/Spam Mock
# # #         if any(word in text_lower for word in ["scam", "refund", "fraud", "hatespeech", "critical"]):
# # #             # Score > 0.95 -> VERY_NEGATIVE (HIGH RISK)
# # #             return [{'label': 'NEGATIVE', 'score': 0.98}]
        
# # #         # Moderate Confidence Negative Mock
# # #         if any(word in text_lower for word in ["bad", "terrible", "disappointed", "poor"]):
# # #             # Score 0.60-0.90 -> NEGATIVE (MODERATE RISK)
# # #             return [{'label': 'NEGATIVE', 'score': 0.80}]

# # #         # High Confidence Very Positive Mock
# # #         if any(word in text_lower for word in ["love", "amazing", "best", "excellent", "high"]):
# # #             # Score > 0.90 -> VERY_POSITIVE (CLEAR)
# # #             return [{'label': 'POSITIVE', 'score': 0.93}]

# # #         # Moderate Confidence Positive Mock
# # #         if any(word in text_lower for word in ["good", "nice", "fine", "ok"]):
# # #             # Score 0.60-0.90 -> POSITIVE (CLEAR)
# # #             return [{'label': 'POSITIVE', 'score': 0.75}]

# # #         # Neutral Mock (Default for unknown/mixed phrases)
# # #         # Low confidence score, which classify_sentiment_risk maps to NEUTRAL (CLEAR)
# # #         return [{'label': 'POSITIVE', 'score': 0.55}] 
    
# # #     sentiment_pipeline = mock_sentiment_analysis
# # #     logging.warning("⚠️ Using HIGH PRECISION MOCK Sentiment Analysis Pipeline for Moderation.")


# # # # 2. In-Memory Data Store 
# # # MANUAL_REVIEW_QUEUE = []
# # # SCHEDULED_POSTS = []

# # # # Initial mock item to demonstrate HIGH risk due to VERY_NEGATIVE sentiment
# # # MANUAL_REVIEW_QUEUE.append({
# # #     "id": 1678886400000,
# # #     "sender_id": "critical_user_1",
# # #     "message_text": "This product is an absolute scam and I demand a full refund immediately.",
# # #     "reason": "Very Negative Sentiment (HIGH Risk)",
# # #     "timestamp": "2024-03-15T10:00:00Z",
# # #     "risk_level": "HIGH", 
# # #     "category": "Customer Complaint"
# # # })
# # # logging.info("--- NOTE: One mock item added to the Manual Review Queue for initial testing. ---")


# # # # --- Middleware (CORS) - Unchanged ---
# # # origins = [
# # #     "http://localhost:8000",
# # #     "http://127.0.0.1:8000",
# # #     "http://localhost:5001",
# # #     "http://127.0.0.1:5001",
# # #     "http://localhost:3000", 
# # #     "http://127.0.0.1:3000",
# # # ]

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=origins,
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )


# # # # --- Data Models (Pydantic) - Unchanged ---

# # # class ContentGenerationInput(BaseModel):
# # #     """Data model for the input parameters of content generation."""
# # #     prompt: str
# # #     length: str
# # #     platform: str
# # #     tone: str 

# # # class SocialMediaPost(BaseModel):
# # #     """The JSON structure we expect the LLM to return."""
# # #     post_content: str = Field(..., description="The main text content of the social media post.")
# # #     suggested_hashtags: List[str] = Field(..., description="A list of relevant hashtags for the post.")
# # #     platform: str = Field(..., description="The platform the post was optimized for (e.g., 'facebook', 'twitter').")
    
# # # class WebhookData(BaseModel):
# # #     """Data model for an incoming DM webhook."""
# # #     sender_id: str
# # #     message_text: str

# # # class ReviewItem(BaseModel):
# # #     """Data model for an item in the manual review queue."""
# # #     id: int
# # #     sender_id: str
# # #     message_text: str
# # #     reason: str
# # #     timestamp: str
# # #     risk_level: str
# # #     category: str


# # # # --- Core Logic Functions (OLLAMA INTEGRATION) ---

# # # # --- OLLAMA PROMPT GENERATION (UNCHANGED) ---
# # # def generate_ollama_prompt(data: ContentGenerationInput) -> str:
# # #     """
# # #     Constructs a detailed system instruction prompt for the LLM to ensure
# # #     it returns the content in the required JSON format and adheres to constraints.
# # #     """
# # #     # System Prompt to force JSON output
# # #     system_prompt = (
# # #         "You are an expert social media content creator. Your task is to generate a single social media post "
# # #         "based on the user's request. **You MUST respond only with a valid, clean JSON object** "
# # #         "that adheres to the following exact schema: "
# # #         "{\"post_content\": \"string\", \"suggested_hashtags\": [\"string\"], \"platform\": \"string\"}."
# # #         "Do not include any introductory text, conversation, or markdown formatting (e.g., ```json) outside the JSON object."
# # #         "The generated content must strictly follow the tone, length, and platform constraints."
# # #     )
    
# # #     # User Query containing all parameters
# # #     user_query = (
# # #         f"Generate a post for the topic: '{data.prompt}'. "
# # #         f"The post should be a '{data.length}' length, "
# # #         f"using a '{data.tone}' tone, and optimized for the '{data.platform}' platform. "
# # #         f"Ensure the post content is professional and engaging."
# # #     )
    
# # #     return f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"


# # # # --- OLLAMA API CALL (UNCHANGED) ---
# # # def generate_post_content_with_ollama(data: ContentGenerationInput) -> SocialMediaPost:
# # #     """
# # #     Function to call the local Ollama API endpoint for content generation.
# # #     """
# # #     prompt = generate_ollama_prompt(data)
    
# # #     # Ollama REST API Payload
# # #     payload = {
# # #         "model": OLLAMA_MODEL,
# # #         "prompt": prompt,
# # #         "stream": False,
# # #         "options": {
# # #             # Low temperature for deterministic, structured output
# # #             "temperature": 0.3 
# # #         }
# # #     }
    
# # #     try:
# # #         # 1. Make the request to the local Ollama server
# # #         response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
# # #         response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)

# # #         # 2. Extract and parse the response
# # #         result = response.json()
# # #         raw_response_text = result.get('response', '').strip()
        
# # #         # 3. Clean and Validate JSON
# # #         try:
# # #             # Attempt to parse the raw text directly as JSON
# # #             post_data = json.loads(raw_response_text)
            
# # #             # Use Pydantic to validate the structure
# # #             validated_post = SocialMediaPost(**post_data)
# # #             return validated_post
        
# # #         except json.JSONDecodeError:
# # #             logging.error(f"Ollama returned non-JSON text: {raw_response_text[:100]}...")
# # #             raise ValueError("LLM returned malformed JSON. Check model and prompt structure.")

# # #     except requests.exceptions.ConnectionError:
# # #         logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
# # #         raise ConnectionError("Could not connect to the local Ollama server. Please check that Ollama is running.")
    
# # #     except requests.exceptions.Timeout:
# # #         raise TimeoutError("Ollama request timed out. Increase timeout or check model performance.")
    
# # #     except Exception as e:
# # #         logging.error(f"An unexpected error occurred during Ollama generation: {e}")
# # #         raise e

# # # def classify_sentiment_risk(label: str, score: float) -> tuple[str, str, bool]:
# # #     """
# # #     Maps the sentiment analysis result (label, score) to a precise 5-class risk level and reason.
# # #     Uses tighter thresholds for high accuracy.
# # #     """
    
# # #     sentiment_reason = ""
# # #     risk_level = "CLEAR"
# # #     needs_review = False

# # #     # --- NEGATIVE SENTIMENTS ---
# # #     if label == 'NEGATIVE':
# # #         if score > 0.90:
# # #             # High confidence in extreme negativity -> VERY_NEGATIVE (HIGH RISK)
# # #             sentiment_reason = "Very Negative Sentiment"
# # #             risk_level = "HIGH"
# # #             needs_review = True
# # #         elif score >= 0.60:
# # #             # Moderate confidence in negativity -> NEGATIVE (MODERATE RISK)
# # #             sentiment_reason = "Negative Sentiment"
# # #             risk_level = "MODERATE"
# # #             needs_review = True
# # #         else:
# # #             # Low confidence negative scores fall into Neutral/Clear
# # #             sentiment_reason = "Neutral Sentiment"
# # #             risk_level = "CLEAR"
            
# # #     # --- POSITIVE & NEUTRAL SENTIMENTS ---
# # #     elif label == 'POSITIVE':
# # #         if score > 0.90:
# # #             # High confidence in extreme positivity -> VERY_POSITIVE (CLEAR)
# # #             sentiment_reason = "Very Positive Sentiment"
# # #             risk_level = "CLEAR"
# # #             needs_review = False
# # #         elif score >= 0.60:
# # #             # Moderate confidence in positivity -> POSITIVE (CLEAR)
# # #             sentiment_reason = "Positive Sentiment"
# # #             risk_level = "CLEAR"
# # #             needs_review = False
# # #         else:
# # #             # Low confidence positive scores fall into Neutral/Clear
# # #             sentiment_reason = "Neutral Sentiment"
# # #             risk_level = "CLEAR"
# # #             needs_review = False
    
# # #     else: 
# # #         # Fallback for unexpected labels
# # #         sentiment_reason = "Neutral Sentiment"
# # #         risk_level = "CLEAR"
# # #         needs_review = False
        
# # #     return sentiment_reason, risk_level, needs_review

# # # def moderate_message(message: str) -> tuple[str, bool]:
# # #     """
# # #     Applies real-time sentiment analysis and spam flagging.
# # #     Returns: (Reason, Needs_Review_Boolean)
# # #     """
    
# # #     # 1. SPAM/PHISHING CHECK (Highest Priority)
# # #     spam_keywords = ["click here for free money", "guaranteed profit", "scam now", "easy cash fast", "link in bio to win"]
# # #     if any(keyword in message.lower() for keyword in spam_keywords):
# # #         return "Spam/Phishing Risk (CRITICAL Risk)", True
    
# # #     # 2. SENTIMENT ANALYSIS
# # #     try:
# # #         results = sentiment_pipeline(message)
# # #         sentiment_label = results[0]['label']
# # #         sentiment_score = results[0]['score']

# # #         logging.info(f"📊 Sentiment Analysis: {sentiment_label} (Score: {sentiment_score:.2f})")
        
# # #         # Map sentiment and score to risk level
# # #         reason, risk_level, needs_review = classify_sentiment_risk(sentiment_label, sentiment_score)

# # #         if needs_review:
# # #             # Append risk level to reason for queue display
# # #             return f"{reason} ({risk_level} Risk)", needs_review

# # #         return reason, needs_review

# # #     except Exception as e:
# # #         logging.error(f"Error during moderation: {e}")
# # #         # Default to review if AI processing fails
# # #         return "ERROR during AI processing", True 


# # # # --- API Endpoints ---

# # # @app.get("/")
# # # def read_root():
# # #     return {"status": "AI Social Media Backend is Running (OLLAMA INTEGRATION ACTIVE)"}

# # # # --- Content Generation Endpoint (OLLAMA ACTIVE) ---
# # # @app.post("/api/content/generate", response_model=SocialMediaPost)
# # # async def generate_post_content(data: ContentGenerationInput):
# # #     """Generates social media content using the local Ollama LLM."""
    
# # #     logging.info(f"⚡ OLLAMA generation requested for {data.platform} post (Topic: {data.prompt})...")

# # #     # The actual LLM call is synchronous, but we wrap it in a thread 
# # #     # for the async FastAPI endpoint to remain non-blocking.
# # #     loop = asyncio.get_event_loop()
    
# # #     try:
# # #         # Call the synchronous LLM function within a thread pool
# # #         validated_post = await loop.run_in_executor(
# # #             None, # Use default thread pool
# # #             generate_post_content_with_ollama, 
# # #             data
# # #         )
        
# # #         logging.info(f"🎉 Ollama Content Generated Successfully for {validated_post.platform}.")
# # #         return validated_post

# # #     except (ConnectionError, TimeoutError) as e:
# # #         # Catch connection-specific errors and return 503 Service Unavailable
# # #         raise HTTPException(
# # #             status_code=503, 
# # #             detail=f"Local LLM Error: {e}"
# # #         )
# # #     except Exception as e:
# # #         # Catch any other error, e.g., JSONDecodeError
# # #         raise HTTPException(status_code=500, detail=f"Failed to generate content: {e}")


# # # # --- Moderation & Review Endpoints (Updated) ---

# # # @app.post("/api/moderation/simulate", response_model=Union[ReviewItem, dict])
# # # def dm_webhook(data: WebhookData):
# # #     """Real-time processing of incoming DMs for moderation."""
# # #     logging.info(f"--- DM Received: \"{data.message_text[:40]}...\" from {data.sender_id} ---")

# # #     reason, needs_review = moderate_message(data.message_text)
    
# # #     if needs_review:
# # #         # Initialize defaults
# # #         risk_level = "LOW" 
# # #         category = "Other"

# # #         # Determine risk level and category based on reason
# # #         if "CRITICAL" in reason:
# # #             risk_level = "CRITICAL"
# # #             category = "Security/Spam"
# # #         elif "HIGH" in reason:
# # #             risk_level = "HIGH"
# # #             category = "Customer Complaint" if "Sentiment" in reason else "Policy Violation"
# # #         elif "MODERATE" in reason:
# # #             risk_level = "MODERATE"
# # #             category = "Customer Complaint" if "Sentiment" in reason else "Policy Violation"
# # #         elif "ERROR" in reason:
# # #             risk_level = "HIGH"
# # #             category = "System Error"
        
# # #         new_item = ReviewItem(
# # #             id=int(time.time() * 1000),
# # #             sender_id=data.sender_id,
# # #             message_text=data.message_text,
# # #             reason=reason, # The detailed reason including risk level
# # #             timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
# # #             risk_level=risk_level, 
# # #             category=category
# # #         ).model_dump()

# # #         MANUAL_REVIEW_QUEUE.insert(0, new_item)
# # #         logging.warning(f"🚨 Added to Review Queue: ID {new_item['id']} ({reason})")
# # #         return new_item
    
# # #     return {"status": "PASSED", "reason": reason}

# # # @app.get("/api/moderation/queue", response_model=List[ReviewItem])
# # # def get_review_queue():
# # #     """Endpoint to retrieve all items in the manual review queue."""
# # #     return MANUAL_REVIEW_QUEUE 

# # # @app.post("/api/moderation/queue/clear")
# # # def clear_review_queue():
# # #     """Endpoint to clear all items from the manual review queue."""
# # #     MANUAL_REVIEW_QUEUE.clear()
    
# # #     MANUAL_REVIEW_QUEUE.append(ReviewItem(
# # #         id=int(time.time() * 1000),
# # #         sender_id="system_message",
# # #         message_text="The queue was manually cleared by a human moderator.",
# # #         reason="Queue Cleared",
# # #         timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
# # #         risk_level="LOW",
# # #         category="Admin Action"
# # #     ).model_dump())

# # #     logging.info("🧹 Manual Review Queue Cleared.")
# # #     return {"status": "success", "message": "Queue cleared."}


# # # # --- Background Scheduler (Mock) - UNCHANGED ---

# # # @app.get("/api/schedule/posts")
# # # def get_scheduled_posts():
# # #     """Get all currently scheduled posts."""
# # #     return {"scheduled_posts": SCHEDULED_POSTS}

# # # async def background_scheduler():
# # #     """This function simulates a persistent background scheduler."""
# # #     logging.info("--- Starting Background Scheduler ---")
# # #     while True:
# # #         await asyncio.sleep(60) 

# # # @app.on_event("startup")
# # # async def startup_event():
# # #     """Starts the background scheduler on API startup."""
# # #     if not hasattr(app.state, 'scheduler_task'):
# # #         app.state.scheduler_task = asyncio.create_task(background_scheduler())

# # # @app.on_event("shutdown")
# # # def shutdown_event():
# # #     """Executes on server shutdown."""
# # #     if hasattr(app.state, 'scheduler_task') and not app.state.scheduler_task.done():
# # #         app.state.scheduler_task.cancel()
# # #     logging.info("Server shutting down.")


# # # # --- MANDATORY SERVER RUN BLOCK ---
# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="127.0.0.1", port=8000)






# # # import time
# # # import asyncio
# # # import json
# # # import logging
# # # import random
# # # from typing import List, Optional, Union
# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from pydantic import BaseModel, Field

# # # # We use 'requests' to talk to the local Ollama API for BOTH content generation and moderation.
# # # import requests 

# # # # --- Configuration ---

# # # # Configure logging to show INFO messages
# # # logging.basicConfig(level=logging.INFO, format='INFO: %(message)s')

# # # # --- Global State and Initialization ---

# # # # --- IMPORTANT LLM CONFIGURATION ---
# # # OLLAMA_MODEL = "mistral" # CHANGE THIS if you pulled a different model (e.g., 'gemma:2b')
# # # OLLAMA_API_URL = "http://localhost:11434/api/generate"
# # # # ---

# # # app = FastAPI(title="Social Media AI Backend (OLLAMA INTEGRATION)", version="0.1.0")

# # # # 1. Moderation Model Initialization (Removed old sentiment pipeline)
# # # # The moderation now relies entirely on the Ollama LLM, which is more accurate for multi-class tasks.
# # # logging.info("✅ Moderation is now powered by Ollama LLM for higher precision.")


# # # # 2. In-Memory Data Store 
# # # MANUAL_REVIEW_QUEUE = []
# # # SCHEDULED_POSTS = []

# # # # Initial mock item to demonstrate HIGH risk due to VERY_NEGATIVE sentiment
# # # MANUAL_REVIEW_QUEUE.append({
# # #     "id": 1678886400000,
# # #     "sender_id": "critical_user_1",
# # #     "message_text": "This product is an absolute scam and I demand a full refund immediately.",
# # #     "reason": "Very Negative Sentiment (HIGH Risk)",
# # #     "timestamp": "2024-03-15T10:00:00Z",
# # #     "risk_level": "HIGH", 
# # #     "category": "Customer Complaint"
# # # })
# # # logging.info("--- NOTE: One mock item added to the Manual Review Queue for initial testing. ---")


# # # # --- Middleware (CORS) - Unchanged ---
# # # origins = [
# # #     "http://localhost:8000",
# # #     "http://127.0.0.1:8000",
# # #     "http://localhost:5001",
# # #     "http://127.0.0.1:5001",
# # #     "http://localhost:3000", 
# # #     "http://127.0.0.1:3000",
# # # ]

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=origins,
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )


# # # # --- Data Models (Pydantic) ---

# # # class ContentGenerationInput(BaseModel):
# # #     """Data model for the input parameters of content generation."""
# # #     prompt: str
# # #     length: str
# # #     platform: str
# # #     tone: str 

# # # class SocialMediaPost(BaseModel):
# # #     """The JSON structure we expect the LLM to return for generation."""
# # #     post_content: str = Field(..., description="The main text content of the social media post.")
# # #     suggested_hashtags: List[str] = Field(..., description="A list of relevant hashtags for the post.")
# # #     platform: str = Field(..., description="The platform the post was optimized for (e.g., 'facebook', 'twitter').")

# # # class ModerationResult(BaseModel):
# # #     """The STRICT JSON structure we expect the LLM to return for moderation."""
# # #     risk_level: str = Field(..., description="Classification: CRITICAL, HIGH, MODERATE, LOW, or CLEAR.")
# # #     category: str = Field(..., description="Classification: Spam/Phishing, Policy Violation, Customer Complaint, or Clear/Other.")
# # #     reason: str = Field(..., description="A brief, human-readable explanation of why this risk level was assigned.")
    
# # # class WebhookData(BaseModel):
# # #     """Data model for an incoming DM webhook."""
# # #     sender_id: str
# # #     message_text: str

# # # class ReviewItem(BaseModel):
# # #     """Data model for an item in the manual review queue."""
# # #     id: int
# # #     sender_id: str
# # #     message_text: str
# # #     reason: str
# # #     timestamp: str
# # #     risk_level: str
# # #     category: str


# # # # --- Core Logic Functions (OLLAMA INTEGRATION) ---

# # # # --- OLLAMA PROMPT GENERATION (UNCHANGED) ---
# # # def generate_ollama_prompt(data: ContentGenerationInput) -> str:
# # #     """
# # #     Constructs a detailed system instruction prompt for the LLM to ensure
# # #     it returns the content in the required JSON format and adheres to constraints.
# # #     """
# # #     # System Prompt to force JSON output
# # #     system_prompt = (
# # #         "You are an expert social media content creator. Your task is to generate a single social media post "
# # #         "based on the user's request. **You MUST respond only with a valid, clean JSON object** "
# # #         "that adheres to the following exact schema: "
# # #         "{\"post_content\": \"string\", \"suggested_hashtags\": [\"string\"], \"platform\": \"string\"}."
# # #         "Do not include any introductory text, conversation, or markdown formatting (e.g., ```json) outside the JSON object."
# # #         "The generated content must strictly follow the tone, length, and platform constraints."
# # #     )
    
# # #     # User Query containing all parameters
# # #     user_query = (
# # #         f"Generate a post for the topic: '{data.prompt}'. "
# # #         f"The post should be a '{data.length}' length, "
# # #         f"using a '{data.tone}' tone, and optimized for the '{data.platform}' platform. "
# # #         f"Ensure the post content is professional and engaging."
# # #     )
    
# # #     return f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"


# # # # --- OLLAMA API CALL (UNCHANGED) ---
# # # def generate_post_content_with_ollama(data: ContentGenerationInput) -> SocialMediaPost:
# # #     """
# # #     Function to call the local Ollama API endpoint for content generation.
# # #     (This function remains UNCHANGED)
# # #     """
# # #     prompt = generate_ollama_prompt(data)
    
# # #     # Ollama REST API Payload
# # #     payload = {
# # #         "model": OLLAMA_MODEL,
# # #         "prompt": prompt,
# # #         "stream": False,
# # #         "options": {
# # #             # Low temperature for deterministic, structured output
# # #             "temperature": 0.3 
# # #         }
# # #     }
    
# # #     try:
# # #         response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
# # #         response.raise_for_status() 

# # #         result = response.json()
# # #         raw_response_text = result.get('response', '').strip()
        
# # #         try:
# # #             post_data = json.loads(raw_response_text)
# # #             validated_post = SocialMediaPost(**post_data)
# # #             return validated_post
        
# # #         except json.JSONDecodeError:
# # #             logging.error(f"Ollama returned malformed JSON for generation: {raw_response_text[:100]}...")
# # #             raise ValueError("LLM returned malformed JSON. Check model and prompt structure.")

# # #     except requests.exceptions.ConnectionError:
# # #         logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
# # #         raise ConnectionError("Could not connect to the local Ollama server. Please check that Ollama is running.")
    
# # #     except requests.exceptions.Timeout:
# # #         raise TimeoutError("Ollama request timed out. Increase timeout or check model performance.")
    
# # #     except Exception as e:
# # #         logging.error(f"An unexpected error occurred during Ollama generation: {e}")
# # #         raise e

# # # # --- NEW OLLAMA MODERATION LOGIC ---
# # # def moderate_message_with_ollama(message: str) -> ModerationResult:
# # #     """
# # #     Uses the Ollama LLM to classify a message into one of five risk levels 
# # #     and returns a structured JSON result for moderation.
# # #     """
    
# # #     # System Prompt to force JSON output with specific categories
# # #     system_prompt = (
# # #         "You are an expert social media moderation AI. Your task is to analyze the user message and "
# # #         "classify its risk level and category. **You MUST respond ONLY with a valid, clean JSON object** "
# # #         "that strictly adheres to the schema: "
# # #         "{\"risk_level\": \"string\", \"category\": \"string\", \"reason\": \"string\"}."
# # #         "Do not include any other text or markdown formatting outside the JSON object."
        
# # #         "\n\n--- CLASSIFICATION RULES ---"
# # #         "\nRISK_LEVEL must be one of: CRITICAL, HIGH, MODERATE, LOW, CLEAR."
# # #         "\nCATEGORY must be one of: Spam/Phishing, Policy Violation, Customer Complaint, Clear/Other."
        
# # #         "\n\nCRITICAL Risk: Direct threats, hate speech, illegal content, or obvious spam/phishing attempts."
# # #         "\nHIGH Risk: Very aggressive or highly offensive language, extreme negative sentiment (e.g., 'scam', 'fraud')."
# # #         "\nMODERATE Risk: Strong negative sentiment (e.g., 'bad', 'terrible'), minor policy violations."
# # #         "\nLOW Risk: Neutral or slightly positive sentiment, very mild or ambiguous concerns."
# # #         "\nCLEAR Risk: Clearly positive or neutral, non-concerning messages."
# # #     )
    
# # #     user_query = f"Analyze and classify the following message:\n'{message}'"
# # #     prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"
    
# # #     payload = {
# # #         "model": OLLAMA_MODEL,
# # #         "prompt": prompt,
# # #         "stream": False,
# # #         "options": {
# # #             # Low temperature for deterministic, structured output
# # #             "temperature": 0.2
# # #         }
# # #     }

# # #     try:
# # #         response = requests.post(OLLAMA_API_URL, json=payload, timeout=45) # Increased timeout for moderation
# # #         response.raise_for_status()

# # #         result = response.json()
# # #         raw_response_text = result.get('response', '').strip()

# # #         try:
# # #             # Attempt to parse the raw text directly as JSON
# # #             moderation_data = json.loads(raw_response_text)
            
# # #             # Use Pydantic to validate the structure
# # #             validated_result = ModerationResult(**moderation_data)
# # #             return validated_result
        
# # #         except json.JSONDecodeError:
# # #             logging.error(f"Ollama returned malformed JSON for moderation: {raw_response_text[:100]}...")
# # #             # Fallback to a high risk if the LLM response is unparseable
# # #             return ModerationResult(
# # #                 risk_level="HIGH",
# # #                 category="System Error",
# # #                 reason="LLM response malformed, manual review required."
# # #             )
        
# # #     except requests.exceptions.ConnectionError:
# # #         logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
# # #         # Fallback to high risk on connection failure
# # #         return ModerationResult(
# # #             risk_level="HIGH",
# # #             category="System Error",
# # #             reason="Ollama connection failed, manual review required."
# # #         )
# # #     except Exception as e:
# # #         logging.error(f"An unexpected error occurred during Ollama moderation: {e}")
# # #         # Fallback to high risk on any other failure
# # #         return ModerationResult(
# # #             risk_level="HIGH",
# # #             category="System Error",
# # #             reason=f"Unexpected error: {str(e)[:50]}"
# # #         )


# # # # --- API Endpoints ---

# # # @app.get("/")
# # # def read_root():
# # #     return {"status": "AI Social Media Backend is Running (OLLAMA INTEGRATION ACTIVE)"}

# # # # --- Content Generation Endpoint (OLLAMA ACTIVE) ---
# # # @app.post("/api/content/generate", response_model=SocialMediaPost)
# # # async def generate_post_content(data: ContentGenerationInput):
# # #     """Generates social media content using the local Ollama LLM."""
    
# # #     logging.info(f"⚡ OLLAMA generation requested for {data.platform} post (Topic: {data.prompt})...")

# # #     loop = asyncio.get_event_loop()
    
# # #     try:
# # #         validated_post = await loop.run_in_executor(
# # #             None, 
# # #             generate_post_content_with_ollama, 
# # #             data
# # #         )
        
# # #         logging.info(f"🎉 Ollama Content Generated Successfully for {validated_post.platform}.")
# # #         return validated_post

# # #     except (ConnectionError, TimeoutError) as e:
# # #         raise HTTPException(
# # #             status_code=503, 
# # #             detail=f"Local LLM Error: {e}"
# # #         )
# # #     except Exception as e:
# # #         raise HTTPException(status_code=500, detail=f"Failed to generate content: {e}")


# # # # --- Moderation & Review Endpoints (Updated) ---

# # # @app.post("/api/moderation/simulate", response_model=Union[ReviewItem, dict])
# # # async def dm_webhook(data: WebhookData):
# # #     """Real-time processing of incoming DMs for moderation using Ollama."""
# # #     logging.info(f"--- DM Received: \"{data.message_text[:40]}...\" from {data.sender_id} ---")

# # #     # Use a thread pool to run the synchronous Ollama call for moderation
# # #     loop = asyncio.get_event_loop()
# # #     moderation_result = await loop.run_in_executor(
# # #         None, 
# # #         moderate_message_with_ollama, 
# # #         data.message_text
# # #     )

# # #     needs_review = moderation_result.risk_level in ["CRITICAL", "HIGH", "MODERATE"]
    
# # #     if needs_review:
        
# # #         # Determine the final category based on the Ollama output
# # #         risk_level = moderation_result.risk_level
# # #         category = moderation_result.category
# # #         reason = f"{moderation_result.reason} ({risk_level} Risk)"
        
# # #         new_item = ReviewItem(
# # #             id=int(time.time() * 1000),
# # #             sender_id=data.sender_id,
# # #             message_text=data.message_text,
# # #             reason=reason,
# # #             timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
# # #             risk_level=risk_level, 
# # #             category=category
# # #         ).model_dump()

# # #         MANUAL_REVIEW_QUEUE.insert(0, new_item)
# # #         logging.warning(f"🚨 Added to Review Queue: ID {new_item['id']} ({reason})")
# # #         return new_item
    
# # #     return {"status": "PASSED", "reason": moderation_result.reason}

# # # @app.get("/api/moderation/queue", response_model=List[ReviewItem])
# # # def get_review_queue():
# # #     """Endpoint to retrieve all items in the manual review queue."""
# # #     return MANUAL_REVIEW_QUEUE 

# # # @app.post("/api/moderation/queue/clear")
# # # def clear_review_queue():
# # #     """Endpoint to clear all items from the manual review queue."""
# # #     MANUAL_REVIEW_QUEUE.clear()
    
# # #     MANUAL_REVIEW_QUEUE.append(ReviewItem(
# # #         id=int(time.time() * 1000),
# # #         sender_id="system_message",
# # #         message_text="The queue was manually cleared by a human moderator.",
# # #         reason="Queue Cleared",
# # #         timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
# # #         risk_level="LOW",
# # #         category="Admin Action"
# # #     ).model_dump())

# # #     logging.info("🧹 Manual Review Queue Cleared.")
# # #     return {"status": "success", "message": "Queue cleared."}


# # # # --- Background Scheduler (Mock) - UNCHANGED ---

# # # @app.get("/api/schedule/posts")
# # # def get_scheduled_posts():
# # #     """Get all currently scheduled posts."""
# # #     return {"scheduled_posts": SCHEDULED_POSTS}

# # # async def background_scheduler():
# # #     """This function simulates a persistent background scheduler."""
# # #     logging.info("--- Starting Background Scheduler ---")
# # #     while True:
# # #         await asyncio.sleep(60) 

# # # @app.on_event("startup")
# # # async def startup_event():
# # #     """Starts the background scheduler on API startup."""
# # #     if not hasattr(app.state, 'scheduler_task'):
# # #         app.state.scheduler_task = asyncio.create_task(background_scheduler())

# # # @app.on_event("shutdown")
# # # def shutdown_event():
# # #     """Executes on server shutdown."""
# # #     if hasattr(app.state, 'scheduler_task') and not app.state.scheduler_task.done():
# # #         app.state.scheduler_task.cancel()
# # #     logging.info("Server shutting down.")


# # # # --- MANDATORY SERVER RUN BLOCK ---
# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="127.0.0.1", port=8000)





# # # import time
# # # import asyncio
# # # import json
# # # import logging
# # # import random
# # # from typing import List, Optional, Union
# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from pydantic import BaseModel, Field

# # # # We use 'requests' to talk to the local Ollama API for BOTH content generation and moderation.
# # # import requests 

# # # # --- Configuration ---

# # # # Configure logging to show INFO messages
# # # logging.basicConfig(level=logging.INFO, format='INFO: %(message)s')

# # # # --- Global State and Initialization ---

# # # # --- IMPORTANT LLM CONFIGURATION ---
# # # OLLAMA_MODEL = "mistral" # CHANGE THIS if you pulled a different model (e.g., 'gemma:2b')
# # # OLLAMA_API_URL = "http://localhost:11434/api/generate"
# # # # ---

# # # app = FastAPI(title="Social Media AI Backend (OLLAMA INTEGRATION)", version="0.1.0")

# # # # 1. Moderation Model Initialization (Using Ollama LLM)
# # # logging.info("✅ Moderation is now powered by Ollama LLM for higher precision.")


# # # # 2. In-Memory Data Store 
# # # MANUAL_REVIEW_QUEUE = []
# # # SCHEDULED_POSTS = []

# # # # Initial mock item to demonstrate HIGH risk due to VERY_NEGATIVE sentiment
# # # MANUAL_REVIEW_QUEUE.append({
# # #     "id": 1678886400000,
# # #     "sender_id": "critical_user_1",
# # #     "message_text": "This product is an absolute scam and I demand a full refund immediately.",
# # #     "reason": "Very Negative Sentiment (HIGH Risk)",
# # #     "timestamp": "2024-03-15T10:00:00Z",
# # #     "risk_level": "HIGH", 
# # #     "category": "Customer Complaint"
# # # })
# # # logging.info("--- NOTE: One mock item added to the Manual Review Queue for initial testing. ---")


# # # # --- Middleware (CORS) - Unchanged ---
# # # origins = [
# # #     "http://localhost:8000",
# # #     "http://127.0.0.1:8000",
# # #     "http://localhost:5001",
# # #     "http://127.0.0.1:5001",
# # #     "http://localhost:3000", 
# # #     "http://127.0.0.1:3000",
# # # ]

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=origins,
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )


# # # # --- Data Models (Pydantic) ---

# # # class ContentGenerationInput(BaseModel):
# # #     """Data model for the input parameters of content generation."""
# # #     prompt: str
# # #     length: str
# # #     platform: str
# # #     tone: str 

# # # class SocialMediaPost(BaseModel):
# # #     """The JSON structure we expect the LLM to return for generation."""
# # #     post_content: str = Field(..., description="The main text content of the social media post.")
# # #     suggested_hashtags: List[str] = Field(..., description="A list of relevant hashtags for the post.")
# # #     platform: str = Field(..., description="The platform the post was optimized for (e.g., 'facebook', 'twitter').")

# # # class ModerationResult(BaseModel):
# # #     """The STRICT JSON structure we expect the LLM to return for moderation."""
# # #     risk_level: str = Field(..., description="Classification: CRITICAL, HIGH, MODERATE, LOW, or CLEAR.")
# # #     category: str = Field(..., description="Classification: Spam/Phishing, Policy Violation, Customer Complaint, or Clear/Other.")
# # #     reason: str = Field(..., description="A brief, human-readable explanation of why this risk level was assigned.")
    
# # # class WebhookData(BaseModel):
# # #     """Data model for an incoming DM webhook."""
# # #     sender_id: str
# # #     message_text: str

# # # class ReviewItem(BaseModel):
# # #     """Data model for an item in the manual review queue."""
# # #     id: int
# # #     sender_id: str
# # #     message_text: str
# # #     reason: str
# # #     timestamp: str
# # #     risk_level: str
# # #     category: str


# # # # --- Core Logic Functions (OLLAMA INTEGRATION) ---

# # # def generate_ollama_prompt(data: ContentGenerationInput) -> str:
# # #     """
# # #     Constructs a detailed system instruction prompt for the LLM to ensure
# # #     it returns the content in the required JSON format and adheres to constraints.
# # #     """
# # #     # System Prompt to force JSON output
# # #     system_prompt = (
# # #         "You are an expert social media content creator. Your task is to generate a single social media post "
# # #         "based on the user's request. **You MUST respond only with a valid, clean JSON object** "
# # #         "that adheres to the following exact schema: "
# # #         "{\"post_content\": \"string\", \"suggested_hashtags\": [\"string\"], \"platform\": \"string\"}."
# # #         "Do not include any introductory text, conversation, or markdown formatting (e.g., ```json) outside the JSON object."
# # #         "The generated content must strictly follow the tone, length, and platform constraints."
# # #     )
    
# # #     # User Query containing all parameters
# # #     user_query = (
# # #         f"Generate a post for the topic: '{data.prompt}'. "
# # #         f"The post should be a '{data.length}' length, "
# # #         f"using a '{data.tone}' tone, and optimized for the '{data.platform}' platform. "
# # #         f"Ensure the post content is professional and engaging."
# # #     )
    
# # #     return f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"


# # # def generate_post_content_with_ollama(data: ContentGenerationInput) -> SocialMediaPost:
# # #     """
# # #     Function to call the local Ollama API endpoint for content generation.
# # #     """
# # #     prompt = generate_ollama_prompt(data)
    
# # #     # Ollama REST API Payload
# # #     payload = {
# # #         "model": OLLAMA_MODEL,
# # #         "prompt": prompt,
# # #         "stream": False,
# # #         "options": {
# # #             # Low temperature for deterministic, structured output
# # #             "temperature": 0.3 
# # #         }
# # #     }
    
# # #     try:
# # #         response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
# # #         response.raise_for_status() 

# # #         result = response.json()
# # #         raw_response_text = result.get('response', '').strip()
        
# # #         try:
# # #             post_data = json.loads(raw_response_text)
# # #             validated_post = SocialMediaPost(**post_data)
# # #             return validated_post
        
# # #         except json.JSONDecodeError:
# # #             logging.error(f"Ollama returned malformed JSON for generation: {raw_response_text[:100]}...")
# # #             raise ValueError("LLM returned malformed JSON. Check model and prompt structure.")

# # #     except requests.exceptions.ConnectionError:
# # #         logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
# # #         raise ConnectionError("Could not connect to the local Ollama server. Please check that Ollama is running.")
    
# # #     except requests.exceptions.Timeout:
# # #         raise TimeoutError("Ollama request timed out. Increase timeout or check model performance.")
    
# # #     except Exception as e:
# # #         logging.error(f"An unexpected error occurred during Ollama generation: {e}")
# # #         raise e

# # # def moderate_message_with_ollama(message: str) -> ModerationResult:
# # #     """
# # #     Uses the Ollama LLM to classify a message into one of five risk levels 
# # #     and returns a structured JSON result for moderation.
# # #     """
    
# # #     # System Prompt to force JSON output with specific categories
# # #     system_prompt = (
# # #         "You are an expert social media moderation AI. Your task is to analyze the user message and "
# # #         "classify its risk level and category. **You MUST respond ONLY with a valid, clean JSON object** "
# # #         "that strictly adheres to the schema: "
# # #         "{\"risk_level\": \"string\", \"category\": \"string\", \"reason\": \"string\"}."
# # #         "Do not include any other text or markdown formatting outside the JSON object."
        
# # #         "\n\n--- CLASSIFICATION RULES ---"
# # #         "\nRISK_LEVEL must be one of: CRITICAL, HIGH, MODERATE, LOW, CLEAR."
# # #         "\nCATEGORY must be one of: Spam/Phishing, Policy Violation, Customer Complaint, Clear/Other."
        
# # #         "\n\nCRITICAL Risk: Direct threats, hate speech, illegal content, or obvious spam/phishing attempts. This requires immediate human review."
# # #         "\nHIGH Risk: Very aggressive or highly offensive language, extreme negative sentiment (e.g., 'scam', 'fraud'). This requires human review."
# # #         "\nMODERATE Risk: Strong negative sentiment (e.g., 'bad', 'terrible'), minor policy violations. This might require human review."
# # #         "\nLOW Risk: Neutral or slightly positive sentiment, very mild or ambiguous concerns. This usually passes."
# # #         "\nCLEAR Risk: Clearly positive or neutral, non-concerning messages. This always passes."
# # #     )
    
# # #     user_query = f"Analyze and classify the following message:\n'{message}'"
# # #     prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"
    
# # #     payload = {
# # #         "model": OLLAMA_MODEL,
# # #         "prompt": prompt,
# # #         "stream": False,
# # #         "options": {
# # #             # Low temperature for deterministic, structured output
# # #             "temperature": 0.2
# # #         }
# # #     }

# # #     try:
# # #         response = requests.post(OLLAMA_API_URL, json=payload, timeout=45) # Increased timeout for moderation
# # #         response.raise_for_status()

# # #         result = response.json()
# # #         raw_response_text = result.get('response', '').strip()

# # #         try:
# # #             # Attempt to parse the raw text directly as JSON
# # #             moderation_data = json.loads(raw_response_text)
            
# # #             # Use Pydantic to validate the structure
# # #             validated_result = ModerationResult(**moderation_data)
# # #             return validated_result
        
# # #         except json.JSONDecodeError:
# # #             logging.error(f"Ollama returned malformed JSON for moderation: {raw_response_text[:100]}...")
# # #             # Fallback to a high risk if the LLM response is unparseable
# # #             return ModerationResult(
# # #                 risk_level="HIGH",
# # #                 category="System Error",
# # #                 reason="LLM response malformed, manual review required."
# # #             )
        
# # #     except requests.exceptions.ConnectionError:
# # #         logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
# # #         # Fallback to high risk on connection failure
# # #         return ModerationResult(
# # #             risk_level="HIGH",
# # #             category="System Error",
# # #             reason="Ollama connection failed, manual review required."
# # #         )
# # #     except Exception as e:
# # #         logging.error(f"An unexpected error occurred during Ollama moderation: {e}")
# # #         # Fallback to high risk on any other failure
# # #         return ModerationResult(
# # #             risk_level="HIGH",
# # #             category="System Error",
# # #             reason=f"Unexpected error: {str(e)[:50]}"
# # #         )


# # # # --- API Endpoints ---

# # # @app.get("/")
# # # def read_root():
# # #     return {"status": "AI Social Media Backend is Running (OLLAMA INTEGRATION ACTIVE)"}

# # # # --- Content Generation Endpoint (OLLAMA ACTIVE) ---
# # # @app.post("/api/content/generate", response_model=SocialMediaPost)
# # # async def generate_post_content(data: ContentGenerationInput):
# # #     """Generates social media content using the local Ollama LLM."""
    
# # #     logging.info(f"⚡ OLLAMA generation requested for {data.platform} post (Topic: {data.prompt})...")

# # #     loop = asyncio.get_event_loop()
    
# # #     try:
# # #         validated_post = await loop.run_in_executor(
# # #             None, 
# # #             generate_post_content_with_ollama, 
# # #             data
# # #         )
        
# # #         logging.info(f"🎉 Ollama Content Generated Successfully for {validated_post.platform}.")
# # #         return validated_post

# # #     except (ConnectionError, TimeoutError) as e:
# # #         raise HTTPException(
# # #             status_code=503, 
# # #             detail=f"Local LLM Error: {e}"
# # #         )
# # #     except Exception as e:
# # #         raise HTTPException(status_code=500, detail=f"Failed to generate content: {e}")


# # # # --- Moderation & Review Endpoints (Updated) ---

# # # @app.post("/api/moderation/simulate", response_model=Union[ReviewItem, dict])
# # # async def dm_webhook(data: WebhookData):
# # #     """Real-time processing of incoming DMs for moderation using Ollama."""
# # #     logging.info(f"--- DM Received: \"{data.message_text[:40]}...\" from {data.sender_id} ---")

# # #     # Use a thread pool to run the synchronous Ollama call for moderation
# # #     loop = asyncio.get_event_loop()
# # #     moderation_result = await loop.run_in_executor(
# # #         None, 
# # #         moderate_message_with_ollama, 
# # #         data.message_text
# # #     )

# # #     # Risk levels that require a manual review item to be created
# # #     needs_review = moderation_result.risk_level in ["CRITICAL", "HIGH", "MODERATE"]
    
# # #     if needs_review:
        
# # #         # Determine the final category based on the Ollama output
# # #         risk_level = moderation_result.risk_level
# # #         category = moderation_result.category
# # #         reason = f"{moderation_result.reason} ({risk_level} Risk)"
        
# # #         new_item = ReviewItem(
# # #             id=int(time.time() * 1000),
# # #             sender_id=data.sender_id,
# # #             message_text=data.message_text,
# # #             reason=reason,
# # #             timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
# # #             risk_level=risk_level, 
# # #             category=category
# # #         ).model_dump()

# # #         MANUAL_REVIEW_QUEUE.insert(0, new_item)
# # #         logging.warning(f"🚨 Added to Review Queue: ID {new_item['id']} ({reason})")
# # #         return new_item
    
# # #     # --- NEW LOGIC: Check for SPAM/PHISHING even if it didn't trigger a review (e.g., LOW/CLEAR risk) ---
# # #     if moderation_result.category == "Spam/Phishing":
# # #         logging.warning(f"🚫 Flagged as SPAM: {moderation_result.reason}")
# # #         # Returns a specific "SPAM" status instead of "PASSED"
# # #         return {"status": "SPAM", "reason": moderation_result.reason}


# # #     # If it's LOW or CLEAR risk and not Spam, it passes
# # #     return {"status": "PASSED", "reason": moderation_result.reason}

# # # @app.get("/api/moderation/queue", response_model=List[ReviewItem])
# # # def get_review_queue():
# # #     """Endpoint to retrieve all items in the manual review queue."""
# # #     return MANUAL_REVIEW_QUEUE 

# # # @app.post("/api/moderation/queue/clear")
# # # def clear_review_queue():
# # #     """Endpoint to clear all items from the manual review queue."""
# # #     MANUAL_REVIEW_QUEUE.clear()
    
# # #     MANUAL_REVIEW_QUEUE.append(ReviewItem(
# # #         id=int(time.time() * 1000),
# # #         sender_id="system_message",
# # #         message_text="The queue was manually cleared by a human moderator.",
# # #         reason="Queue Cleared",
# # #         timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
# # #         risk_level="LOW",
# # #         category="Admin Action"
# # #     ).model_dump())

# # #     logging.info("🧹 Manual Review Queue Cleared.")
# # #     return {"status": "success", "message": "Queue cleared."}


# # # # --- Background Scheduler (Mock) - UNCHANGED ---

# # # @app.get("/api/schedule/posts")
# # # def get_scheduled_posts():
# # #     """Get all currently scheduled posts."""
# # #     return {"scheduled_posts": SCHEDULED_POSTS}

# # # async def background_scheduler():
# # #     """This function simulates a persistent background scheduler."""
# # #     logging.info("--- Starting Background Scheduler ---")
# # #     while True:
# # #         await asyncio.sleep(60) 

# # # @app.on_event("startup")
# # # async def startup_event():
# # #     """Starts the background scheduler on API startup."""
# # #     if not hasattr(app.state, 'scheduler_task'):
# # #         app.state.scheduler_task = asyncio.create_task(background_scheduler())

# # # @app.on_event("shutdown")
# # # def shutdown_event():
# # #     """Executes on server shutdown."""
# # #     if hasattr(app.state, 'scheduler_task') and not app.state.scheduler_task.done():
# # #         app.state.scheduler_task.cancel()
# # #     logging.info("Server shutting down.")


# # # # --- MANDATORY SERVER RUN BLOCK ---
# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="127.0.0.1", port=8000)





# # # import time
# # # import asyncio
# # # import json
# # # import logging
# # # from collections import defaultdict # New import for easy counting
# # # from typing import List, Optional, Union
# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from pydantic import BaseModel, Field

# # # # We use 'requests' to talk to the local Ollama API for BOTH content generation and moderation.
# # # import requests 

# # # # --- Configuration ---

# # # # Configure logging to show INFO messages
# # # logging.basicConfig(level=logging.INFO, format='INFO: %(message)s')

# # # # --- Global State and Initialization ---

# # # # --- IMPORTANT LLM CONFIGURATION ---
# # # OLLAMA_MODEL = "mistral" # CHANGE THIS if you pulled a different model (e.g., 'gemma:2b')
# # # OLLAMA_API_URL = "http://localhost:11434/api/generate"
# # # # ---

# # # app = FastAPI(title="Social Media AI Backend (OLLAMA INTEGRATION)", version="0.1.0")

# # # # 1. Moderation Model Initialization (Using Ollama LLM)
# # # logging.info("✅ Moderation is now powered by Ollama LLM for higher precision.")


# # # # 2. In-Memory Data Store 
# # # MANUAL_REVIEW_QUEUE = []
# # # SCHEDULED_POSTS = []

# # # # --- NEW GLOBAL DATA STRUCTURES FOR ANALYTICS ---

# # # # Log of all moderation results for the analytics dashboard
# # # # Stores: {'risk_level': str, 'category': str, 'timestamp': int}
# # # ALL_MODERATION_LOGS = []

# # # # Log of all content generation requests
# # # # Stores: {'platform': str, 'tone': str, 'timestamp': int}
# # # ALL_GENERATION_LOGS = []

# # # # --- END NEW GLOBAL DATA STRUCTURES ---


# # # # Initial mock item to demonstrate HIGH risk due to VERY_NEGATIVE sentiment
# # # MOCK_REVIEW_ITEM = {
# # #     "id": 1678886400000,
# # #     "sender_id": "critical_user_1",
# # #     "message_text": "This product is an absolute scam and I demand a full refund immediately.",
# # #     "reason": "Very Negative Sentiment (HIGH Risk)",
# # #     "timestamp": "2024-03-15T10:00:00Z",
# # #     "risk_level": "HIGH", 
# # #     "category": "Customer Complaint"
# # # }
# # # MANUAL_REVIEW_QUEUE.append(MOCK_REVIEW_ITEM)

# # # # Log the initial mock item so it shows up in the dashboard
# # # ALL_MODERATION_LOGS.append({
# # #     "risk_level": MOCK_REVIEW_ITEM["risk_level"],
# # #     "category": MOCK_REVIEW_ITEM["category"],
# # #     "timestamp": int(time.time() - 3600), # 1 hour ago
# # # })
# # # logging.info("--- NOTE: One mock item added to the Manual Review Queue and Analytics Log for initial testing. ---")


# # # # --- Middleware (CORS) - Unchanged ---
# # # origins = [
# # #     "http://localhost:8000",
# # #     "http://127.0.0.1:8000",
# # #     "http://localhost:5001",
# # #     "http://127.0.0.1:5001",
# # #     "http://localhost:3000", 
# # #     "http://127.0.0.1:3000",
# # # ]

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=origins,
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )


# # # # --- Data Models (Pydantic) ---

# # # class ContentGenerationInput(BaseModel):
# # #     """Data model for the input parameters of content generation."""
# # #     prompt: str
# # #     length: str
# # #     platform: str
# # #     tone: str 

# # # class SocialMediaPost(BaseModel):
# # #     """The JSON structure we expect the LLM to return for generation."""
# # #     post_content: str = Field(..., description="The main text content of the social media post.")
# # #     suggested_hashtags: List[str] = Field(..., description="A list of relevant hashtags for the post.")
# # #     platform: str = Field(..., description="The platform the post was optimized for (e.g., 'facebook', 'twitter').")

# # # class ModerationResult(BaseModel):
# # #     """The STRICT JSON structure we expect the LLM to return for moderation."""
# # #     risk_level: str = Field(..., description="Classification: CRITICAL, HIGH, MODERATE, LOW, or CLEAR.")
# # #     category: str = Field(..., description="Classification: Spam/Phishing, Policy Violation, Customer Complaint, or Clear/Other.")
# # #     reason: str = Field(..., description="A brief, human-readable explanation of why this risk level was assigned.")
    
# # # class WebhookData(BaseModel):
# # #     """Data model for an incoming DM webhook."""
# # #     sender_id: str
# # #     message_text: str

# # # class ReviewItem(BaseModel):
# # #     """Data model for an item in the manual review queue."""
# # #     id: int
# # #     sender_id: str
# # #     message_text: str
# # #     reason: str
# # #     timestamp: str
# # #     risk_level: str
# # #     category: str


# # # # --- Core Logic Functions (OLLAMA INTEGRATION) ---
# # # # ... (generate_ollama_prompt and generate_post_content_with_ollama remain unchanged) ...

# # # def generate_ollama_prompt(data: ContentGenerationInput) -> str:
# # #     """
# # #     Constructs a detailed system instruction prompt for the LLM to ensure
# # #     it returns the content in the required JSON format and adheres to constraints.
# # #     """
# # #     # System Prompt to force JSON output
# # #     system_prompt = (
# # #         "You are an expert social media content creator. Your task is to generate a single social media post "
# # #         "based on the user's request. **You MUST respond only with a valid, clean JSON object** "
# # #         "that adheres to the following exact schema: "
# # #         "{\"post_content\": \"string\", \"suggested_hashtags\": [\"string\"], \"platform\": \"string\"}."
# # #         "Do not include any introductory text, conversation, or markdown formatting (e.g., ```json) outside the JSON object."
# # #         "The generated content must strictly follow the tone, length, and platform constraints."
# # #     )
    
# # #     # User Query containing all parameters
# # #     user_query = (
# # #         f"Generate a post for the topic: '{data.prompt}'. "
# # #         f"The post should be a '{data.length}' length, "
# # #         f"using a '{data.tone}' tone, and optimized for the '{data.platform}' platform. "
# # #         f"Ensure the post content is professional and engaging."
# # #     )
    
# # #     return f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"


# # # def generate_post_content_with_ollama(data: ContentGenerationInput) -> SocialMediaPost:
# # #     """
# # #     Function to call the local Ollama API endpoint for content generation.
# # #     """
# # #     prompt = generate_ollama_prompt(data)
    
# # #     # Ollama REST API Payload
# # #     payload = {
# # #         "model": OLLAMA_MODEL,
# # #         "prompt": prompt,
# # #         "stream": False,
# # #         "options": {
# # #             # Low temperature for deterministic, structured output
# # #             "temperature": 0.3 
# # #         }
# # #     }
    
# # #     try:
# # #         response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
# # #         response.raise_for_status() 

# # #         result = response.json()
# # #         raw_response_text = result.get('response', '').strip()
        
# # #         try:
# # #             post_data = json.loads(raw_response_text)
# # #             validated_post = SocialMediaPost(**post_data)
# # #             return validated_post
        
# # #         except json.JSONDecodeError:
# # #             logging.error(f"Ollama returned malformed JSON for generation: {raw_response_text[:100]}...")
# # #             raise ValueError("LLM returned malformed JSON. Check model and prompt structure.")

# # #     except requests.exceptions.ConnectionError:
# # #         logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
# # #         raise ConnectionError("Could not connect to the local Ollama server. Please check that Ollama is running.")
    
# # #     except requests.exceptions.Timeout:
# # #         raise TimeoutError("Ollama request timed out. Increase timeout or check model performance.")
    
# # #     except Exception as e:
# # #         logging.error(f"An unexpected error occurred during Ollama generation: {e}")
# # #         raise e

# # # def moderate_message_with_ollama(message: str) -> ModerationResult:
# # #     """
# # #     Uses the Ollama LLM to classify a message into one of five risk levels 
# # #     and returns a structured JSON result for moderation.
# # #     """
    
# # #     # System Prompt to force JSON output with specific categories
# # #     system_prompt = (
# # #         "You are an expert social media moderation AI. Your task is to analyze the user message and "
# # #         "classify its risk level and category. **You MUST respond ONLY with a valid, clean JSON object** "
# # #         "that strictly adheres to the schema: "
# # #         "{\"risk_level\": \"string\", \"category\": \"string\", \"reason\": \"string\"}."
# # #         "Do not include any other text or markdown formatting outside the JSON object."
        
# # #         "\n\n--- CLASSIFICATION RULES ---"
# # #         "\nRISK_LEVEL must be one of: CRITICAL, HIGH, MODERATE, LOW, or CLEAR."
# # #         "\nCATEGORY must be one of: Spam/Phishing, Policy Violation, Customer Complaint, Clear/Other."
        
# # #         "\n\nCRITICAL Risk: Direct threats, hate speech, illegal content, or obvious spam/phishing attempts. This requires immediate human review."
# # #         "\nHIGH Risk: Very aggressive or highly offensive language, extreme negative sentiment (e.g., 'scam', 'fraud'). This requires human review."
# # #         "\nMODERATE Risk: Strong negative sentiment (e.g., 'bad', 'terrible'), minor policy violations. This might require human review."
# # #         "\nLOW Risk: Neutral or slightly positive sentiment, very mild or ambiguous concerns. This usually passes."
# # #         "\nCLEAR Risk: Clearly positive or neutral, non-concerning messages. This always passes."
# # #     )
    
# # #     user_query = f"Analyze and classify the following message:\n'{message}'"
# # #     prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"
    
# # #     payload = {
# # #         "model": OLLAMA_MODEL,
# # #         "prompt": prompt,
# # #         "stream": False,
# # #         "options": {
# # #             # Low temperature for deterministic, structured output
# # #             "temperature": 0.2
# # #         }
# # #     }

# # #     try:
# # #         response = requests.post(OLLAMA_API_URL, json=payload, timeout=45) # Increased timeout for moderation
# # #         response.raise_for_status()

# # #         result = response.json()
# # #         raw_response_text = result.get('response', '').strip()

# # #         try:
# # #             # Attempt to parse the raw text directly as JSON
# # #             moderation_data = json.loads(raw_response_text)
            
# # #             # Use Pydantic to validate the structure
# # #             validated_result = ModerationResult(**moderation_data)
# # #             return validated_result
        
# # #         except json.JSONDecodeError:
# # #             logging.error(f"Ollama returned malformed JSON for moderation: {raw_response_text[:100]}...")
# # #             # Fallback to a high risk if the LLM response is unparseable
# # #             return ModerationResult(
# # #                 risk_level="HIGH",
# # #                 category="System Error",
# # #                 reason="LLM response malformed, manual review required."
# # #             )
        
# # #     except requests.exceptions.ConnectionError:
# # #         logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
# # #         # Fallback to high risk on connection failure
# # #         return ModerationResult(
# # #             risk_level="HIGH",
# # #             category="System Error",
# # #             reason="Ollama connection failed, manual review required."
# # #         )
# # #     except Exception as e:
# # #         logging.error(f"An unexpected error occurred during Ollama moderation: {e}")
# # #         # Fallback to high risk on any other failure
# # #         return ModerationResult(
# # #             risk_level="HIGH",
# # #             category="System Error",
# # #             reason=f"Unexpected error: {str(e)[:50]}"
# # #         )


# # # # --- API Endpoints ---

# # # @app.get("/")
# # # def read_root():
# # #     return {"status": "AI Social Media Backend is Running (OLLAMA INTEGRATION ACTIVE)"}

# # # # --- Content Generation Endpoint (OLLAMA ACTIVE) ---
# # # @app.post("/api/content/generate", response_model=SocialMediaPost)
# # # async def generate_post_content(data: ContentGenerationInput):
# # #     """Generates social media content using the local Ollama LLM."""
    
# # #     logging.info(f"⚡ OLLAMA generation requested for {data.platform} post (Topic: {data.prompt})...")

# # #     loop = asyncio.get_event_loop()
    
# # #     try:
# # #         validated_post = await loop.run_in_executor(
# # #             None, 
# # #             generate_post_content_with_ollama, 
# # #             data
# # #         )
        
# # #         # --- ANALYTICS LOGGING: GENERATION ---
# # #         ALL_GENERATION_LOGS.append({
# # #             "platform": data.platform,
# # #             "tone": data.tone,
# # #             "timestamp": int(time.time()),
# # #         })
# # #         # -----------------------------------

# # #         logging.info(f"🎉 Ollama Content Generated Successfully for {validated_post.platform}.")
# # #         return validated_post

# # #     except (ConnectionError, TimeoutError) as e:
# # #         raise HTTPException(
# # #             status_code=503, 
# # #             detail=f"Local LLM Error: {e}"
# # #         )
# # #     except Exception as e:
# # #         raise HTTPException(status_code=500, detail=f"Failed to generate content: {e}")


# # # # --- Moderation & Review Endpoints (Updated) ---

# # # @app.post("/api/moderation/simulate", response_model=Union[ReviewItem, dict])
# # # async def dm_webhook(data: WebhookData):
# # #     """Real-time processing of incoming DMs for moderation using Ollama."""
# # #     logging.info(f"--- DM Received: \"{data.message_text[:40]}...\" from {data.sender_id} ---")

# # #     # Use a thread pool to run the synchronous Ollama call for moderation
# # #     loop = asyncio.get_event_loop()
# # #     moderation_result = await loop.run_in_executor(
# # #         None, 
# # #         moderate_message_with_ollama, 
# # #         data.message_text
# # #     )

# # #     # Risk levels that require a manual review item to be created
# # #     needs_review = moderation_result.risk_level in ["CRITICAL", "HIGH", "MODERATE"]

# # #     # --- ANALYTICS LOGGING: MODERATION (Log ALL events) ---
# # #     ALL_MODERATION_LOGS.append({
# # #         "risk_level": moderation_result.risk_level,
# # #         "category": moderation_result.category,
# # #         "timestamp": int(time.time()),
# # #     })
# # #     # -----------------------------------
    
# # #     if needs_review:
        
# # #         # Determine the final category based on the Ollama output
# # #         risk_level = moderation_result.risk_level
# # #         category = moderation_result.category
# # #         reason = f"{moderation_result.reason} ({risk_level} Risk)"
        
# # #         new_item = ReviewItem(
# # #             id=int(time.time() * 1000),
# # #             sender_id=data.sender_id,
# # #             message_text=data.message_text,
# # #             reason=reason,
# # #             timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
# # #             risk_level=risk_level, 
# # #             category=category
# # #         ).model_dump()

# # #         MANUAL_REVIEW_QUEUE.insert(0, new_item)
# # #         logging.warning(f"🚨 Added to Review Queue: ID {new_item['id']} ({reason})")
# # #         return new_item
    
# # #     # --- NEW LOGIC: Check for SPAM/PHISHING even if it didn't trigger a review (e.g., LOW/CLEAR risk) ---
# # #     if moderation_result.category == "Spam/Phishing":
# # #         logging.warning(f"🚫 Flagged as SPAM: {moderation_result.reason}")
# # #         # Returns a specific "SPAM" status instead of "PASSED"
# # #         return {"status": "SPAM", "reason": moderation_result.reason}


# # #     # If it's LOW or CLEAR risk and not Spam, it passes
# # #     return {"status": "PASSED", "reason": moderation_result.reason}

# # # @app.get("/api/moderation/queue", response_model=List[ReviewItem])
# # # def get_review_queue():
# # #     """Endpoint to retrieve all items in the manual review queue."""
# # #     return MANUAL_REVIEW_QUEUE 

# # # @app.post("/api/moderation/queue/clear")
# # # def clear_review_queue():
# # #     """Endpoint to clear all items from the manual review queue."""
# # #     MANUAL_REVIEW_QUEUE.clear()
    
# # #     MANUAL_REVIEW_QUEUE.append(ReviewItem(
# # #         id=int(time.time() * 1000),
# # #         sender_id="system_message",
# # #         message_text="The queue was manually cleared by a human moderator.",
# # #         reason="Queue Cleared",
# # #         timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
# # #         risk_level="LOW",
# # #         category="Admin Action"
# # #     ).model_dump())

# # #     logging.info("🧹 Manual Review Queue Cleared.")
# # #     return {"status": "success", "message": "Queue cleared."}


# # # # --- NEW ANALYTICS AGGREGATION ENDPOINT ---

# # # @app.get("/api/analytics/summary")
# # # def get_analytics_summary():
# # #     """Aggregates all logs into a summary for the dashboard."""
    
# # #     # --- MODERATION METRICS ---
    
# # #     total_processed = len(ALL_MODERATION_LOGS)
    
# # #     risk_counts = defaultdict(int)
# # #     category_counts = defaultdict(int)
    
# # #     for log in ALL_MODERATION_LOGS:
# # #         # Standardizing risk levels for counting, useful if the LLM sometimes returns variations
# # #         risk = log['risk_level'].upper()
# # #         category = log['category']
        
# # #         # Count all occurrences
# # #         risk_counts[risk] += 1
# # #         category_counts[category] += 1
        
# # #     # Calculate flag rate (CRITICAL, HIGH, MODERATE, any explicit SPAM)
# # #     flagged_risk_levels = ['CRITICAL', 'HIGH', 'MODERATE', 'SPAM']
# # #     flagged_count = sum(count for level, count in risk_counts.items() if level in flagged_risk_levels)
    
# # #     flag_rate = (flagged_count / total_processed * 100) if total_processed > 0 else 0
    
# # #     # --- GENERATION METRICS ---
    
# # #     total_generated = len(ALL_GENERATION_LOGS)
# # #     platform_counts = defaultdict(int)
# # #     tone_counts = defaultdict(int)

# # #     for log in ALL_GENERATION_LOGS:
# # #         platform_counts[log['platform']] += 1
# # #         tone_counts[log['tone']] += 1

# # #     return {
# # #         "moderation_stats": {
# # #             "total_processed": total_processed,
# # #             "flag_rate_percent": round(flag_rate, 2),
# # #             "review_queue_size": len(MANUAL_REVIEW_QUEUE),
# # #             # Convert defaultdicts to dict for JSON serialization
# # #             "risk_counts": dict(risk_counts),
# # #             "category_counts": dict(category_counts)
# # #         },
# # #         "generation_stats": {
# # #             "total_generated": total_generated,
# # #             "platform_counts": dict(platform_counts),
# # #             "tone_counts": dict(tone_counts)
# # #         }
# # #     }

# # # # --- END NEW ANALYTICS AGGREGATION ENDPOINT ---

# # # # --- Background Scheduler (Mock) - UNCHANGED ---

# # # @app.get("/api/schedule/posts")
# # # def get_scheduled_posts():
# # #     """Get all currently scheduled posts."""
# # #     return {"scheduled_posts": SCHEDULED_POSTS}

# # # async def background_scheduler():
# # #     """This function simulates a persistent background scheduler."""
# # #     logging.info("--- Starting Background Scheduler ---")
# # #     while True:
# # #         await asyncio.sleep(60) 

# # # @app.on_event("startup")
# # # async def startup_event():
# # #     """Starts the background scheduler on API startup."""
# # #     if not hasattr(app.state, 'scheduler_task'):
# # #         app.state.scheduler_task = asyncio.create_task(background_scheduler())

# # # @app.on_event("shutdown")
# # # def shutdown_event():
# # #     """Executes on server shutdown."""
# # #     if hasattr(app.state, 'scheduler_task') and not app.state.scheduler_task.done():
# # #         app.state.scheduler_task.cancel()
# # #     logging.info("Server shutting down.")


# # # # --- MANDATORY SERVER RUN BLOCK ---
# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="127.0.0.1", port=8000)




# # import time
# # import asyncio
# # import json
# # import logging
# # from collections import defaultdict
# # from typing import List, Optional, Union
# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel, Field

# # # We use 'requests' to talk to the local Ollama API for BOTH content generation and moderation.
# # import requests 

# # # --- Configuration ---

# # # Configure logging to show INFO messages
# # logging.basicConfig(level=logging.INFO, format='INFO: %(message)s')

# # # --- Global State and Initialization ---

# # # --- IMPORTANT LLM CONFIGURATION ---
# # OLLAMA_MODEL = "mistral" # CHANGE THIS if you pulled a different model (e.g., 'gemma:2b')
# # OLLAMA_API_URL = "http://localhost:11434/api/generate"
# # # ---

# # app = FastAPI(title="Social Media AI Backend (OLLAMA INTEGRATION)", version="0.1.0")

# # # 1. Moderation Model Initialization (Using Ollama LLM)
# # logging.info("✅ Moderation is now powered by Ollama LLM for higher precision.")


# # # 2. In-Memory Data Store 
# # MANUAL_REVIEW_QUEUE = []
# # SCHEDULED_POSTS = []

# # # --- NEW GLOBAL DATA STRUCTURES FOR ANALYTICS ---

# # # Log of all moderation results for the analytics dashboard
# # # Stores: {'risk_level': str, 'category': str, 'timestamp': int}
# # ALL_MODERATION_LOGS = []

# # # Log of all content generation requests
# # # Stores: {'platform': str, 'tone': str, 'timestamp': int}
# # ALL_GENERATION_LOGS = []

# # # --- END NEW GLOBAL DATA STRUCTURES ---


# # # Initial mock item to demonstrate HIGH risk due to VERY_NEGATIVE sentiment
# # MOCK_REVIEW_ITEM = {
# #     "id": 1678886400000,
# #     "sender_id": "critical_user_1",
# #     "message_text": "This product is an absolute scam and I demand a full refund immediately.",
# #     "reason": "Very Negative Sentiment (HIGH Risk)",
# #     "timestamp": "2024-03-15T10:00:00Z",
# #     "risk_level": "HIGH", 
# #     "category": "Customer Complaint"
# # }
# # MANUAL_REVIEW_QUEUE.append(MOCK_REVIEW_ITEM)

# # # Log the initial mock item so it shows up in the dashboard
# # ALL_MODERATION_LOGS.append({
# #     "risk_level": MOCK_REVIEW_ITEM["risk_level"],
# #     "category": MOCK_REVIEW_ITEM["category"],
# #     "timestamp": int(time.time() - 3600), # 1 hour ago
# # })
# # logging.info("--- NOTE: One mock item added to the Manual Review Queue and Analytics Log for initial testing. ---")


# # # --- Middleware (CORS) - CRITICAL FIX ---
# # # Added the frontend server addresses (5500) to allow communication.
# # origins = [
# #     "http://localhost:8000",
# #     "http://127.0.0.1:8000",
# #     "http://localhost:5500",      # <-- NEW: Frontend local server address
# #     "http://127.0.0.1:5500",    # <-- NEW: Frontend local server address
# #     "http://localhost:5001",
# #     "http://127.0.0.1:5001",
# #     "http://localhost:3000", 
# #     "http://127.0.0.1:3000",
# # ]

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=origins,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )
# # # --- END CRITICAL FIX ---


# # # --- Data Models (Pydantic) ---

# # class ContentGenerationInput(BaseModel):
# #     """Data model for the input parameters of content generation."""
# #     prompt: str
# #     length: str
# #     platform: str
# #     tone: str 

# # class SocialMediaPost(BaseModel):
# #     """The JSON structure we expect the LLM to return for generation."""
# #     post_content: str = Field(..., description="The main text content of the social media post.")
# #     suggested_hashtags: List[str] = Field(..., description="A list of relevant hashtags for the post.")
# #     platform: str = Field(..., description="The platform the post was optimized for (e.g., 'facebook', 'twitter').")

# # class ModerationResult(BaseModel):
# #     """The STRICT JSON structure we expect the LLM to return for moderation."""
# #     risk_level: str = Field(..., description="Classification: CRITICAL, HIGH, MODERATE, LOW, or CLEAR.")
# #     category: str = Field(..., description="Classification: Spam/Phishing, Policy Violation, Customer Complaint, or Clear/Other.")
# #     reason: str = Field(..., description="A brief, human-readable explanation of why this risk level was assigned.")
    
# # class WebhookData(BaseModel):
# #     """Data model for an incoming DM webhook."""
# #     sender_id: str
# #     message_text: str

# # class ReviewItem(BaseModel):
# #     """Data model for an item in the manual review queue."""
# #     id: int
# #     sender_id: str
# #     message_text: str
# #     reason: str
# #     timestamp: str
# #     risk_level: str
# #     category: str


# # # --- Core Logic Functions (OLLAMA INTEGRATION) ---
# # # ... (generate_ollama_prompt and generate_post_content_with_ollama remain unchanged) ...

# # def generate_ollama_prompt(data: ContentGenerationInput) -> str:
# #     """
# #     Constructs a detailed system instruction prompt for the LLM to ensure
# #     it returns the content in the required JSON format and adheres to constraints.
# #     """
# #     # System Prompt to force JSON output
# #     system_prompt = (
# #         "You are an expert social media content creator. Your task is to generate a single social media post "
# #         "based on the user's request. **You MUST respond only with a valid, clean JSON object** "
# #         "that adheres to the following exact schema: "
# #         "{\"post_content\": \"string\", \"suggested_hashtags\": [\"string\"], \"platform\": \"string\"}."
# #         "Do not include any introductory text, conversation, or markdown formatting (e.g., ```json) outside the JSON object."
# #         "The generated content must strictly follow the tone, length, and platform constraints."
# #     )
    
# #     # User Query containing all parameters
# #     user_query = (
# #         f"Generate a post for the topic: '{data.prompt}'. "
# #         f"The post should be a '{data.length}' length, "
# #         f"using a '{data.tone}' tone, and optimized for the '{data.platform}' platform. "
# #         f"Ensure the post content is professional and engaging."
# #     )
    
# #     return f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"


# # def generate_post_content_with_ollama(data: ContentGenerationInput) -> SocialMediaPost:
# #     """
# #     Function to call the local Ollama API endpoint for content generation.
# #     """
# #     prompt = generate_ollama_prompt(data)
    
# #     # Ollama REST API Payload
# #     payload = {
# #         "model": OLLAMA_MODEL,
# #         "prompt": prompt,
# #         "stream": False,
# #         "options": {
# #             # Low temperature for deterministic, structured output
# #             "temperature": 0.3 
# #         }
# #     }
    
# #     try:
# #         response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
# #         response.raise_for_status() 

# #         result = response.json()
# #         raw_response_text = result.get('response', '').strip()
        
# #         try:
# #             post_data = json.loads(raw_response_text)
# #             validated_post = SocialMediaPost(**post_data)
# #             return validated_post
        
# #         except json.JSONDecodeError:
# #             logging.error(f"Ollama returned malformed JSON for generation: {raw_response_text[:100]}...")
# #             raise ValueError("LLM returned malformed JSON. Check model and prompt structure.")

# #     except requests.exceptions.ConnectionError:
# #         logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
# #         raise ConnectionError("Could not connect to the local Ollama server. Please check that Ollama is running.")
    
# #     except requests.exceptions.Timeout:
# #         raise TimeoutError("Ollama request timed out. Increase timeout or check model performance.")
    
# #     except Exception as e:
# #         logging.error(f"An unexpected error occurred during Ollama generation: {e}")
# #         raise e

# # def moderate_message_with_ollama(message: str) -> ModerationResult:
# #     """
# #     Uses the Ollama LLM to classify a message into one of five risk levels 
# #     and returns a structured JSON result for moderation.
# #     """
    
# #     # System Prompt to force JSON output with specific categories
# #     system_prompt = (
# #         "You are an expert social media moderation AI. Your task is to analyze the user message and "
# #         "classify its risk level and category. **You MUST respond ONLY with a valid, clean JSON object** "
# #         "that strictly adheres to the schema: "
# #         "{\"risk_level\": \"string\", \"category\": \"string\", \"reason\": \"string\"}."
# #         "Do not include any other text or markdown formatting outside the JSON object."
        
# #         "\n\n--- CLASSIFICATION RULES ---"
# #         "\nRISK_LEVEL must be one of: CRITICAL, HIGH, MODERATE, LOW, or CLEAR."
# #         "\nCATEGORY must be one of: Spam/Phishing, Policy Violation, Customer Complaint, Clear/Other."
        
# #         "\n\nCRITICAL Risk: Direct threats, hate speech, illegal content, or obvious spam/phishing attempts. This requires immediate human review."
# #         "\nHIGH Risk: Very aggressive or highly offensive language, extreme negative sentiment (e.g., 'scam', 'fraud'). This requires human review."
# #         "\nMODERATE Risk: Strong negative sentiment (e.g., 'bad', 'terrible'), minor policy violations. This might require human review."
# #         "\nLOW Risk: Neutral or slightly positive sentiment, very mild or ambiguous concerns. This usually passes."
# #         "\nCLEAR Risk: Clearly positive or neutral, non-concerning messages. This always passes."
# #     )
    
# #     user_query = f"Analyze and classify the following message:\n'{message}'"
# #     prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"
    
# #     payload = {
# #         "model": OLLAMA_MODEL,
# #         "prompt": prompt,
# #         "stream": False,
# #         "options": {
# #             # Low temperature for deterministic, structured output
# #             "temperature": 0.2
# #         }
# #     }

# #     try:
# #         response = requests.post(OLLAMA_API_URL, json=payload, timeout=45) # Increased timeout for moderation
# #         response.raise_for_status()

# #         result = response.json()
# #         raw_response_text = result.get('response', '').strip()

# #         try:
# #             # Attempt to parse the raw text directly as JSON
# #             moderation_data = json.loads(raw_response_text)
            
# #             # Use Pydantic to validate the structure
# #             validated_result = ModerationResult(**moderation_data)
# #             return validated_result
        
# #         except json.JSONDecodeError:
# #             logging.error(f"Ollama returned malformed JSON for moderation: {raw_response_text[:100]}...")
# #             # Fallback to a high risk if the LLM response is unparseable
# #             return ModerationResult(
# #                 risk_level="HIGH",
# #                 category="System Error",
# #                 reason="LLM response malformed, manual review required."
# #             )
        
# #     except requests.exceptions.ConnectionError:
# #         logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
# #         # Fallback to high risk on connection failure
# #         return ModerationResult(
# #             risk_level="HIGH",
# #             category="System Error",
# #             reason="Ollama connection failed, manual review required."
# #         )
# #     except Exception as e:
# #         logging.error(f"An unexpected error occurred during Ollama moderation: {e}")
# #         # Fallback to high risk on any other failure
# #         return ModerationResult(
# #             risk_level="HIGH",
# #             category="System Error",
# #             reason=f"Unexpected error: {str(e)[:50]}"
# #         )


# # # --- API Endpoints ---

# # @app.get("/")
# # def read_root():
# #     return {"status": "AI Social Media Backend is Running (OLLAMA INTEGRATION ACTIVE)"}

# # # --- Content Generation Endpoint (OLLAMA ACTIVE) ---
# # @app.post("/api/content/generate", response_model=SocialMediaPost)
# # async def generate_post_content(data: ContentGenerationInput):
# #     """Generates social media content using the local Ollama LLM."""
    
# #     logging.info(f"⚡ OLLAMA generation requested for {data.platform} post (Topic: {data.prompt})...")

# #     loop = asyncio.get_event_loop()
    
# #     try:
# #         validated_post = await loop.run_in_executor(
# #             None, 
# #             generate_post_content_with_ollama, 
# #             data
# #         )
        
# #         # --- ANALYTICS LOGGING: GENERATION ---
# #         ALL_GENERATION_LOGS.append({
# #             "platform": data.platform,
# #             "tone": data.tone,
# #             "timestamp": int(time.time()),
# #         })
# #         # -----------------------------------

# #         logging.info(f"🎉 Ollama Content Generated Successfully for {validated_post.platform}.")
# #         return validated_post

# #     except (ConnectionError, TimeoutError) as e:
# #         raise HTTPException(
# #             status_code=503, 
# #             detail=f"Local LLM Error: {e}"
# #         )
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Failed to generate content: {e}")


# # # --- Moderation & Review Endpoints (Updated) ---

# # @app.post("/api/moderation/simulate", response_model=Union[ReviewItem, dict])
# # async def dm_webhook(data: WebhookData):
# #     """Real-time processing of incoming DMs for moderation using Ollama."""
# #     logging.info(f"--- DM Received: \"{data.message_text[:40]}...\" from {data.sender_id} ---")

# #     # Use a thread pool to run the synchronous Ollama call for moderation
# #     loop = asyncio.get_event_loop()
# #     moderation_result = await loop.run_in_executor(
# #         None, 
# #         moderate_message_with_ollama, 
# #         data.message_text
# #     )

# #     # Risk levels that require a manual review item to be created
# #     needs_review = moderation_result.risk_level in ["CRITICAL", "HIGH", "MODERATE"]

# #     # --- ANALYTICS LOGGING: MODERATION (Log ALL events) ---
# #     ALL_MODERATION_LOGS.append({
# #         "risk_level": moderation_result.risk_level,
# #         "category": moderation_result.category,
# #         "timestamp": int(time.time()),
# #     })
# #     # -----------------------------------
    
# #     if needs_review:
        
# #         # Determine the final category based on the Ollama output
# #         risk_level = moderation_result.risk_level
# #         category = moderation_result.category
# #         reason = f"{moderation_result.reason} ({risk_level} Risk)"
        
# #         new_item = ReviewItem(
# #             id=int(time.time() * 1000),
# #             sender_id=data.sender_id,
# #             message_text=data.message_text,
# #             reason=reason,
# #             timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
# #             risk_level=risk_level, 
# #             category=category
# #         ).model_dump()

# #         MANUAL_REVIEW_QUEUE.insert(0, new_item)
# #         logging.warning(f"🚨 Added to Review Queue: ID {new_item['id']} ({reason})")
# #         return new_item
    
# #     # --- NEW LOGIC: Check for SPAM/PHISHING even if it didn't trigger a review (e.g., LOW/CLEAR risk) ---
# #     if moderation_result.category == "Spam/Phishing":
# #         logging.warning(f"🚫 Flagged as SPAM: {moderation_result.reason}")
# #         # Returns a specific "SPAM" status instead of "PASSED"
# #         return {"status": "SPAM", "reason": moderation_result.reason}


# #     # If it's LOW or CLEAR risk and not Spam, it passes
# #     return {"status": "PASSED", "reason": moderation_result.reason}

# # @app.get("/api/moderation/queue", response_model=List[ReviewItem])
# # def get_review_queue():
# #     """Endpoint to retrieve all items in the manual review queue."""
# #     return MANUAL_REVIEW_QUEUE 

# # @app.post("/api/moderation/queue/clear")
# # def clear_review_queue():
# #     """Endpoint to clear all items from the manual review queue."""
# #     MANUAL_REVIEW_QUEUE.clear()
    
# #     MANUAL_REVIEW_QUEUE.append(ReviewItem(
# #         id=int(time.time() * 1000),
# #         sender_id="system_message",
# #         message_text="The queue was manually cleared by a human moderator.",
# #         reason="Queue Cleared",
# #         timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
# #         risk_level="LOW",
# #         category="Admin Action"
# #     ).model_dump())

# #     logging.info("🧹 Manual Review Queue Cleared.")
# #     return {"status": "success", "message": "Queue cleared."}


# # # --- NEW ANALYTICS AGGREGATION ENDPOINT ---

# # @app.get("/api/analytics/summary")
# # def get_analytics_summary():
# #     """Aggregates all logs into a summary for the dashboard."""
    
# #     # --- MODERATION METRICS ---
    
# #     total_processed = len(ALL_MODERATION_LOGS)
    
# #     risk_counts = defaultdict(int)
# #     category_counts = defaultdict(int)
    
# #     for log in ALL_MODERATION_LOGS:
# #         # Standardizing risk levels for counting, useful if the LLM sometimes returns variations
# #         risk = log['risk_level'].upper()
# #         category = log['category']
        
# #         # Count all occurrences
# #         risk_counts[risk] += 1
# #         category_counts[category] += 1
        
# #     # Calculate flag rate (CRITICAL, HIGH, MODERATE, any explicit SPAM)
# #     flagged_risk_levels = ['CRITICAL', 'HIGH', 'MODERATE', 'SPAM']
# #     flagged_count = sum(count for level, count in risk_counts.items() if level in flagged_risk_levels)
    
# #     flag_rate = (flagged_count / total_processed * 100) if total_processed > 0 else 0
    
# #     # --- GENERATION METRICS ---
    
# #     total_generated = len(ALL_GENERATION_LOGS)
# #     platform_counts = defaultdict(int)
# #     tone_counts = defaultdict(int)

# #     for log in ALL_GENERATION_LOGS:
# #         platform_counts[log['platform']] += 1
# #         tone_counts[log['tone']] += 1

# #     return {
# #         "moderation_stats": {
# #             "total_processed": total_processed,
# #             "flag_rate_percent": round(flag_rate, 2),
# #             "review_queue_size": len(MANUAL_REVIEW_QUEUE),
# #             # Convert defaultdicts to dict for JSON serialization
# #             "risk_counts": dict(risk_counts),
# #             "category_counts": dict(category_counts)
# #         },
# #         "generation_stats": {
# #             "total_generated": total_generated,
# #             "platform_counts": dict(platform_counts),
# #             "tone_counts": dict(tone_counts)
# #         }
# #     }

# # # --- END NEW ANALYTICS AGGREGATION ENDPOINT ---

# # # --- Background Scheduler (Mock) - UNCHANGED ---

# # @app.get("/api/schedule/posts")
# # def get_scheduled_posts():
# #     """Get all currently scheduled posts."""
# #     return {"scheduled_posts": SCHEDULED_POSTS}

# # async def background_scheduler():
# #     """This function simulates a persistent background scheduler."""
# #     logging.info("--- Starting Background Scheduler ---")
# #     while True:
# #         await asyncio.sleep(60) 

# # @app.on_event("startup")
# # async def startup_event():
# #     """Starts the background scheduler on API startup."""
# #     if not hasattr(app.state, 'scheduler_task'):
# #         app.state.scheduler_task = asyncio.create_task(background_scheduler())

# # @app.on_event("shutdown")
# # def shutdown_event():
# #     """Executes on server shutdown."""
# #     if hasattr(app.state, 'scheduler_task') and not app.state.scheduler_task.done():
# #         app.state.scheduler_task.cancel()
# #     logging.info("Server shutting down.")


# # # --- MANDATORY SERVER RUN BLOCK ---
# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="127.0.0.1", port=8000)


# import time
# import asyncio
# import json
# import logging
# from collections import defaultdict
# from typing import List, Optional, Union
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field

# # We use 'requests' to talk to the local Ollama API for BOTH content generation and moderation.
# import requests 

# # --- Configuration ---

# # Configure logging to show INFO messages
# logging.basicConfig(level=logging.INFO, format='INFO: %(message)s')

# # --- Global State and Initialization ---

# # --- IMPORTANT LLM CONFIGURATION ---
# OLLAMA_MODEL = "mistral" # CHANGE THIS if you pulled a different model (e.g., 'gemma:2b')
# OLLAMA_API_URL = "http://localhost:11434/api/generate"
# # ---

# app = FastAPI(title="Social Media AI Backend (OLLAMA INTEGRATION)", version="0.1.0")

# # 1. Moderation Model Initialization (Using Ollama LLM)
# logging.info("✅ Moderation is now powered by Ollama LLM for higher precision.")


# # 2. In-Memory Data Store 
# MANUAL_REVIEW_QUEUE = []
# SCHEDULED_POSTS = []

# # --- NEW GLOBAL DATA STRUCTURES FOR ANALYTICS ---

# # Log of all moderation results for the analytics dashboard
# # Stores: {'risk_level': str, 'category': str, 'timestamp': int}
# ALL_MODERATION_LOGS = []

# # Log of all content generation requests
# # Stores: {'platform': str, 'tone': str, 'timestamp': int}
# ALL_GENERATION_LOGS = []

# # --- END NEW GLOBAL DATA STRUCTURES ---


# # Initial mock item to demonstrate HIGH risk due to VERY_NEGATIVE sentiment
# MOCK_REVIEW_ITEM = {
#     "id": 1678886400000,
#     "sender_id": "critical_user_1",
#     "message_text": "This product is an absolute scam and I demand a full refund immediately.",
#     "reason": "Very Negative Sentiment (HIGH Risk)",
#     "timestamp": "2024-03-15T10:00:00Z",
#     "risk_level": "HIGH", 
#     "category": "Customer Complaint"
# }
# MANUAL_REVIEW_QUEUE.append(MOCK_REVIEW_ITEM)

# # Log the initial mock item so it shows up in the dashboard
# ALL_MODERATION_LOGS.append({
#     "risk_level": MOCK_REVIEW_ITEM["risk_level"],
#     "category": MOCK_REVIEW_ITEM["category"],
#     "timestamp": int(time.time() - 3600), # 1 hour ago
# })
# logging.info("--- NOTE: One mock item added to the Manual Review Queue and Analytics Log for initial testing. ---")


# # --- Middleware (CORS) - CRITICAL FIX ---
# # Added the frontend server addresses (5500) to allow communication.
# origins = [
#     "http://localhost:8000",
#     "http://127.0.0.1:8000", # <-- FIX: Added 127.0.0.1:8000 for local uvicorn execution
#     "http://localhost:5500",      
#     "http://127.0.0.1:5500",    
#     "http://localhost:5001",
#     "http://127.0.0.1:5001",
#     "http://localhost:3000", 
#     "http://127.0.0.1:3000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # --- END CRITICAL FIX ---


# # --- Data Models (Pydantic) ---

# class ContentGenerationInput(BaseModel):
#     """Data model for the input parameters of content generation."""
#     prompt: str
#     length: str
#     platform: str
#     tone: str 

# class SocialMediaPost(BaseModel):
#     """The JSON structure we expect the LLM to return for generation."""
#     post_content: str = Field(..., description="The main text content of the social media post.")
#     suggested_hashtags: List[str] = Field(..., description="A list of relevant hashtags for the post.")
#     platform: str = Field(..., description="The platform the post was optimized for (e.g., 'facebook', 'twitter').")
#     # Added field to support scheduling from frontend
#     created_at: Optional[str] = Field(None, description="Timestamp when the post was created/scheduled.")

# class ModerationResult(BaseModel):
#     """The STRICT JSON structure we expect the LLM to return for moderation."""
#     risk_level: str = Field(..., description="Classification: CRITICAL, HIGH, MODERATE, LOW, or CLEAR.")
#     category: str = Field(..., description="Classification: Spam/Phishing, Policy Violation, Customer Complaint, or Clear/Other.")
#     reason: str = Field(..., description="A brief, human-readable explanation of why this risk level was assigned.")
    
# class WebhookData(BaseModel):
#     """Data model for an incoming DM webhook."""
#     sender_id: str
#     message_text: str

# class ReviewItem(BaseModel):
#     """Data model for an item in the manual review queue."""
#     id: int
#     sender_id: str
#     message_text: str
#     reason: str
#     timestamp: str
#     risk_level: str
#     category: str


# # --- Core Logic Functions (OLLAMA INTEGRATION) ---

# def generate_ollama_prompt(data: ContentGenerationInput) -> str:
#     """
#     Constructs a detailed system instruction prompt for the LLM to ensure
#     it returns the content in the required JSON format and adheres to constraints.
#     """
#     # System Prompt to force JSON output
#     system_prompt = (
#         "You are an expert social media content creator. Your task is to generate a single social media post "
#         "based on the user's request. **You MUST respond only with a valid, clean JSON object** "
#         "that adheres to the following exact schema: "
#         "{\"post_content\": \"string\", \"suggested_hashtags\": [\"string\"], \"platform\": \"string\"}."
#         "Do not include any introductory text, conversation, or markdown formatting (e.g., ```json) outside the JSON object."
#         "The generated content must strictly follow the tone, length, and platform constraints."
#     )
    
#     # User Query containing all parameters
#     user_query = (
#         f"Generate a post for the topic: '{data.prompt}'. "
#         f"The post should be a '{data.length}' length, "
#         f"using a '{data.tone}' tone, and optimized for the '{data.platform}' platform. "
#         f"Ensure the post content is professional and engaging."
#     )
    
#     return f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"


# def generate_post_content_with_ollama(data: ContentGenerationInput) -> SocialMediaPost:
#     """
#     Function to call the local Ollama API endpoint for content generation.
#     """
#     prompt = generate_ollama_prompt(data)
    
#     # Ollama REST API Payload
#     payload = {
#         "model": OLLAMA_MODEL,
#         "prompt": prompt,
#         "stream": False,
#         "options": {
#             # Low temperature for deterministic, structured output
#             "temperature": 0.3 
#         }
#     }
    
#     try:
#         response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
#         response.raise_for_status() 

#         result = response.json()
#         raw_response_text = result.get('response', '').strip()
        
#         try:
#             post_data = json.loads(raw_response_text)
#             # Add creation timestamp before validation
#             post_data['created_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
#             validated_post = SocialMediaPost(**post_data)
#             return validated_post
        
#         except json.JSONDecodeError:
#             logging.error(f"Ollama returned malformed JSON for generation: {raw_response_text[:100]}...")
#             raise ValueError("LLM returned malformed JSON. Check model and prompt structure.")

#     except requests.exceptions.ConnectionError:
#         logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
#         raise ConnectionError("Could not connect to the local Ollama server. Please check that Ollama is running.")
    
#     except requests.exceptions.Timeout:
#         raise TimeoutError("Ollama request timed out. Increase timeout or check model performance.")
    
#     except Exception as e:
#         logging.error(f"An unexpected error occurred during Ollama generation: {e}")
#         raise e

# def moderate_message_with_ollama(message: str) -> ModerationResult:
#     """
#     Uses the Ollama LLM to classify a message into one of five risk levels 
#     and returns a structured JSON result for moderation.
#     """
    
#     # System Prompt to force JSON output with specific categories
#     system_prompt = (
#         "You are an expert social media moderation AI. Your task is to analyze the user message and "
#         "classify its risk level and category. **You MUST respond ONLY with a valid, clean JSON object** "
#         "that strictly adheres to the schema: "
#         "{\"risk_level\": \"string\", \"category\": \"string\", \"reason\": \"string\"}."
#         "Do not include any other text or markdown formatting outside the JSON object."
        
#         "\n\n--- CLASSIFICATION RULES ---"
#         "\nRISK_LEVEL must be one of: CRITICAL, HIGH, MODERATE, LOW, or CLEAR."
#         "\nCATEGORY must be one of: Spam/Phishing, Policy Violation, Customer Complaint, Clear/Other."
        
#         "\n\nCRITICAL Risk: Direct threats, hate speech, illegal content, or obvious spam/phishing attempts. This requires immediate human review."
#         "\nHIGH Risk: Very aggressive or highly offensive language, extreme negative sentiment (e.g., 'scam', 'fraud'). This requires human review."
#         "\nMODERATE Risk: Strong negative sentiment (e.g., 'bad', 'terrible'), minor policy violations. This might require human review."
#         "\nLOW Risk: Neutral or slightly positive sentiment, very mild or ambiguous concerns. This usually passes."
#         "\nCLEAR Risk: Clearly positive or neutral, non-concerning messages. This always passes."
#     )
    
#     user_query = f"Analyze and classify the following message:\n'{message}'"
#     prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"
    
#     payload = {
#         "model": OLLAMA_MODEL,
#         "prompt": prompt,
#         "stream": False,
#         "options": {
#             # Low temperature for deterministic, structured output
#             "temperature": 0.2
#         }
#     }

#     try:
#         response = requests.post(OLLAMA_API_URL, json=payload, timeout=45) # Increased timeout for moderation
#         response.raise_for_status()

#         result = response.json()
#         raw_response_text = result.get('response', '').strip()

#         try:
#             # Attempt to parse the raw text directly as JSON
#             moderation_data = json.loads(raw_response_text)
            
#             # Use Pydantic to validate the structure
#             validated_result = ModerationResult(**moderation_data)
#             return validated_result
        
#         except json.JSONDecodeError:
#             logging.error(f"Ollama returned malformed JSON for moderation: {raw_response_text[:100]}...")
#             # Fallback to a high risk if the LLM response is unparseable
#             return ModerationResult(
#                 risk_level="HIGH",
#                 category="System Error",
#                 reason="LLM response malformed, manual review required."
#             )
        
#     except requests.exceptions.ConnectionError:
#         logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
#         # Fallback to high risk on connection failure
#         return ModerationResult(
#             risk_level="HIGH",
#             category="System Error",
#             reason="Ollama connection failed, manual review required."
#         )
#     except Exception as e:
#         logging.error(f"An unexpected error occurred during Ollama moderation: {e}")
#         # Fallback to high risk on any other failure
#         return ModerationResult(
#             risk_level="HIGH",
#             category="System Error",
#             reason=f"Unexpected error: {str(e)[:50]}"
#         )


# # --- API Endpoints ---

# @app.get("/")
# def read_root():
#     return {"status": "AI Social Media Backend is Running (OLLAMA INTEGRATION ACTIVE)"}

# # --- Content Generation Endpoint (OLLAMA ACTIVE) ---
# @app.post("/api/content/generate", response_model=SocialMediaPost)
# async def generate_post_content(data: ContentGenerationInput):
#     """Generates social media content using the local Ollama LLM."""
    
#     logging.info(f"⚡ OLLAMA generation requested for {data.platform} post (Topic: {data.prompt})...")

#     loop = asyncio.get_event_loop()
    
#     try:
#         validated_post = await loop.run_in_executor(
#             None, 
#             generate_post_content_with_ollama, 
#             data
#         )
        
#         # --- ANALYTICS LOGGING: GENERATION ---
#         ALL_GENERATION_LOGS.append({
#             "platform": data.platform,
#             "tone": data.tone,
#             "timestamp": int(time.time()),
#         })
#         # -----------------------------------

#         logging.info(f"🎉 Ollama Content Generated Successfully for {validated_post.platform}.")
#         return validated_post

#     except (ConnectionError, TimeoutError) as e:
#         raise HTTPException(
#             status_code=503, 
#             detail=f"Local LLM Error: {e}"
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to generate content: {e}")


# # --- Moderation & Review Endpoints (Updated) ---

# @app.post("/api/moderation/simulate", response_model=Union[ReviewItem, dict])
# async def dm_webhook(data: WebhookData):
#     """Real-time processing of incoming DMs for moderation using Ollama."""
#     logging.info(f"--- DM Received: \"{data.message_text[:40]}...\" from {data.sender_id} ---")

#     # Use a thread pool to run the synchronous Ollama call for moderation
#     loop = asyncio.get_event_loop()
#     moderation_result = await loop.run_in_executor(
#         None, 
#         moderate_message_with_ollama, 
#         data.message_text
#     )

#     # Risk levels that require a manual review item to be created
#     needs_review = moderation_result.risk_level in ["CRITICAL", "HIGH", "MODERATE"]

#     # --- ANALYTICS LOGGING: MODERATION (Log ALL events) ---
#     ALL_MODERATION_LOGS.append({
#         "risk_level": moderation_result.risk_level,
#         "category": moderation_result.category,
#         "timestamp": int(time.time()),
#     })
#     # -----------------------------------
    
#     if needs_review:
        
#         # Determine the final category based on the Ollama output
#         risk_level = moderation_result.risk_level
#         category = moderation_result.category
#         reason = f"{moderation_result.reason} ({risk_level} Risk)"
        
#         new_item = ReviewItem(
#             id=int(time.time() * 1000),
#             sender_id=data.sender_id,
#             message_text=data.message_text,
#             reason=reason,
#             timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#             risk_level=risk_level, 
#             category=category
#         ).model_dump()

#         MANUAL_REVIEW_QUEUE.insert(0, new_item)
#         logging.warning(f"🚨 Added to Review Queue: ID {new_item['id']} ({reason})")
#         return new_item
    
#     # --- NEW LOGIC: Check for SPAM/PHISHING even if it didn't trigger a review (e.g., LOW/CLEAR risk) ---
#     if moderation_result.category == "Spam/Phishing":
#         logging.warning(f"🚫 Flagged as SPAM: {moderation_result.reason}")
#         # Returns a specific "SPAM" status instead of "PASSED"
#         return {"status": "SPAM", "reason": moderation_result.reason}


#     # If it's LOW or CLEAR risk and not Spam, it passes
#     return {"status": "PASSED", "reason": moderation_result.reason}

# @app.get("/api/moderation/queue", response_model=List[ReviewItem])
# def get_review_queue():
#     """Endpoint to retrieve all items in the manual review queue."""
#     return MANUAL_REVIEW_QUEUE 

# @app.post("/api/moderation/queue/clear")
# def clear_review_queue():
#     """Endpoint to clear all items from the manual review queue."""
#     MANUAL_REVIEW_QUEUE.clear()
    
#     MANUAL_REVIEW_QUEUE.append(ReviewItem(
#         id=int(time.time() * 1000),
#         sender_id="system_message",
#         message_text="The queue was manually cleared by a human moderator.",
#         reason="Queue Cleared",
#         timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#         risk_level="LOW",
#         category="Admin Action"
#     ).model_dump())

#     logging.info("🧹 Manual Review Queue Cleared.")
#     return {"status": "success", "message": "Queue cleared."}


# # --- NEW POST SCHEDULING ENDPOINT ---
# @app.post("/api/schedule/post")
# def schedule_post(post: SocialMediaPost):
#     """Schedules a generated social media post for later publishing."""
    
#     # In a real application, this would save to a persistent database (like Firestore)
#     # and include a 'publish_time' field. For this mock, we just store it.
    
#     # Ensure it has a timestamp if one wasn't explicitly set by the generation logic (should be rare)
#     if not post.created_at:
#         post.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

#     SCHEDULED_POSTS.insert(0, post.model_dump())
    
#     logging.info(f"📅 Post scheduled for {post.platform}: {post.post_content[:20]}...")
#     return {"status": "success", "message": "Post successfully scheduled."}
# # --- END NEW POST SCHEDULING ENDPOINT ---


# # --- ANALYTICS AGGREGATION ENDPOINT ---

# @app.get("/api/analytics/summary")
# def get_analytics_summary():
#     """Aggregates all logs into a summary for the dashboard."""
    
#     # --- MODERATION METRICS ---
    
#     total_processed = len(ALL_MODERATION_LOGS)
    
#     risk_counts = defaultdict(int)
#     category_counts = defaultdict(int)
    
#     for log in ALL_MODERATION_LOGS:
#         # Standardizing risk levels for counting, useful if the LLM sometimes returns variations
#         risk = log['risk_level'].upper()
#         category = log['category']
        
#         # Count all occurrences
#         risk_counts[risk] += 1
#         category_counts[category] += 1
        
#     # Calculate flag rate (CRITICAL, HIGH, MODERATE, any explicit SPAM)
#     # We also check the category for "System Error" and count those towards flagged content
#     flagged_categories = ['CRITICAL', 'HIGH', 'MODERATE', 'SPAM', 'SYSTEM ERROR']
#     flagged_count = sum(count for level, count in risk_counts.items() if level in flagged_categories)
    
#     flag_rate = (flagged_count / total_processed * 100) if total_processed > 0 else 0
    
#     # --- GENERATION METRICS ---
    
#     total_generated = len(ALL_GENERATION_LOGS)
#     platform_counts = defaultdict(int)
#     tone_counts = defaultdict(int)

#     for log in ALL_GENERATION_LOGS:
#         platform_counts[log['platform']] += 1
#         tone_counts[log['tone']] += 1

#     return {
#         "moderation_stats": {
#             "total_processed": total_processed,
#             "flag_rate_percent": round(flag_rate, 2),
#             "review_queue_size": len(MANUAL_REVIEW_QUEUE),
#             # Convert defaultdicts to dict for JSON serialization
#             "risk_counts": dict(risk_counts),
#             "category_counts": dict(category_counts)
#         },
#         "generation_stats": {
#             "total_generated": total_generated,
#             "platform_counts": dict(platform_counts),
#             "tone_counts": dict(tone_counts)
#         }
#     }

# # --- END ANALYTICS AGGREGATION ENDPOINT ---

# # --- Background Scheduler (Mock) ---

# @app.get("/api/schedule/posts", response_model=List[SocialMediaPost])
# def get_scheduled_posts():
#     """Get all currently scheduled posts."""
#     # Returns the list directly
#     return SCHEDULED_POSTS

# async def background_scheduler():
#     """This function simulates a persistent background scheduler."""
#     logging.info("--- Starting Background Scheduler ---")
#     while True:
#         await asyncio.sleep(60) 

# @app.on_event("startup")
# async def startup_event():
#     """Starts the background scheduler on API startup."""
#     if not hasattr(app.state, 'scheduler_task'):
#         app.state.scheduler_task = asyncio.create_task(background_scheduler())

# @app.on_event("shutdown")
# def shutdown_event():
#     """Executes on server shutdown."""
#     if hasattr(app.state, 'scheduler_task') and not app.state.scheduler_task.done():
#         app.state.scheduler_task.cancel()
#     logging.info("Server shutting down.")


# # --- MANDATORY SERVER RUN BLOCK ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000


import time
import asyncio
import json
import logging
from collections import defaultdict
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# We use 'requests' to talk to the local Ollama API for BOTH content generation and moderation.
import requests 

# --- Configuration ---

# Configure logging to show INFO messages
logging.basicConfig(level=logging.INFO, format='INFO: %(message)s')

# --- Global State and Initialization ---

# --- IMPORTANT LLM CONFIGURATION ---
OLLAMA_MODEL = "mistral" # CHANGE THIS if you pulled a different model (e.g., 'gemma:2b')
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# ---

app = FastAPI(title="Social Media AI Backend (OLLAMA INTEGRATION)", version="0.1.0")

# 1. Moderation Model Initialization (Using Ollama LLM)
logging.info("✅ Moderation is now powered by Ollama LLM for higher precision.")


# 2. In-Memory Data Store 
MANUAL_REVIEW_QUEUE = []
SCHEDULED_POSTS = []

# --- NEW GLOBAL DATA STRUCTURES FOR ANALYTICS ---

# Log of all moderation results for the analytics dashboard
# Stores: {'risk_level': str, 'category': str, 'timestamp': int}
ALL_MODERATION_LOGS = []

# Log of all content generation requests
# Stores: {'platform': str, 'tone': str, 'timestamp': int}
ALL_GENERATION_LOGS = []

# --- END NEW GLOBAL DATA STRUCTURES ---


# Initial mock item to demonstrate HIGH risk due to VERY_NEGATIVE sentiment
MOCK_REVIEW_ITEM = {
    "id": 1678886400000,
    "sender_id": "critical_user_1",
    "message_text": "This product is an absolute scam and I demand a full refund immediately.",
    "reason": "Very Negative Sentiment (HIGH Risk)",
    "timestamp": "2024-03-15T10:00:00Z",
    "risk_level": "HIGH", 
    "category": "Customer Complaint"
}
MANUAL_REVIEW_QUEUE.append(MOCK_REVIEW_ITEM)

# Log the initial mock item so it shows up in the dashboard
ALL_MODERATION_LOGS.append({
    "risk_level": MOCK_REVIEW_ITEM["risk_level"],
    "category": MOCK_REVIEW_ITEM["category"],
    "timestamp": int(time.time() - 3600), # 1 hour ago
})
logging.info("--- NOTE: One mock item added to the Manual Review Queue and Analytics Log for initial testing. ---")


# --- Middleware (CORS) - CRITICAL FIX ---
# Added the frontend server addresses (5500) to allow communication.
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000", # <-- FIX: Added 127.0.0.1:8000 for local uvicorn execution
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5500",      
    "http://127.0.0.1:5500",    
    "http://localhost:5001",
    "http://127.0.0.1:5001",
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
    "file://", # Added for maximum compatibility if running directly from file
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allowing all origins is often simpler for local development
    allow_credentials=True,
    allow_methods=["*"], # Essential for allowing the OPTIONS preflight check
    allow_headers=["*"],
)
# --- END CRITICAL FIX ---


# --- Data Models (Pydantic) ---

class ContentGenerationInput(BaseModel):
    """Data model for the input parameters of content generation."""
    prompt: str
    length: str
    platform: str
    tone: str 

class SocialMediaPost(BaseModel):
    """The JSON structure we expect the LLM to return for generation."""
    post_content: str = Field(..., description="The main text content of the social media post.")
    suggested_hashtags: List[str] = Field(..., description="A list of relevant hashtags for the post.")
    platform: str = Field(..., description="The platform the post was optimized for (e.g., 'facebook', 'twitter').")
    # Added field to support scheduling from frontend
    created_at: Optional[str] = Field(None, description="Timestamp when the post was created/scheduled.")

class ModerationResult(BaseModel):
    """The STRICT JSON structure we expect the LLM to return for moderation."""
    risk_level: str = Field(..., description="Classification: CRITICAL, HIGH, MODERATE, LOW, or CLEAR.")
    category: str = Field(..., description="Classification: Spam/Phishing, Policy Violation, Customer Complaint, or Clear/Other.")
    reason: str = Field(..., description="A brief, human-readable explanation of why this risk level was assigned.")
    
class WebhookData(BaseModel):
    """Data model for an incoming DM webhook."""
    sender_id: str
    message_text: str

class ReviewItem(BaseModel):
    """Data model for an item in the manual review queue."""
    id: int
    sender_id: str
    message_text: str
    reason: str
    timestamp: str
    risk_level: str
    category: str


# --- Core Logic Functions (OLLAMA INTEGRATION) ---

def generate_ollama_prompt(data: ContentGenerationInput) -> str:
    """
    Constructs a detailed system instruction prompt for the LLM to ensure
    it returns the content in the required JSON format and adheres to constraints.
    """
    # System Prompt to force JSON output
    system_prompt = (
        "You are an expert social media content creator. Your task is to generate a single social media post "
        "based on the user's request. **You MUST respond only with a valid, clean JSON object** "
        "that adheres to the following exact schema: "
        "{\"post_content\": \"string\", \"suggested_hashtags\": [\"string\"], \"platform\": \"string\"}."
        "Do not include any introductory text, conversation, or markdown formatting (e.g., ```json) outside the JSON object."
        "The generated content must strictly follow the tone, length, and platform constraints."
    )
    
    # User Query containing all parameters
    user_query = (
        f"Generate a post for the topic: '{data.prompt}'. "
        f"The post should be a '{data.length}' length, "
        f"using a '{data.tone}' tone, and optimized for the '{data.platform}' platform. "
        f"Ensure the post content is professional and engaging."
    )
    
    return f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"


def generate_post_content_with_ollama(data: ContentGenerationInput) -> SocialMediaPost:
    """
    Function to call the local Ollama API endpoint for content generation.
    """
    prompt = generate_ollama_prompt(data)
    
    # Ollama REST API Payload
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            # Low temperature for deterministic, structured output
            "temperature": 0.3 
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status() 

        result = response.json()
        raw_response_text = result.get('response', '').strip()
        
        try:
            post_data = json.loads(raw_response_text)
            # Add creation timestamp before validation
            post_data['created_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            validated_post = SocialMediaPost(**post_data)
            return validated_post
        
        except json.JSONDecodeError:
            logging.error(f"Ollama returned malformed JSON for generation: {raw_response_text[:100]}...")
            raise ValueError("LLM returned malformed JSON. Check model and prompt structure.")

    except requests.exceptions.ConnectionError:
        logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
        raise ConnectionError("Could not connect to the local Ollama server. Please check that Ollama is running.")
    
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama request timed out. Increase timeout or check model performance.")
    
    except Exception as e:
        logging.error(f"An unexpected error occurred during Ollama generation: {e}")
        raise e

def moderate_message_with_ollama(message: str) -> ModerationResult:
    """
    Uses the Ollama LLM to classify a message into one of five risk levels 
    and returns a structured JSON result for moderation.
    """
    
    # System Prompt to force JSON output with specific categories
    system_prompt = (
        "You are an expert social media moderation AI. Your task is to analyze the user message and "
        "classify its risk level and category. **You MUST respond ONLY with a valid, clean JSON object** "
        "that strictly adheres to the schema: "
        "{\"risk_level\": \"string\", \"category\": \"string\", \"reason\": \"string\"}."
        "Do not include any other text or markdown formatting outside the JSON object."
        
        "\n\n--- CLASSIFICATION RULES ---"
        "\nRISK_LEVEL must be one of: CRITICAL, HIGH, MODERATE, LOW, or CLEAR."
        "\nCATEGORY must be one of: Spam/Phishing, Policy Violation, Customer Complaint, Clear/Other."
        
        "\n\nCRITICAL Risk: Direct threats, hate speech, illegal content, or obvious spam/phishing attempts. This requires immediate human review."
        "\nHIGH Risk: Very aggressive or highly offensive language, extreme negative sentiment (e.g., 'scam', 'fraud'). This requires human review."
        "\nMODERATE Risk: Strong negative sentiment (e.g., 'bad', 'terrible'), minor policy violations. This might require human review."
        "\nLOW Risk: Neutral or slightly positive sentiment, very mild or ambiguous concerns. This usually passes."
        "\nCLEAR Risk: Clearly positive or neutral, non-concerning messages. This always passes."
    )
    
    user_query = f"Analyze and classify the following message:\n'{message}'"
    prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_query}"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            # Low temperature for deterministic, structured output
            "temperature": 0.2
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=45) # Increased timeout for moderation
        response.raise_for_status()

        result = response.json()
        raw_response_text = result.get('response', '').strip()

        try:
            # Attempt to parse the raw text directly as JSON
            moderation_data = json.loads(raw_response_text)
            
            # Use Pydantic to validate the structure
            validated_result = ModerationResult(**moderation_data)
            return validated_result
        
        except json.JSONDecodeError:
            logging.error(f"Ollama returned malformed JSON for moderation: {raw_response_text[:100]}...")
            # Fallback to a high risk if the LLM response is unparseable
            return ModerationResult(
                risk_level="HIGH",
                category="System Error",
                reason="LLM response malformed, manual review required."
            )
        
    except requests.exceptions.ConnectionError:
        logging.error(f"Ollama Connection Error. Is Ollama running and is the model '{OLLAMA_MODEL}' loaded?")
        # Fallback to high risk on connection failure
        return ModerationResult(
            risk_level="HIGH",
            category="System Error",
            reason="Ollama connection failed, manual review required."
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred during Ollama moderation: {e}")
        # Fallback to high risk on any other failure
        return ModerationResult(
            risk_level="HIGH",
            category="System Error",
            reason=f"Unexpected error: {str(e)[:50]}"
        )


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "AI Social Media Backend is Running (OLLAMA INTEGRATION ACTIVE)"}

# --- Content Generation Endpoint (OLLAMA ACTIVE) ---
@app.post("/api/content/generate", response_model=SocialMediaPost)
async def generate_post_content(data: ContentGenerationInput):
    """Generates social media content using the local Ollama LLM."""
    
    logging.info(f"⚡ OLLAMA generation requested for {data.platform} post (Topic: {data.prompt})...")

    loop = asyncio.get_event_loop()
    
    try:
        validated_post = await loop.run_in_executor(
            None, 
            generate_post_content_with_ollama, 
            data
        )
        
        # --- ANALYTICS LOGGING: GENERATION ---
        ALL_GENERATION_LOGS.append({
            "platform": data.platform,
            "tone": data.tone,
            "timestamp": int(time.time()),
        })
        # -----------------------------------

        logging.info(f"🎉 Ollama Content Generated Successfully for {validated_post.platform}.")
        return validated_post

    except (ConnectionError, TimeoutError) as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Local LLM Error: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate content: {e}")


# --- Moderation & Review Endpoints (Updated) ---

@app.post("/api/moderation/simulate", response_model=Union[ReviewItem, dict])
async def dm_webhook(data: WebhookData):
    """Real-time processing of incoming DMs for moderation using Ollama."""
    logging.info(f"--- DM Received: \"{data.message_text[:40]}...\" from {data.sender_id} ---")

    # Use a thread pool to run the synchronous Ollama call for moderation
    loop = asyncio.get_event_loop()
    moderation_result = await loop.run_in_executor(
        None, 
        moderate_message_with_ollama, 
        data.message_text
    )

    # Risk levels that require a manual review item to be created
    needs_review = moderation_result.risk_level in ["CRITICAL", "HIGH", "MODERATE"]

    # --- ANALYTICS LOGGING: MODERATION (Log ALL events) ---
    ALL_MODERATION_LOGS.append({
        "risk_level": moderation_result.risk_level,
        "category": moderation_result.category,
        "timestamp": int(time.time()),
    })
    # -----------------------------------
    
    if needs_review:
        
        # Determine the final category based on the Ollama output
        risk_level = moderation_result.risk_level
        category = moderation_result.category
        reason = f"{moderation_result.reason} ({risk_level} Risk)"
        
        new_item = ReviewItem(
            id=int(time.time() * 1000),
            sender_id=data.sender_id,
            message_text=data.message_text,
            reason=reason,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            risk_level=risk_level, 
            category=category
        ).model_dump()

        MANUAL_REVIEW_QUEUE.insert(0, new_item)
        logging.warning(f"🚨 Added to Review Queue: ID {new_item['id']} ({reason})")
        return new_item
    
    # --- NEW LOGIC: Check for SPAM/PHISHING even if it didn't trigger a review (e.g., LOW/CLEAR risk) ---
    if moderation_result.category == "Spam/Phishing":
        logging.warning(f"🚫 Flagged as SPAM: {moderation_result.reason}")
        # Returns a specific "SPAM" status instead of "PASSED"
        return {"status": "SPAM", "reason": moderation_result.reason}


    # If it's LOW or CLEAR risk and not Spam, it passes
    return {"status": "PASSED", "reason": moderation_result.reason}

@app.get("/api/moderation/queue", response_model=List[ReviewItem])
def get_review_queue():
    """Endpoint to retrieve all items in the manual review queue."""
    return MANUAL_REVIEW_QUEUE 

# --- NEW ENDPOINT: Resolve a specific item in the review queue ---
@app.post("/api/moderation/queue/resolve/{item_id}")
def resolve_queue_item(item_id: int, resolution_action: str):
    """
    Removes an item from the queue by ID and logs the resolution action.
    resolution_action is expected to be 'approve' or 'deny'.
    """
    global MANUAL_REVIEW_QUEUE
    
    # Convert item_id to int to match the format stored in the queue
    target_id = int(item_id)
    
    # Check current size for error handling
    initial_queue_size = len(MANUAL_REVIEW_QUEUE)
    
    # Filter out the item with the matching ID
    MANUAL_REVIEW_QUEUE = [item for item in MANUAL_REVIEW_QUEUE if item.get('id') != target_id]
    
    if len(MANUAL_REVIEW_QUEUE) == initial_queue_size:
        raise HTTPException(status_code=404, detail=f"Item with ID {item_id} not found in the queue.")

    # Log the action
    logging.info(f"✅ Item ID {item_id} resolved by moderator: {resolution_action.upper()}.")

    return {"status": "resolved", "action": resolution_action, "id": item_id}
# --- END NEW RESOLVE ENDPOINT ---

@app.post("/api/moderation/queue/clear")
def clear_review_queue():
    """Endpoint to clear all items from the manual review queue."""
    MANUAL_REVIEW_QUEUE.clear()
    
    MANUAL_REVIEW_QUEUE.append(ReviewItem(
        id=int(time.time() * 1000),
        sender_id="system_message",
        message_text="The queue was manually cleared by a human moderator.",
        reason="Queue Cleared",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        risk_level="LOW",
        category="Admin Action"
    ).model_dump())

    logging.info("🧹 Manual Review Queue Cleared.")
    return {"status": "success", "message": "Queue cleared."}


# --- NEW POST SCHEDULING ENDPOINT ---
@app.post("/api/schedule/post")
def schedule_post(post: SocialMediaPost):
    """Schedules a generated social media post for later publishing."""
    
    # In a real application, this would save to a persistent database (like Firestore)
    # and include a 'publish_time' field. For this mock, we just store it.
    
    # Ensure it has a timestamp if one wasn't explicitly set by the generation logic (should be rare)
    if not post.created_at:
        post.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    SCHEDULED_POSTS.insert(0, post.model_dump())
    
    logging.info(f"📅 Post scheduled for {post.platform}: {post.post_content[:20]}...")
    return {"status": "success", "message": "Post successfully scheduled."}
# --- END NEW POST SCHEDULING ENDPOINT ---


# --- ANALYTICS AGGREGATION ENDPOINT ---

@app.get("/api/analytics/summary")
def get_analytics_summary():
    """Aggregates all logs into a summary for the dashboard."""
    
    # --- MODERATION METRICS ---
    
    total_processed = len(ALL_MODERATION_LOGS)
    
    risk_counts = defaultdict(int)
    category_counts = defaultdict(int)
    
    for log in ALL_MODERATION_LOGS:
        # Standardizing risk levels for counting, useful if the LLM sometimes returns variations
        risk = log['risk_level'].upper()
        category = log['category']
        
        # Count all occurrences
        risk_counts[risk] += 1
        category_counts[category] += 1
        
    # Calculate flag rate (CRITICAL, HIGH, MODERATE, any explicit SPAM)
    # We also check the category for "System Error" and count those towards flagged content
    flagged_categories = ['CRITICAL', 'HIGH', 'MODERATE', 'SPAM', 'SYSTEM ERROR']
    
    # Recalculate flagged_count using the risk_counts dictionary (which is what we populate)
    flagged_count = sum(count for level, count in risk_counts.items() if level in flagged_categories)
    
    flag_rate = (flagged_count / total_processed * 100) if total_processed > 0 else 0
    
    # --- GENERATION METRICS ---
    
    total_generated = len(ALL_GENERATION_LOGS)
    platform_counts = defaultdict(int)
    tone_counts = defaultdict(int)

    for log in ALL_GENERATION_LOGS:
        platform_counts[log['platform']] += 1
        tone_counts[log['tone']] += 1

    return {
        "moderation_stats": {
            "total_processed": total_processed,
            "flag_rate_percent": round(flag_rate, 2),
            "review_queue_size": len(MANUAL_REVIEW_QUEUE),
            # Convert defaultdicts to dict for JSON serialization
            "risk_counts": dict(risk_counts),
            "category_counts": dict(category_counts)
        },
        "generation_stats": {
            "total_generated": total_generated,
            "platform_counts": dict(platform_counts),
            "tone_counts": dict(tone_counts)
        }
    }

# --- END ANALYTICS AGGREGATION ENDPOINT ---

# --- Background Scheduler (Mock) ---

@app.get("/api/schedule/posts", response_model=List[SocialMediaPost])
def get_scheduled_posts():
    """Get all currently scheduled posts."""
    # Returns the list directly
    return SCHEDULED_POSTS

async def background_scheduler():
    """This function simulates a persistent background scheduler."""
    logging.info("--- Starting Background Scheduler ---")
    while True:
        await asyncio.sleep(60) 

@app.on_event("startup")
async def startup_event():
    """Starts the background scheduler on API startup."""
    if not hasattr(app.state, 'scheduler_task'):
        app.state.scheduler_task = asyncio.create_task(background_scheduler())

@app.on_event("shutdown")
def shutdown_event():
    """Executes on server shutdown."""
    if hasattr(app.state, 'scheduler_task') and not app.state.scheduler_task.done():
        app.state.scheduler_task.cancel()
    logging.info("Server shutting down.")


# --- MANDATORY SERVER RUN BLOCK ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)