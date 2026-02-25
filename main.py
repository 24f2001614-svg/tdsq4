import os
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Customer Feedback Analysis API")

# Define Structured Output Schema
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5)

class CommentRequest(BaseModel):
    comment: str

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    try:
        # OpenAI Structured Output call
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a customer feedback analyzer. Analyze the sentiment and rating (1-5) of the given comment."},
                {"role": "user", "content": request.comment},
            ],
            response_format=SentimentResponse,
        )

        analysis = completion.choices[0].message.parsed
        
        if not analysis:
            raise HTTPException(status_code=500, detail="Failed to parse sentiment analysis.")
            
        return analysis

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
