from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow frontend access during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "AI Literacy Chatbot is running."}

@app.post("/chat")
async def chat_endpoint(user_message: Message):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o",  # or use gpt-3.5-turbo for cheaper testing
            messages=[
                {"role": "system", "content": "You are an AI literacy tutor. Follow the curriculum strictly."},
                {"role": "user", "content": user_message.message}
            ]
        )
        reply = completion.choices[0].message.content
        return {"response": reply}
    except Exception as e:
        return {"error": str(e)}