from fastapi import FastAPI
from app.router import titanic

app = FastAPI()

app.include_router(titanic.router)

@app.get("/")
def root():
    return {"message": "Hello World"}