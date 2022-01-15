from fastapi import FastAPI
from app.router import iris

app = FastAPI()

app.include_router(iris.router)

@app.get("/")
def root():
    return {"message": "Hello World"}