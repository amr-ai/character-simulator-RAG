import asyncio
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn

from personas.predefined import get_predefined_personas
from personas.custom import CustomPersona

# --- Application State ---
# We load models once at startup to avoid reloading on every request
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load predefined personas on startup
    print("Loading predefined personas...")
    app_state["personas"] = get_predefined_personas()
    print("Personas loaded.")
    yield
    # Clean up resources on shutdown
    app_state.clear()
    print("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# --- Static Files and Templates ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- HTML Routes ---

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Renders the homepage with persona selection."""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "personas": app_state.get("personas", {})}
    )

@app.get("/chat/{persona_name}", response_class=HTMLResponse)
async def get_chat_page(request: Request, persona_name: str):
    """Renders the chat interface for a specific persona."""
    persona = app_state.get("personas", {}).get(persona_name)
    if not persona:
        return RedirectResponse(url="/")
    return templates.TemplateResponse(
        "chat.html", 
        {"request": request, "persona": persona}
    )

@app.get("/create", response_class=HTMLResponse)
async def get_create_page(request: Request):
    """Renders the page to create a new custom persona."""
    return templates.TemplateResponse("create_persona.html", {"request": request})


# --- API / Logic Routes ---

@app.post("/chat/{persona_name}")
async def post_chat_message(persona_name: str, question: str = Form(...)):
    """Handles a chat message and returns the persona's response."""
    persona = app_state.get("personas", {}).get(persona_name)
    if not persona:
        return JSONResponse({"error": "Persona not found."}, status_code=404)
    
    # Run the potentially slow model invocation in a separate thread
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, persona.ask, question)
    
    return JSONResponse({"response": response})


@app.post("/create")
async def handle_create_persona(
    request: Request,
    name: str = Form(...),
    characteristics: str = Form(...),
    image_url: str = Form("/static/img/default-avatar.svg") # Default avatar
):
    """Creates a custom persona, builds its knowledge base, and redirects to chat."""
    if name in app_state.get("personas", {}):
        # Handle case where persona already exists
        return templates.TemplateResponse(
            "create_persona.html",
            {"request": request, "error": f"A persona named '{name}' already exists."},
        )

    # Instantiate the custom persona
    custom_persona = CustomPersona(name, characteristics, image_url)
    
    # Build the index (can be slow)
    loop = asyncio.get_event_loop()
    result_message = await loop.run_in_executor(None, custom_persona.create_index_from_wikipedia)

    if "error" in result_message.lower() or "not found" in result_message.lower():
        return templates.TemplateResponse(
            "create_persona.html",
            {"request": request, "error": result_message}
        )
    
    # Store the new persona and redirect
    app_state["personas"][name] = custom_persona
    return RedirectResponse(url=f"/chat/{name}", status_code=303)


# --- To run the app ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)