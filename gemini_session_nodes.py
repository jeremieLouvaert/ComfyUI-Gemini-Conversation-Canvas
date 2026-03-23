"""
ComfyUI Gemini Conversation Canvas
Multi-turn conversational image editing with Google Gemini.
"""

import os
import io
import json
import uuid
import numpy as np
from datetime import datetime, timezone

import torch
from PIL import Image

try:
    from filelock import FileLock
except ImportError:
    class FileLock:
        def __init__(self, lock_file, timeout=-1):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

try:
    import folder_paths
except ImportError:
    folder_paths = None

# ---------------------------------------------------------------------------
# GEMINI SDK IMPORT
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ---------------------------------------------------------------------------
# CUSTOM TYPE — passes ComfyUI type-checking for session state
# ---------------------------------------------------------------------------

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

GEMINI_SESSION_TYPE = "GEMINI_SESSION"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image",
]

ASPECT_RATIOS = ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]

ALL_RESOLUTIONS = ["1K", "2K", "4K"]

CATEGORY = "Gemini Conversation Canvas"

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def _get_output_dir():
    if folder_paths:
        return folder_paths.get_output_directory()
    return os.path.join(os.path.dirname(__file__), "output")


def _sessions_dir():
    d = os.path.join(_get_output_dir(), "gemini_sessions")
    os.makedirs(d, exist_ok=True)
    return d


def _session_dir(session_name):
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in session_name)
    d = os.path.join(_sessions_dir(), safe)
    os.makedirs(d, exist_ok=True)
    return d


def _resolve_api_key(api_key_input=""):
    """Resolve API key: direct input > env var > file."""
    if api_key_input and api_key_input.strip():
        return api_key_input.strip()

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    # Check for key file in ComfyUI root
    if folder_paths:
        root = os.path.dirname(folder_paths.get_output_directory())
    else:
        root = os.path.dirname(__file__)
    key_file = os.path.join(root, "gemini_api_key.txt")
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            file_key = f.read().strip()
        if file_key:
            return file_key

    raise ValueError(
        "[Gemini Canvas] No API key found. Provide one via:\n"
        "  1. The api_key input on the node\n"
        "  2. GEMINI_API_KEY environment variable\n"
        "  3. A gemini_api_key.txt file in your ComfyUI root directory"
    )


def _get_client(api_key):
    if not HAS_GENAI:
        raise ImportError(
            "[Gemini Canvas] google-genai package not installed.\n"
            "Run: pip install google-genai"
        )
    return genai.Client(api_key=api_key)


def _tensor_to_pil(tensor):
    """Convert ComfyUI IMAGE tensor (B,H,W,C float32 0-1) to PIL Image."""
    if tensor is None:
        return None
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first image from batch
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _pil_to_tensor(pil_img):
    """Convert PIL Image to ComfyUI IMAGE tensor (1,H,W,C float32 0-1)."""
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _pil_to_bytes(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def _bytes_to_pil(data):
    return Image.open(io.BytesIO(data))


def _extract_image_and_text(response):
    """Extract image bytes and text from a Gemini response."""
    image_data = None
    text_parts = []
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            image_data = part.inline_data.data
        if part.text:
            text_parts.append(part.text)
    return image_data, "\n".join(text_parts)


def _save_turn_image(session_dir, turn_index, image_bytes):
    path = os.path.join(session_dir, f"turn_{turn_index:03d}.png")
    pil = _bytes_to_pil(image_bytes)
    pil.save(path, "PNG")
    return path


def _create_empty_session(session_name, model, api_key):
    return {
        "session_id": str(uuid.uuid4()),
        "session_name": session_name,
        "model": model,
        "history": [],
        "turn_images": [],
        "turn_texts": [],
        "turn_count": 0,
        "api_key": api_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _save_session_to_disk(session):
    """Persist session metadata to disk. Images are already saved as PNGs."""
    sdir = _session_dir(session["session_name"])
    meta_file = os.path.join(sdir, "session.json")
    lock_file = meta_file + ".lock"

    # Save a copy without the api_key for security
    save_data = {k: v for k, v in session.items() if k != "api_key"}

    with FileLock(lock_file, timeout=10):
        tmp = meta_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, meta_file)

    print(f"💾 [Gemini Canvas] Session '{session['session_name']}' saved ({session['turn_count']} turns)")


def _load_session_from_disk(session_name, api_key):
    """Load session from disk."""
    sdir = _session_dir(session_name)
    meta_file = os.path.join(sdir, "session.json")
    lock_file = meta_file + ".lock"

    if not os.path.exists(meta_file):
        raise FileNotFoundError(
            f"[Gemini Canvas] No saved session found: '{session_name}'\n"
            f"Expected at: {meta_file}"
        )

    with FileLock(lock_file, timeout=10):
        with open(meta_file, "r", encoding="utf-8") as f:
            session = json.load(f)

    session["api_key"] = api_key
    print(f"📂 [Gemini Canvas] Loaded session '{session_name}' ({session['turn_count']} turns)")
    return session


def _rebuild_chat(session):
    """Rebuild a Gemini Chat object from serialized history."""
    client = _get_client(session["api_key"])

    # Deserialize history back to Content objects
    history = []
    for item in session["history"]:
        history.append(types.Content.model_validate(item))

    chat = client.chats.create(
        model=session["model"],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
        history=history,
    )
    return chat, client


def _list_saved_sessions():
    """List all saved session names."""
    sdir = _sessions_dir()
    sessions = []
    if os.path.exists(sdir):
        for name in sorted(os.listdir(sdir)):
            meta = os.path.join(sdir, name, "session.json")
            if os.path.exists(meta):
                sessions.append(name)
    return sessions if sessions else ["(no saved sessions)"]


# ============================================================================
# NODE 1: GEMINI SESSION START
# ============================================================================

class GeminiSessionStart:
    """Creates a new Gemini conversation and generates the initial image."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt for the initial image generation",
                }),
                "session_name": ("STRING", {
                    "default": "my_session",
                    "tooltip": "Name for saving/loading this session",
                }),
                "model": (MODELS, {
                    "default": MODELS[0],
                    "tooltip": "Gemini model to use for image generation",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "1:1",
                    "tooltip": "Aspect ratio for generated images",
                }),
                "resolution": (ALL_RESOLUTIONS, {
                    "default": "2K",
                    "tooltip": "Output resolution (1K, 2K, 4K)",
                }),
            },
            "optional": {
                "input_image": ("IMAGE", {
                    "tooltip": "Optional input image for image-to-image editing",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Google AI API key. Falls back to GEMINI_API_KEY env var or gemini_api_key.txt",
                }),
            },
        }

    RETURN_TYPES = (any_type, "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("session", "image", "text", "turn_count")
    FUNCTION = "start_session"
    CATEGORY = CATEGORY

    def start_session(self, prompt, session_name, model, aspect_ratio, resolution="2K",
                      input_image=None, api_key=""):

        resolved_key = _resolve_api_key(api_key)
        client = _get_client(resolved_key)
        session = _create_empty_session(session_name, model, resolved_key)
        sdir = _session_dir(session_name)

        # Build message parts
        message_parts = []

        if input_image is not None:
            pil_img = _tensor_to_pil(input_image)
            img_bytes = _pil_to_bytes(pil_img)
            message_parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        message_parts.append(prompt)

        # Build image config — omit aspect_ratio when set to "auto"
        image_config_kwargs = {"image_size": resolution}
        if aspect_ratio != "auto":
            image_config_kwargs["aspect_ratio"] = aspect_ratio

        # Create chat and send first message
        chat = client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(**image_config_kwargs),
            ),
        )

        print(f"🎬 [Gemini Canvas] Starting new session '{session_name}' with model {model}")
        response = chat.send_message(message_parts)

        # Extract results
        image_bytes, response_text = _extract_image_and_text(response)

        if image_bytes is None:
            raise RuntimeError(
                f"[Gemini Canvas] No image generated. Model response: {response_text}\n"
                "Try rephrasing your prompt or check content safety filters."
            )

        # Save turn image to disk
        turn_path = _save_turn_image(sdir, 0, image_bytes)

        # Serialize chat history
        history = chat.get_history()
        session["history"] = [c.model_dump(exclude_none=True) for c in history]
        session["turn_images"] = [turn_path]
        session["turn_texts"] = [response_text]
        session["turn_count"] = 1
        session["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Convert to ComfyUI tensor
        output_pil = _bytes_to_pil(image_bytes)
        output_tensor = _pil_to_tensor(output_pil)

        print(f"🟢 [Gemini Canvas] Turn 0 complete. Image: {output_pil.size[0]}x{output_pil.size[1]}")
        if response_text:
            print(f"   💬 {response_text[:100]}{'...' if len(response_text) > 100 else ''}")

        return (session, output_tensor, response_text, 1)


# ============================================================================
# NODE 2: GEMINI EDIT TURN
# ============================================================================

class GeminiEditTurn:
    """Sends an edit instruction to an existing Gemini conversation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "session": (any_type, {
                    "forceInput": True,
                    "tooltip": "Gemini session from Session Start or previous Edit Turn",
                }),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Natural language edit instruction (e.g. 'add rain to the scene')",
                }),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional reference image to include with the edit instruction",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "1:1",
                    "tooltip": "Aspect ratio (can change between turns)",
                }),
                "resolution": (ALL_RESOLUTIONS, {
                    "default": "2K",
                    "tooltip": "Output resolution (can change between turns)",
                }),
            },
        }

    RETURN_TYPES = (any_type, "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("session", "image", "text", "turn_count")
    FUNCTION = "edit_turn"
    CATEGORY = CATEGORY

    def edit_turn(self, session, instruction, reference_image=None, aspect_ratio="1:1", resolution="2K"):
        if not isinstance(session, dict) or "history" not in session:
            raise ValueError("[Gemini Canvas] Invalid session input. Connect a Session Start or Load node.")

        chat, client = _rebuild_chat(session)
        sdir = _session_dir(session["session_name"])
        turn_index = session["turn_count"]

        # Build message parts
        message_parts = []

        if reference_image is not None:
            pil_img = _tensor_to_pil(reference_image)
            img_bytes = _pil_to_bytes(pil_img)
            message_parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        message_parts.append(instruction)

        print(f"✏️ [Gemini Canvas] Turn {turn_index}: \"{instruction[:80]}{'...' if len(instruction) > 80 else ''}\"")

        # Build image config — omit aspect_ratio when set to "auto"
        image_config_kwargs = {"image_size": resolution}
        if aspect_ratio != "auto":
            image_config_kwargs["aspect_ratio"] = aspect_ratio

        response = chat.send_message(
            message_parts,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(**image_config_kwargs),
            ),
        )

        # Extract results
        image_bytes, response_text = _extract_image_and_text(response)

        if image_bytes is None:
            raise RuntimeError(
                f"[Gemini Canvas] No image generated on turn {turn_index}. "
                f"Model response: {response_text}\n"
                "Try rephrasing your instruction or check content safety filters."
            )

        # Save turn image
        turn_path = _save_turn_image(sdir, turn_index, image_bytes)

        # Update session
        history = chat.get_history()
        session["history"] = [c.model_dump(exclude_none=True) for c in history]
        session["turn_images"].append(turn_path)
        session["turn_texts"].append(response_text)
        session["turn_count"] = turn_index + 1
        session["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Convert to tensor
        output_pil = _bytes_to_pil(image_bytes)
        output_tensor = _pil_to_tensor(output_pil)

        print(f"🟢 [Gemini Canvas] Turn {turn_index} complete. Image: {output_pil.size[0]}x{output_pil.size[1]}")
        if response_text:
            print(f"   💬 {response_text[:100]}{'...' if len(response_text) > 100 else ''}")

        return (session, output_tensor, response_text, turn_index + 1)


# ============================================================================
# NODE 3: GEMINI SESSION SAVE
# ============================================================================

class GeminiSessionSave:
    """Persists a Gemini conversation session to disk."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "session": (any_type, {
                    "forceInput": True,
                    "tooltip": "Gemini session to save",
                }),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("session",)
    FUNCTION = "save_session"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def save_session(self, session):
        if not isinstance(session, dict) or "session_name" not in session:
            raise ValueError("[Gemini Canvas] Invalid session input.")
        _save_session_to_disk(session)
        return (session,)


# ============================================================================
# NODE 4: GEMINI SESSION LOAD
# ============================================================================

class GeminiSessionLoad:
    """Loads a previously saved Gemini conversation session from disk."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "session_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name of the saved session to load (folder name under output/gemini_sessions/)",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Google AI API key (needed to resume the conversation)",
                }),
            },
        }

    RETURN_TYPES = (any_type, "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("session", "last_image", "last_text", "turn_count")
    FUNCTION = "load_session"
    CATEGORY = CATEGORY

    @classmethod
    def VALIDATE_INPUTS(s, session_name, api_key=""):
        return True

    def load_session(self, session_name, api_key=""):
        resolved_key = _resolve_api_key(api_key)
        session = _load_session_from_disk(session_name, resolved_key)

        # Load the last turn image as tensor
        last_tensor = None
        if session["turn_images"]:
            last_path = session["turn_images"][-1]
            if os.path.exists(last_path):
                pil_img = Image.open(last_path).convert("RGB")
                last_tensor = _pil_to_tensor(pil_img)

        if last_tensor is None:
            # Return a 1x1 black placeholder if no images
            last_tensor = torch.zeros(1, 64, 64, 3, dtype=torch.float32)

        last_text = session["turn_texts"][-1] if session["turn_texts"] else ""

        return (session, last_tensor, last_text, session["turn_count"])


# ============================================================================
# NODE 5: GEMINI SESSION GALLERY
# ============================================================================

class GeminiSessionGallery:
    """Outputs all turn images from a session as a batch for preview."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "session": (any_type, {
                    "forceInput": True,
                    "tooltip": "Gemini session to display all turns from",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "turn_log", "turn_count")
    FUNCTION = "gallery"
    CATEGORY = CATEGORY

    def gallery(self, session):
        if not isinstance(session, dict) or "turn_images" not in session:
            raise ValueError("[Gemini Canvas] Invalid session input.")

        tensors = []
        max_h, max_w = 0, 0

        # First pass: load all images and find max dimensions
        pil_images = []
        for path in session["turn_images"]:
            if os.path.exists(path):
                pil = Image.open(path).convert("RGB")
                pil_images.append(pil)
                max_h = max(max_h, pil.size[1])
                max_w = max(max_w, pil.size[0])

        if not pil_images:
            empty = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
            return (empty, "(no images)", 0)

        # Second pass: pad all images to same size and convert to tensors
        for pil in pil_images:
            if pil.size != (max_w, max_h):
                padded = Image.new("RGB", (max_w, max_h), (0, 0, 0))
                padded.paste(pil, (0, 0))
                pil = padded
            tensors.append(_pil_to_tensor(pil))

        batch = torch.cat(tensors, dim=0)

        # Build turn log
        log_lines = []
        for i, text in enumerate(session.get("turn_texts", [])):
            preview = text[:120] + "..." if len(text) > 120 else text
            log_lines.append(f"Turn {i}: {preview}")
        turn_log = "\n".join(log_lines) if log_lines else "(no turns)"

        print(f"🖼️ [Gemini Canvas] Gallery: {len(pil_images)} turns")
        return (batch, turn_log, session["turn_count"])


# ============================================================================
# MAPPINGS
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "GeminiSessionStart": GeminiSessionStart,
    "GeminiEditTurn": GeminiEditTurn,
    "GeminiSessionSave": GeminiSessionSave,
    "GeminiSessionLoad": GeminiSessionLoad,
    "GeminiSessionGallery": GeminiSessionGallery,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiSessionStart": "🎬 Gemini Session (Start)",
    "GeminiEditTurn": "✏️ Gemini Edit Turn",
    "GeminiSessionSave": "💾 Gemini Session (Save)",
    "GeminiSessionLoad": "📂 Gemini Session (Load)",
    "GeminiSessionGallery": "🖼️ Gemini Session Gallery",
}
