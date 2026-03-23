# ComfyUI Gemini Conversation Canvas

Multi-turn conversational image editing for ComfyUI, powered by Google Gemini's native image generation.

Unlike traditional image generation APIs that are fire-and-forget, Gemini maintains conversation context across turns — each edit builds on the previous result with full scene coherence. This node suite brings that capability into ComfyUI with persistent sessions, edit chaining, and a visual gallery.

## Why Gemini Conversations?

Most image APIs (DALL-E, Midjourney, Flux) are single-shot: you send a prompt, you get an image. There's no memory between generations.

Gemini is different. Its image generation is native to the language model, so it **remembers** what it generated and can reason about modifications:

```
Turn 0: "A cozy cafe interior with warm lighting"        → initial image
Turn 1: "Add rain visible through the windows"           → same cafe, now with rain
Turn 2: "Make it evening, warm lamp light inside"         → same scene, evening mood
Turn 3: "Add a cat sleeping on the corner chair"          → cat added coherently
```

Each turn preserves the scene, characters, and style — no need for ControlNet, IP-Adapter, or manual inpainting.

## Included Nodes

### 1. Gemini Session (Start)

Creates a new conversation and generates the initial image.

| Input | Type | Description |
|-------|------|-------------|
| `prompt` | STRING | Text prompt for the initial image |
| `session_name` | STRING | Name for saving/loading this session |
| `model` | COMBO | Gemini model selection |
| `aspect_ratio` | COMBO | Aspect ratio (1:1, 16:9, 9:16, etc.) |
| `input_image` | IMAGE (optional) | Source image for image-to-image editing |
| `api_key` | STRING (optional) | Google AI API key |

**Outputs:** `session`, `image`, `text`, `turn_count`

### 2. Gemini Edit Turn

Sends an edit instruction to an existing conversation. Chainable — connect multiple in sequence for multi-step edits in a single workflow run.

| Input | Type | Description |
|-------|------|-------------|
| `session` | SESSION | From Session Start, Load, or previous Edit Turn |
| `instruction` | STRING | Natural language edit instruction |
| `reference_image` | IMAGE (optional) | Reference image to include with the instruction |
| `aspect_ratio` | COMBO | Can change between turns |

**Outputs:** `session`, `image`, `text`, `turn_count`

### 3. Gemini Session (Save)

Persists the conversation to disk. Session history and all turn images are saved so you can resume later.

### 4. Gemini Session (Load)

Loads a previously saved session from disk. Connect to an Edit Turn node to continue the conversation.

### 5. Gemini Session Gallery

Outputs all turn images from a session as a batch tensor for visual comparison.

**Outputs:** `images` (batch), `turn_log` (text summary), `turn_count`

## Usage

### Basic: Single-Run Multi-Turn Editing

Chain multiple Edit Turn nodes for iterative refinement in one workflow execution:

```
[Session Start]                    [Edit Turn]               [Edit Turn]            [Save]
  prompt: "a red sports car"  →   "change to blue"     →   "add sunset sky"   →    💾
  ↓ image_0                        ↓ image_1                 ↓ image_2
```

### Advanced: Cross-Run Session Persistence

**Run 1:** Generate and save
```
[Session Start] → [Edit Turn] → [Session Save]
```

**Run 2:** Load, add more edits, save again
```
[Session Load] → [Edit Turn] → [Edit Turn] → [Session Save]
```

**Anytime:** View all turns
```
[Session Load] → [Session Gallery] → [Preview Image]
```

### Image-to-Image Editing

Connect an existing image to the `input_image` input on Session Start:

```
[Load Image] → [Session Start: "remove the background and replace with a beach"] → image
```

## API Key Setup

Provide your Google AI API key via one of these methods (checked in order):

1. **Direct input** — paste into the `api_key` field on Session Start or Session Load
2. **Environment variable** — set `GEMINI_API_KEY` in your system environment
3. **Key file** — create a `gemini_api_key.txt` file in your ComfyUI root directory

Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).

## Output Files

All session data is stored under your ComfyUI output directory:

| Path | Description |
|------|-------------|
| `output/gemini_sessions/{name}/session.json` | Conversation history and metadata |
| `output/gemini_sessions/{name}/turn_000.png` | Initial generated image |
| `output/gemini_sessions/{name}/turn_001.png` | Image after first edit |
| `output/gemini_sessions/{name}/turn_NNN.png` | Image after Nth edit |

## Installation

Clone this repository into your `ComfyUI/custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/jeremieLouvaert/ComfyUI-Gemini-Conversation-Canvas.git
pip install -r ComfyUI-Gemini-Conversation-Canvas/requirements.txt
```

Restart ComfyUI.

### Dependencies

- **google-genai** >= 1.0.0 — Google's Gen AI Python SDK
- **filelock** — concurrent-safe file operations
- **Pillow** — image conversion
- **PyTorch** — already present in any ComfyUI installation

## Supported Models

| Model | ID | Notes |
|-------|----|-------|
| Gemini 2.5 Flash Image | `gemini-2.5-flash-preview-image-generation` | Fast, affordable, production-ready |
| Gemini 2.0 Flash Image | `gemini-2.0-flash-preview-image-generation` | Previous generation |

## Tips

- **Be specific with edit instructions.** "Add a red umbrella to the person on the left" works better than "add umbrella."
- **The model remembers everything.** You can reference elements from previous turns: "make the cat from turn 3 larger."
- **Aspect ratio can change per turn.** Start in 1:1, then switch to 16:9 for a cinematic crop.
- **Save frequently.** Sessions persist across ComfyUI restarts, so save after important edits.
- **Use the Gallery node** to compare all turns side by side and verify the edit progression.

## License

MIT
