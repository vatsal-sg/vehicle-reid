"""Vehicle re-identification evaluation via Ollama vision models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import ollama

# Generic prompt usable across vision models (Gemma, Qwen-VL, LLaVA, etc.).
VEHICLE_REID_PROMPT = """You are evaluating vehicle re-identification (ReID).

You are given exactly two images:
- Image A = first image
- Image B = second image

Task: Decide whether Image A and Image B show the SAME physical vehicle instance (same unique car), or DIFFERENT vehicles.

Important distinctions:
- SAME identity: one specific car seen again (possibly different camera, angle, time, weather, lighting, crop, or occlusion).
- SAME make/model/color is NOT enough for a match if they could be two different instances.
- DIFFERENT identity: another vehicle, even if similar class/color/model.

Analyze systematically (note uncertainty when evidence is weak):

1) Global / class-level
   - Body type / silhouette (sedan, SUV, hatchback, pickup, van, coupe, etc.)
   - Approximate size / proportions
   - Primary color and secondary colors (account for lighting, shadows, white-balance, night/IR)
   - Make and model cues if visible (badge, grille family, lamp signature, bumper shape)

2) Structure / geometry
   - Front/rear fascia shape, grille, bumper openings
   - Headlamp / taillamp shape and internal structure
   - Mirror shape, roof line, window geometry, wheel-arch shape
   - Wheel / hubcap / rim design if visible
   - Any aftermarket parts (racks, antennas, wraps)

3) Instance-level identifiers (strongest ReID signals)
   - License plate / plate region (partial plates count)
   - Stickers, decals, text, parking permits
   - Damage, dents, scratches, cracks, missing parts
   - Dirt / wear patterns if distinctive
   - Unique accessories (tow hitch, antennas, bike rack, cargo)

4) Capture conditions (do not confuse with identity)
   - Viewpoint (front/side/rear/oblique), distance, crop, blur, resolution
   - Occlusion, motion blur, glare, night lighting
   Explain whether apparent differences are likely imaging artifacts vs real physical differences.

Decision rules:
- Prefer MISMATCH if class-level features clearly conflict (body type, lamp family, clear color under good light), unless explained by imaging.
- Prefer MATCH only with converging instance cues (plate, damage, stickers, distinctive accessory) and compatible global features.
- Use UNCERTAIN when images are too degraded, too dissimilar in viewpoint with no shared identifiers, or evidence conflicts.

Output format (strict):
1) Short per-image summary (2–4 bullets each for A and B).
2) Comparison with evidence for MATCH vs MISMATCH.
3) Final block exactly as:

Verdict: MATCH | MISMATCH | UNCERTAIN
Confidence: <integer 0-100>
Key_evidence: <one sentence>
"""


def _validate_image_path(path: str) -> str:
    """Return an absolute path if the file exists; otherwise raise FileNotFoundError."""
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return str(image_path)


def _pull_model(model: str) -> None:
    """Ensure the Ollama model is present locally, streaming download progress."""
    print("Starting download (if not already cached)...")
    for progress in ollama.pull(model, stream=True):
        status = progress.get("status", "")
        total = progress.get("total") or 0
        completed = progress.get("completed") or 0
        if total > 0:
            percent = (completed / total) * 100
            digest = str(progress.get("digest", ""))[7:17]
            print(f"Downloading layer [{digest}] ... {percent:.1f}%", end="\r")
        else:
            print(f"{str(status).capitalize()}" + " " * 20, end="\r")
    print("\n\nModel ready! Analyzing images...\n")


def _print_response(response: dict[str, Any] | Any) -> None:
    """Print model content, falling back to thinking if content is empty."""
    # ollama may return a dict-like ChatResponse; support both.
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif isinstance(response, dict):
        payload = response
    else:
        payload = {
            "message": {
                "content": getattr(getattr(response, "message", None), "content", None),
                "thinking": getattr(getattr(response, "message", None), "thinking", None),
            },
            "done_reason": getattr(response, "done_reason", None),
        }

    message = payload.get("message") or {}
    content = (message.get("content") or "").strip()
    thinking = (message.get("thinking") or "").strip()
    done_reason = payload.get("done_reason")

    if thinking and not content:
        print("--- THINKING (content was empty) ---")
        print(thinking)
        print("-------------------")
        print(
            "Note: Final answer is empty. Thinking likely consumed the token budget "
            "(done_reason={!r}). Re-run with --no-think or raise --num-predict.".format(
                done_reason
            )
        )
        return

    if thinking:
        print("--- THINKING ---")
        print(thinking)
        print("-------------------")

    print("--- AI RESPONSE ---")
    print(content or "Error: Model returned empty response.")
    print("-------------------")
    if done_reason:
        print(f"done_reason: {done_reason}")


def main() -> None:
    """Run vehicle ReID comparison on two images using an Ollama vision model."""
    parser = argparse.ArgumentParser(
        description="Evaluate vehicle re-identification (same vs different car) via Ollama."
    )
    parser.add_argument("image1", help="Path to Image A")
    parser.add_argument("image2", help="Path to Image B")
    parser.add_argument(
        "--model",
        default="gemma4:e4b-it-q8_0",
        help="Ollama model tag (default: gemma4:e4b-it-q8_0)",
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model thinking mode (default: disabled to avoid empty content)",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=1024,
        help="Max tokens to generate (default: 1024; use 2048+ if --think)",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="Context window size (default: 8192)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    args = parser.parse_args()

    try:
        image1 = _validate_image_path(args.image1)
        image2 = _validate_image_path(args.image2)
    except FileNotFoundError as exc:
        print(f"An error occurred: {exc}")
        sys.exit(1)

    print(f"Loading {args.model} and analyzing images...\n")

    try:
        _pull_model(args.model)

        response = ollama.chat(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": VEHICLE_REID_PROMPT,
                    "images": [image1, image2],
                }
            ],
            think=args.think,
            options={
                "temperature": args.temperature,
                "num_predict": args.num_predict,
                "num_ctx": args.num_ctx,
            },
        )
        _print_response(response)

    except Exception as exc:  # noqa: BLE001 - surface Ollama/runtime errors cleanly at CLI
        print(f"An error occurred: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
