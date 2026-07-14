import argparse
import sys

import ollama


def main():
    # 1. Setup Argparse for CLI inputs
    parser = argparse.ArgumentParser(description="Evaluate Vehicle Re-ID using Gemma 4 E4B via Ollama.")
    parser.add_argument("image1", help="Path to the first image")
    parser.add_argument("image2", help="Path to the second image")
    parser.add_argument("--model", default="gemma4:e4b", help="Ollama model to use (default: gemma4:e4b)")

    args = parser.parse_args()

    # Combine everything into a single user prompt
    combined_prompt = """You are a forensic vehicle identification AI. Analyze the two provided images and determine if they show the exact same physical vehicle. 

    You must methodically analyze the following parameters in order:

    1. Macro Features: Compare Make, Model, Body Style, and Color (accounting for lighting).
    2. Micro Features: Compare structural geometry (grille shape, headlamp angles) and unique identifiers (license plates, stickers, visible damage).
    3. Image Context: Acknowledge if one image is pixelated or taken from a skewed angle, and reason whether differences are physical or just camera distortions.

    First, write out your step-by-step reasoning. 
    At the very end of your response, you must output exactly two lines in this format:
    Verdict: [MATCH / MISMATCH / UNCERTAIN]
    Confidence: [0-100]%

    Are the vehicles in these two images the exact same car?"""

    print(f"Loading {args.model} and analyzing images...\n")

    try:
        # Force Ollama to download the model and stream the progress
        print("Starting download (if not already cached)...")
        for progress in ollama.pull(args.model, stream=True):
            status = progress.get('status', '')
            if 'total' in progress and 'completed' in progress and progress['total'] > 0:
                percent = (progress['completed'] / progress['total']) * 100
                digest = progress.get('digest', '')[7:17]
                print(f"Downloading layer [{digest}] ... {percent:.1f}%", end='\r')
            else:
                print(f"{status.capitalize()}" + " " * 20, end='\r')

        print("\n\nModel ready! Analyzing images...\n")

        # Call the Ollama Python API with a single 'user' message
        response = ollama.chat(
            model=args.model,
            messages=[
                {
                    'role': 'user',
                    'content': combined_prompt,
                    'images': [args.image1, args.image2]
                }
            ],
            options={
                "temperature": 0.1,
                "num_predict": 800,  # Increased to ensure it doesn't cut off early
                "num_ctx": 8192  # Explicitly allocate 8k context window for dual images
            }
        )

        # Print the output
        print("--- AI RESPONSE ---")
        print(response.get('message', {}).get('content', 'Error: Model returned empty response again.'))
        print("-------------------")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
