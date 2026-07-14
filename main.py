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

    # 2. Define the System Prompt
    system_prompt = """<|think|> You are a forensic vehicle identification AI. Analyze the two provided images and determine if they show the exact same physical vehicle. 

You must methodically analyze the following parameters in order:

1. Macro Features: Compare Make, Model, Body Style, and Color (accounting for lighting).
2. Micro Features: Compare structural geometry (grille shape, headlamp angles) and unique identifiers (license plates, stickers, visible damage).
3. Image Context: Acknowledge if one image is pixelated or taken from a skewed angle, and reason whether differences are physical or just camera distortions.

First, write out your step-by-step reasoning. 
At the very end of your response, you must output exactly two lines in this format:
Verdict: [MATCH / MISMATCH]
Confidence: [0-100]%"""

    print(f"Loading {args.model} and analyzing images...\n")

    try:
        # 3. Call the Ollama Python API
        response = ollama.chat(
            model=args.model,
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': 'Are the vehicles in these two images the exact same car?',
                    'images': [args.image1, args.image2]
                }
            ],
            options={
                "temperature": 0.1,  # Keep it deterministic
                "num_predict": 400  # Give it enough room to reason and print the score
            }
        )

        # 4. Print the output
        print("--- AI RESPONSE ---")
        print(response['message']['content'])
        print("-------------------")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
