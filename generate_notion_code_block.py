
import json
from pathlib import Path

def generate_code_block_json(file_path):
    try:
        text = Path(file_path).read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Split into 2000-char chunks
    chunks = []
    chunk_size = 1800
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    
    # Create a list of code blocks, one for each chunk
    blocks = []
    for chunk in chunks:
        blocks.append({
            "object": "block",
            "type": "code",
            "code": {
                "language": "python",
                "rich_text": [{
                    "type": "text",
                    "text": {
                        "content": chunk
                    }
                }]
            }
        })

    # Output as a list of children
    Path('code_blocks_multiple.json').write_text(json.dumps(blocks, indent=2), encoding='utf-8') 

if __name__ == "__main__":
    generate_code_block_json("app.py")
