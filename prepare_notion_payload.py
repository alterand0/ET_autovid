import json
import textwrap

def split_text(text, chunk_size=2000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def create_payload():
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            code_content = f.read()
    except FileNotFoundError:
        print("Error: app.py not found")
        return

    code_chunks = split_text(code_content)
    code_rich_text = [{"type": "text", "text": {"content": chunk}} for chunk in code_chunks]

    payload = [
        {
            "object": "block",
            "type": "toggle",
            "toggle": {
                "rich_text": [{"type": "text", "text": {"content": "07/02/2026"}}]
            },
            "children": [
                {
                    "object": "block",
                    "type": "toggle",
                    "toggle": {
                        "rich_text": [{"type": "text", "text": {"content": "Código (app.py)"}}]
                    },
                    "children": [
                        {
                            "object": "block",
                            "type": "code",
                            "code": {
                                "language": "python",
                                "rich_text": code_rich_text
                            }
                        }
                    ]
                },
                {
                    "object": "block",
                    "type": "toggle",
                    "toggle": {
                        "rich_text": [{"type": "text", "text": {"content": "Cambios funcionales"}}]
                    },
                    "children": [
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": "Reorganización de la Interfaz de Usuario: Flujo lógico (Contenido -> Configuración -> Cierre -> Generar)."}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": "Eliminación completa de la funcionalidad 'Mosca' (Logo overlay)."}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": "Corrección de ubicación de opciones de Música y Cierre en modo Manual."}}]
                            }
                        }
                    ]
                }
            ]
        }
    ]

    with open('notion_payload.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("Payload generated in notion_payload.json")

if __name__ == "__main__":
    create_payload()
