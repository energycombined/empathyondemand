import os
import base64
import gradio as gr
import torch
# Load the APP environment variable
app_code = os.getenv("APP")

if not app_code:
    raise ValueError("The APP environment variable is not set.")

# Decode the app_code (assuming it's encoded, e.g., using Base64)
decoded_app_code = base64.b64decode(app_code).decode("utf-8")

# Execute the decoded code in the global scope
exec(decoded_app_code)

# If the code contains a Gradio app (like `demo`), it will launch when executed
if __name__ == "__main__":
    demo.launch()
