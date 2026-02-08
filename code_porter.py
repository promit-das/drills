"""Gradio UI to port Python code to high-performance C++ or Rust using LLM backends."""



# imports
import os
import io
import sys
from dotenv import load_dotenv
from gradio.themes.utils.colors import pink
from openai import OpenAI
import ollama
import gradio as gr
import subprocess
from IPython.display import Markdown, display

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
# anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
# grok_api_key = os.getenv('GROK_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

# Connect to client libraries
openai = OpenAI()

# anthropic_url = "https://api.anthropic.com/v1/"
# gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
# grok_url = "https://api.x.ai/v1"
groq_url = "https://api.groq.com/openai/v1"
ollama_url = "http://localhost:11434/v1"
openrouter_url = "https://openrouter.ai/api/v1"

# anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
# gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)
# grok = OpenAI(api_key=grok_api_key, base_url=grok_url)
groq = OpenAI(api_key=groq_api_key, base_url=groq_url)
ollama = OpenAI(api_key="ollama", base_url=ollama_url)
openrouter = OpenAI(api_key=openrouter_api_key, base_url=openrouter_url)

# model selection
models = [
    "gpt-5", 
    # "claude-sonnet-4-5-20250929", 
    # "grok-4", 
    # "gemini-2.5-pro", 
    "qwen2.5-coder", 
    "deepseek-coder-v2", 
    "gpt-oss:20b", 
    "qwen/qwen3-coder-30b-a3b-instruct", 
    "openai/gpt-oss-120b", 
    ]

clients = {
    "gpt-5": openai, 
    # "claude-sonnet-4-5-20250929": anthropic, 
    # "grok-4": grok, 
    # "gemini-2.5-pro": gemini,
    "qwen2.5-coder": ollama, 
    "deepseek-coder-v2": ollama, 
    "gpt-oss:20b": ollama, 
    "qwen/qwen3-coder-30b-a3b-instruct": openrouter,
    "openai/gpt-oss-120b": groq, 
    }

# fetch system info
from system_info import retrieve_system_info, rust_toolchain_info
system_info = retrieve_system_info()
rust_info = rust_toolchain_info()

"""
The following commented sections are meant to check your local for dependencies needed to compile and run C++ or Rust locally
Ignore these sections and run C++ on https://www.programiz.com/cpp-programming/online-compiler/ instead
"""

## message to check dependecies for compiling and running C++
# message = f"""
# Here is a report of the system information for my computer.
# I want to run a C++ compiler to compile a single C++ file called main.cpp and then execute it in the simplest way possible.
# Please reply with whether I need to install any C++ compiler to do this. If so, please provide the simplest step by step instructions to do so.

# If I'm already set up to compile C++ code, then I'd like to run something like this in Python to compile and execute the code:
# ```python
# compile_command = # something here - to achieve the fastest possible runtime performance
# compile_result = subprocess.run(compile_command, check=True, text=True, capture_output=True)
# run_command = # something here
# run_result = subprocess.run(run_command, check=True, text=True, capture_output=True)
# return run_result.stdout
# ```
# Please tell me exactly what I should use for the compile_command and run_command.

# System information:
# {system_info}
# """

## message to check dependecies for compiling and running Rust
# message = f"""
# Here is a report of the system information for my computer.
# I want to run a Rust compiler to compile a single rust file called main.rs and then execute it in the simplest way possible.
# Please reply with whether I need to install a Rust toolchain to do this. If so, please provide the simplest step by step instructions to do so.

# If I'm already set up to compile Rust code, then I'd like to run something like this in Python to compile and execute the code:
# ```python
# compile_command = # something here - to achieve the fastest possible runtime performance
# compile_result = subprocess.run(compile_command, check=True, text=True, capture_output=True)
# run_command = # something here
# run_result = subprocess.run(run_command, check=True, text=True, capture_output=True)
# return run_result.stdout
# ```
# Please tell me exactly what I should use for the compile_command and run_command.
# Have the maximum possible runtime performance in mind; compile time can be slow. Fastest possible runtime performance for this platform is key.
# Reply with the commands in markdown.

# System information:
# {system_info}

# Rust toolchain information:
# {rust_info}
# """

# response = openai.chat.completions.create(model=models[0], messages=[{"role": "user", "content": message}])
# display(Markdown(response.choices[0].message.content))

# # on the CLI run the above commands needed to compile and run C++ and/or Rust locally.


# select language
language = "C++"
# language = "Rust" 
extension = "rs" if language == "Rust" else "cpp"

# system prompt
system_prompt = f"""
Your task is to convert Python code into high performance {language} code.
Respond only with {language} code. Do not provide any explanation other than occasional comments.
The {language} response needs to produce an identical output in the fastest possible time.
"""

# user prompt
def user_prompt_for(python_code):
    return f"""
Port this Python code to {language} with the fastest possible implementation that produces identical output in the least time.
The system information is:
{system_info}
Write your response to a file called main.{extension}.
Respond only with {language} code.
Python code to port:

```python
{python_code}
```
"""

# # user prompt for compiling
# def user_prompt_for(python):
#     return f"""
# Port this Python code to {language} with the fastest possible implementation that produces identical output in the least time.
# The system information is:
# {system_info}
# Your response will be written to a file called main.{language} and then compiled and executed; the compilation command is:
# {compile_command}
# Respond only with {language} code.
# Python code to port:

# ```python
# {python}
# ```
# """


def messages_for(python):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(python)}
    ]

def write_output(code):
    with open(f"main.{extension}", "w") as f:
        f.write(code)

def port(model, python):
    client = clients[model]
    request = {"model": model, "messages": messages_for(python)}
    if client is openai and model == "gpt-5":
        request["reasoning_effort"] = "high"
    response = client.chat.completions.create(**request)
    reply = response.choices[0].message.content
    reply = reply.replace('```cpp','').replace('```rust','').replace('```','')
    return reply

def run_python(code):
    globals_dict = {"__builtins__": __builtins__}

    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    try:
        exec(code, globals_dict)
        output = buffer.getvalue()
    except Exception as e:
        output = f"Error: {e}"
    finally:
        sys.stdout = old_stdout

    return output

# # Use the commands from GPT 5
# def compile_and_run(code):
#     write_output(code)
#     try:
#         subprocess.run(compile_command, check=True, text=True, capture_output=True)
#         run_result = subprocess.run(run_command, check=True, text=True, capture_output=True)
#         return run_result.stdout
#     except subprocess.CalledProcessError as e:
#         return f"An error occurred:\n{e.stderr}"



# style the ui
from styles import CSS

# default to empty; user pastes code into the UI
python_code = ""

with gr.Blocks(title=f"Port from Python to {language}") as ui:
    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            python = gr.Code(
                label="Python (original)",
                value=python_code,
                language="python",
                lines=26
            )
        with gr.Column(scale=6):
            cpp = gr.Code(
                label=f"{language} (generated)",
                value="",
                language="cpp",
                lines=26
            )

    with gr.Row(elem_classes=["controls"]):
        python_run = gr.Button("Run Python", elem_classes=["run-btn", "py"])
        model = gr.Dropdown(models, value=models[0], show_label=False)
        convert = gr.Button(f"Port to {language}", elem_classes=["convert-btn"])
        # cpp_run = gr.Button(f"Run {language}", elem_classes=["run-btn", "cpp"])

    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            python_out = gr.TextArea(label="Python result", lines=8, elem_classes=["py-out"])
        # with gr.Column(scale=6):
        #     cpp_out = gr.TextArea(label=f"{language} result", lines=8, elem_classes=["cpp-out"])

    convert.click(fn=port, inputs=[model, python], outputs=[cpp])
    python_run.click(fn=run_python, inputs=[python], outputs=[python_out])
    # cpp_run.click(fn=compile_and_run, inputs=[cpp], outputs=[cpp_out])

ui.launch(inbrowser=True, css=CSS, theme=gr.themes.Monochrome())
