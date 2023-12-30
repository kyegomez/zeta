import inspect
import os
import sys
import threading

from dotenv import load_dotenv

from scripts.auto_tests_docs.docs import DOCUMENTATION_WRITER_SOP
from swarms import OpenAIChat
from zeta.ops import *

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIChat(
    model_name="gpt-4-1106-preview",
    openai_api_key=api_key,
    max_tokens=2000,
)


def process_documentation(item):
    """
    Process the documentation for a given function using OpenAI model and save it in a Markdown file.
    """
    try:
        doc = inspect.getdoc(item)
        source = inspect.getsource(item)
        input_content = (
            f"Name: {item.__name__}\n\nDocumentation:\n{doc}\n\nSource"
            f" Code:\n{source}"
        )

        # Process with OpenAI model
        processed_content = model(
            DOCUMENTATION_WRITER_SOP(input_content, "zeta.ops")
        )

        doc_content = f"# {item.__name__}\n\n{processed_content}\n"

        # Create the directory if it doesn't exist
        dir_path = "docs/zeta/ops"
        os.makedirs(dir_path, exist_ok=True)

        # Write the processed documentation to a Markdown file
        file_path = os.path.join(dir_path, f"{item.__name__.lower()}.md")
        with open(file_path, "w") as file:
            file.write(doc_content)

        print(f"Succesfully processed {item.__name__}.")
    except Exception as e:
        print(f"Error processing {item.__name__}: {e}")


def main():
    # Gathering all functions from the zeta.ops module
    functions = [
        obj
        for name, obj in inspect.getmembers(sys.modules["zeta.ops"])
        if inspect.isfunction(obj)
    ]

    threads = []
    for func in functions:
        thread = threading.Thread(target=process_documentation, args=(func,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("Documentation generated in 'docs/zeta/ops' directory.")


if __name__ == "__main__":
    main()
