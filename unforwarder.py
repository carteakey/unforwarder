import base64
import os
import shutil
import logging
from io import BytesIO

from IPython.display import HTML, display
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm
from load_dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def convert_to_base64(pil_image: Image.Image) -> str:
    """
    Convert a PIL Image to a Base64 encoded JPEG string.
    
    Args:
        pil_image: PIL Image object.
    
    Returns:
        Base64 encoded string of the image in JPEG format.
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def display_base64_image(img_base64: str) -> None:
    """
    Display a base64 encoded JPEG image in the IPython display.
    
    Args:
        img_base64: Base64 encoded image string.
    """
    html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    display(HTML(html))


def create_llm() -> ChatOllama:
    """
    Initialize and return a ChatOllama instance using environment variables.
    """
    return ChatOllama(
        base_url=os.environ.get("OLLAMA_SERVER_URL"),
        model=os.environ.get("OLLAMA_MODEL"),
        temperature=0,
    )


def prompt_func(data: dict) -> list:
    """
    Create a prompt message for the LLM from image and text data.
    
    Args:
        data: Dictionary containing 'text' and 'image' keys.
    
    Returns:
        A list containing a HumanMessage with the specified content parts.
    """
    text = data.get("text", "")
    image = data.get("image", "")

    content_parts = [
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"},
        {"type": "text", "text": text},
    ]
    return [HumanMessage(content=content_parts)]


def process_images(source_folder: str) -> None:
    """
    Process images from the source folder using an LLM to decide where to move the images.
    
    Args:
        source_folder: Path to the folder containing the .jpg images.
    """
    save_folder = os.path.join(source_folder, "save")
    review_folder = os.path.join(source_folder, "review-n-delete")

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(review_folder, exist_ok=True)

    files_to_process = [f for f in os.listdir(source_folder) if f.lower().endswith(".jpg")]

    # Initialize LLM chain (compose prompt_func, LLM and output parser)
    llm = create_llm()
    chain = prompt_func | llm | StrOutputParser()

    for filename in tqdm(files_to_process, desc="Processing images"):
        file_path = os.path.join(source_folder, filename)
        try:
            pil_image = Image.open(file_path)
        except Exception as e:
            logging.error(f"Failed to open {filename}: {e}")
            continue

        image_b64 = convert_to_base64(pil_image)
        data = {
            "text": ("Is this image one of: social media texts, screenshots, good morning messages, "
                     "super blurry images, memes, or any other forwarded content? Just answer yes or no."),
            "image": image_b64,
        }
        try:
            response = chain.invoke(data)
        except Exception as e:
            logging.error(f"LLM processing failed for {filename}: {e}")
            continue

        logging.info(f"Response for {filename}: {response}")

        if response.strip().lower().startswith("no"):
            destination = os.path.join(save_folder, filename)
        elif response.strip().lower().startswith("yes"):
            destination = os.path.join(review_folder, filename)
        else:
            logging.warning(f"Unexpected response for {filename}: {response}. Moving to review folder.")
            destination = os.path.join(review_folder, filename)

        try:
            shutil.move(file_path, destination)
        except Exception as e:
            logging.error(f"Failed to move {filename} to {destination}: {e}")

    logging.info("Processing complete.")


if __name__ == "__main__":
    source_folder = os.environ.get("SOURCE_FOLDER")
    process_images(source_folder)

