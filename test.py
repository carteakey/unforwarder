import base64
import os
import shutil
from io import BytesIO

from IPython.display import HTML, display
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def plt_img_base64(img_base64):
    """
    Display base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))

llm = ChatOllama(model="llava-llama3:latest", temperature=0)

def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]

chain = prompt_func | llm | StrOutputParser()

source_folder = "/Users/kchauhan/Downloads/whatsapp/WhatsApp Chat - God bless us all"
meme_folder = os.path.join(source_folder, "meme")
non_meme_folder = os.path.join(source_folder, "non-meme")

if not os.path.exists(meme_folder):
    os.makedirs(meme_folder)

if not os.path.exists(non_meme_folder):
    os.makedirs(non_meme_folder)

# Get list of files to process
files_to_process = [f for f in os.listdir(source_folder) if f.lower().endswith(".jpg")]

# Use tqdm to create a progress bar
for filename in tqdm(files_to_process, desc="Processing images"):
    print("Now processing:", filename)
    file_path = os.path.join(source_folder, filename)
    pil_image = Image.open(file_path)
    image_b64 = convert_to_base64(pil_image)

    response = chain.invoke(
        {"text": "Does this image contain any content that is not a personal photo, such as social media texts, screenshots, good morning messages, super blurry images, memes, or any other forwarded content? Just answer yes or no. Be strict in your filtering.", "image": image_b64}
    )
    print(response)
    if response.startswith("Yes"):
        shutil.move(file_path, os.path.join(meme_folder, filename))
    elif response.startswith("No"):
        shutil.move(file_path, os.path.join(non_meme_folder, filename))

print("Processing complete.")
