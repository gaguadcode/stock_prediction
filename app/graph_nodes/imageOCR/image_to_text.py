import base64
import requests
from app.utils.config import MISTRAL_API_KEY
from app.utils.logger import get_logger

class MistralOCR:
    """
    Extracts text from a Base64 encoded image using the Mistral API.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.api_url = "https://api.mistral.ai/v1/ocr"  # Replace with actual Mistral OCR API endpoint
        self.headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

    def process_image(self, base64_image: str) -> str:
        """
        Sends a Base64-encoded image to the Mistral API for OCR text extraction.

        Args:
            base64_image (str): Base64-encoded image string.

        Returns:
            str: Extracted text from the image.
        """
        try:
            self.logger.info("📤 Sending image to Mistral API for OCR processing...")

            payload = {"image": base64_image}

            response = requests.post(self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()  # Raises an error for HTTP failure codes

            data = response.json()
            extracted_text = data.get("text", "")

            self.logger.info(f"✅ Successfully extracted text: {extracted_text}")
            return extracted_text

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Request to Mistral API failed: {e}")
            return ""

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """
        Converts an image file to a Base64-encoded string.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64-encoded string of the image.
        """
        try:
            with open(image_path, "rb") as image_file:
                base64_string = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_string
        except Exception as e:
            raise ValueError(f"❌ Failed to encode image to Base64: {e}")
