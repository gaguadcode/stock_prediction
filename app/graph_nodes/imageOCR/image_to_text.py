from app.utils.config import config
from app.utils.datatypes import WorkflowState
import base64
import mimetypes
import os
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from mistralai import Mistral

class OcrProcessor:
    """
    Handles the process of converting image content to text using Mistral OCR API
    and updates the workflow state with the extracted text.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OCR processor.
        
        Args:
            api_key: Mistral API key. If not provided, will use MISTRAL_API_KEY from config.
        """
        self.api_key = api_key or config.MISTRAL_API_KEY
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via config.MISTRAL_API_KEY")
        
        self.client = Mistral(api_key=self.api_key)
        self.model_name = "mistral-ocr-latest"
    
    def encode_image(self, image_path: str) -> Optional[str]:
        """
        Encode an image to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64-encoded string or None if error occurs
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: The file {image_path} was not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def get_mime_type(self, image_path: str) -> str:
        """
        Get the MIME type of an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MIME type string
        """
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            # Default to JPEG if mime type cannot be determined
            mime_type = "image/jpeg"
        return mime_type
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image through the Mistral OCR API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            OCR API response
        """
        base64_image = self.encode_image(image_path)
        if not base64_image:
            raise ValueError(f"Failed to encode image at {image_path}")
        
        mime_type = self.get_mime_type(image_path)
        
        ocr_response = self.client.ocr.process(
            model=self.model_name,
            document={
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{base64_image}"
            }
        )
        
        return ocr_response
    
    def extract_markdown(self, ocr_response) -> str:
        """
        Extract markdown content from the OCR response.
        
        Args:
            ocr_response: Response from the OCR API
            
        Returns:
            Extracted markdown content
        """
        markdown_content = ""
        
        if ocr_response.pages:
            for page in ocr_response.pages:
                if page.markdown:
                    markdown_content += page.markdown + "\n\n"
        
        return markdown_content.strip()
    
    def format_ocr_response_as_markdown(self, ocr_response) -> str:
        """
        Format the OCR response as markdown, including metadata.
        
        Args:
            ocr_response: Response from the OCR API
            
        Returns:
            Formatted markdown string with metadata
        """
        # Start with the basic markdown from the OCR response
        markdown_content = self.extract_markdown(ocr_response)
        
        # Add metadata as a header comment
        metadata = f"""
---
OCR Processing Information:
- Model: {ocr_response.model}
- Pages Processed: {ocr_response.usage_info.pages_processed if hasattr(ocr_response, 'usage_info') else 'Unknown'}
- Processing Date: {self._get_current_datetime()}
---

"""
        
        return metadata + markdown_content
    
    def _get_current_datetime(self) -> str:
        """Get current datetime as a formatted string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def save_markdown_to_file(self, markdown_content: str, output_path: str) -> str:
        """
        Save the markdown content to a file.
        
        Args:
            markdown_content: Markdown content to save
            output_path: Path where to save the file
            
        Returns:
            Path to the saved file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        return output_path
    
    def process_and_update_state(self, state: WorkflowState, image_path: str) -> WorkflowState:
        """
        Process an image and update the workflow state with the extracted markdown.
        
        Args:
            state: Current workflow state
            image_path: Path to the image file
            
        Returns:
            Updated workflow state
        """
        try:
            # Process the image
            ocr_response = self.process_image(image_path)
            
            # Format as markdown
            markdown_content = self.format_ocr_response_as_markdown(ocr_response)
            
            # Update the state
            state.user_input = markdown_content
            
            # Optional: save to file with same name as image but .md extension
            output_path = Path(image_path).with_suffix('.md')
            self.save_markdown_to_file(markdown_content, str(output_path))
            
            return state
            
        except Exception as e:
            # In case of errors, update state with error message
            error_message = f"Error during OCR processing: {str(e)}"
            state.user_input = error_message
            return state
    
    def process_directory_and_update_state(self, state: WorkflowState, directory_path: str, file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.pdf']) -> WorkflowState:
        """
        Process all images in a directory and update the workflow state with the combined markdown.
        
        Args:
            state: Current workflow state
            directory_path: Path to the directory containing images
            file_extensions: List of file extensions to process
            
        Returns:
            Updated workflow state
        """
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            state.user_input = f"Error: Directory {directory_path} does not exist or is not a directory."
            return state
        
        markdown_contents = []
        
        for ext in file_extensions:
            for file_path in path.glob(f"*{ext}"):
                try:
                    ocr_response = self.process_image(str(file_path))
                    markdown_content = self.format_ocr_response_as_markdown(ocr_response)
                    markdown_contents.append(f"# Document: {file_path.name}\n\n{markdown_content}")
                except Exception as e:
                    markdown_contents.append(f"# Document: {file_path.name}\n\nError during processing: {str(e)}")
        
        if markdown_contents:
            combined_markdown = "\n\n" + "-" * 50 + "\n\n".join(markdown_contents)
            state.user_input = combined_markdown
        else:
            state.user_input = f"No image files found in directory {directory_path} with extensions {file_extensions}."
        
        return state