"""Voice transcription module using OpenAI's Whisper model."""
import os
import time
import tempfile
from typing import Optional, Tuple

import requests
import whisper
import torch
from loguru import logger


class Whisper:
    """A wrapper around OpenAI's Whisper model for audio transcription.
    
    This class handles loading the Whisper model and transcribing audio files.
    The model is loaded once during initialization for better performance.
    
    Attributes:
        model: The loaded Whisper model
        model_size: Size of the Whisper model (e.g., 'tiny', 'base', 'small', 'medium', 'large')
        device: The device (CPU/GPU) where the model is loaded
    """
    
    def __init__(self, model_size: str = 'tiny'):
        """Initialize the Whisper transcriber with the specified model size.
        
        Args:
            model_size: Size of the Whisper model to use. Options are:
                      'tiny', 'base', 'small', 'medium', 'large'
        """
        self.model_size = model_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading Whisper {self.model_size} model on {self.device}...")
        self.model = whisper.load_model(self.model_size, device=self.device)
        logger.info(f"Whisper {self.model_size} model loaded successfully")
    
    def transcribe(self, audio_file: str) -> Tuple[str, str]:
        """Transcribe an audio file to text.
        
        Args:
            audio_file: Path to the audio file to transcribe
            
        Returns:
            Tuple containing (transcribed_text, detected_language)
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            RuntimeError: If transcription fails
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        try:
            start_time = time.time()
            logger.info(f"Transcribing audio file: {audio_file}")
            
            # Transcribe with the model
            result = self.model.transcribe(
                audio_file,
                fp16=(self.device == 'cuda')  # Use FP16 if on GPU
            )
            
            transcription_time = time.time() - start_time
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
            
            return result["text"].strip(), result.get("language", "")
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise RuntimeError(f"Failed to transcribe audio: {str(e)}")
    
    def transcribe_audio_data(self, audio_data: bytes, language: Optional[str] = None) -> str:
        """Transcribe raw audio data to text.
        
        Args:
            audio_data: Raw audio data as bytes
            language: Optional language code (e.g., 'en', 'es', 'fr'). 
                     If None, the model will auto-detect the language.
                     
        Returns:
            str: The transcribed text
            
        Raises:
            RuntimeError: If transcription fails
        """
        try:
            # Create a temporary file to store the audio data
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                result = self.transcribe(temp_file_path)
                return result[0]  # Return just the text, not the language
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file_path}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error in transcribe_audio_data: {str(e)}")
            raise RuntimeError(f"Failed to process audio data: {str(e)}")


def download_audio(recording_url: str, local_path: str = "temp_audio.mp3") -> str:
    """
    Download audio file from a URL (e.g., Twilio RecordingUrl).
    
    Args:
        recording_url: URL of the audio file to download
        local_path: Local path where the file should be saved
        
    Returns:
        str: Path to the downloaded file
        
    Raises:
        Exception: If download fails or returns non-200 status
    """
    # Ensure .mp3 is added if not present
    if not recording_url.endswith(".mp3"):
        recording_url += ".mp3"
    
    logger.info(f"Downloading audio from {recording_url}...")
    response = requests.get(recording_url)

    if response.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Audio downloaded successfully to {local_path}")
        return local_path
    else:
        error_msg = f"Failed to download audio from {recording_url}: HTTP {response.status_code}"
        logger.error(error_msg)
        raise Exception(error_msg)

if __name__ == "__main__":
    
    # Create a global instance for easy import
    whisper_transcriber = Whisper(model_size='tiny')
