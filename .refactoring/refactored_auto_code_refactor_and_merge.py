import os
import argparse
import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from functools import wraps
import aiohttp
from dotenv import load_dotenv
from lxml import etree
import aiofiles
import tiktoken
from pydantic_settings import BaseSettings
import random
import pytest
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Constants
XML_TAGS = ["REVIEW", "REFACTORED_CODE", "TESTS"]
OUTPUT_DIR = ".refactoring"
LOG_FILE = "code_refactor.log"

# Configure logging
def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, mode='w')
        ]
    )

configure_logging()
logger = logging.getLogger(__name__)

encoder = tiktoken.encoding_for_model("gpt-4")

@dataclass
class APIResponse:
    content: str
    status: str

@dataclass
class CodeProcessingResult:
    original: str
    refactored: str
    review: str
    tests: Optional[str] = None

class Config(BaseSettings):
    BASE_URL: str = "https://api.anthropic.com/v1/messages"
    MODEL: str = "claude-3-opus-20240229"
    MAX_TOKENS: int = 4096
    MAX_RETRIES: int = 5
    TIMEOUT: int = 160
    API_KEY: str

    model_config = {
        "env_prefix": "ANTHROPIC_"
    }

class AnthropicAPI:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @staticmethod
    def retry_with_exponential_backoff(func):
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            max_retries = Config().MAX_RETRIES
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except aiohttp.ClientResponseError as e:
                    if e.status == 429:
                        wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                        logger.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
            raise Exception("Exceeded maximum retry attempts for API requests")
        return wrapper

    @retry_with_exponential_backoff
    async def call_api(self, prompt: str, system_prompt: str) -> APIResponse:
        headers = {
            "x-api-key": self.config.API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.config.MODEL,
            "max_tokens": self.config.MAX_TOKENS,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        }
        logger.debug(f"prompt tokens: {len(encoder.encode(payload['messages'][0]['content']))}")
        logger.debug(f"system tokens: {len(encoder.encode(payload['system']))}")
        
        if not self.session:
            raise RuntimeError("API client session not initialized")

        async with self.session.post(self.config.BASE_URL, headers=headers, json=payload, timeout=self.config.TIMEOUT) as response:
            response.raise_for_status()
            data = await response.json()
            content = data['content'][0]['text']
            return APIResponse(content=content, status="complete")

class CodeProcessor:
    @staticmethod
    def extract_content(text: str, tag: str) -> Optional[str]:
        match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    async def process_file(file_path: str, project_files: List[str], doc_files: List[str], save_intermediate: bool, create_review: bool, generate_features: bool, generate_tests: bool, user_prompt: Optional[str] = None) -> Optional[CodeProcessingResult]:
        if os.path.basename(file_path).startswith('.'):
            logger.info(f"Skipping hidden file: {file_path}")
            return None

        try:
            source_code = await FileUtils.read_file(file_path)
            full_project = await FileUtils.read_files(project_files)
            docs_content = await FileUtils.read_files(doc_files)
            
            system_prompt = get_system_prompt(generate_features, generate_tests, user_prompt)
            prompt = f"""
            Project context:
            {full_project}

            Additional docs:
            {docs_content}

            Refactor this code:
            {source_code}

            Remember, each file's content is wrapped in XML tags. Please analyze the content within these tags.
            """

            async with AnthropicAPI() as api:
                response = await api.call_api(prompt, system_prompt)
            
            refactored_code = CodeProcessor.extract_content(response.content, 'REFACTORED_CODE')
            review = CodeProcessor.extract_content(response.content, 'REVIEW')
            tests = CodeProcessor.extract_content(response.content, 'TESTS') if generate_tests else None
            
            if not refactored_code or not review:
                logger.error(f"Failed to extract refactored code or review for {file_path}")
                logger.error(f"API Response: {response.content}")
                return None
            
            output_dir = os.path.join(os.path.dirname(file_path), OUTPUT_DIR)
            os.makedirs(output_dir, exist_ok=True)
            
            refactored_file = os.path.join(output_dir, f"refactored_{os.path.basename(file_path)}")
            review_file = os.path.join(output_dir, f"review_{os.path.basename(file_path)}.txt")
            
            await FileUtils.write_file(refactored_file, refactored_code)
            await FileUtils.write_file(review_file, review)
            
            if tests:
                tests_file = os.path.join(output_dir, f"tests_{os.path.basename(file_path)}")
                await FileUtils.write_file(tests_file, tests)
            
            if save_intermediate:
                await FileUtils.write_file(os.path.join(output_dir, f"intermediate_{os.path.basename(file_path)}"), response.content)
            
            logger.info(f"Successfully processed {file_path}")
            return CodeProcessingResult(original=file_path, refactored=refactored_file, review=review_file, tests=tests_file if tests else None)
        except Exception as e:
            logger.exception(f"Error processing file {file_path}")
            return None

class XMLHandler:
    @staticmethod
    def extract_from_xml(xml_string: str) -> Dict[str, str]:
        try:
            root = etree.fromstring(xml_string)
            return {
                "original_code": ''.join(root.xpath("//original_code/text()")),
                "refactored_code": ''.join(root.xpath("//refactored_code/text()")),
                "review": ''.join(root.xpath("//review/text()")),
                "merge_status": ''.join(root.xpath("//merge_status/text()")),
                "tests": ''.join(root.xpath("//tests/text()"))
            }
        except etree.ParseError:
            logger.warning("XML parsing failed. Attempting to extract content using regex.")
            return {
                "original_code": CodeProcessor.extract_content(xml_string, "original_code") or "",
                "refactored_code": CodeProcessor.extract_content(xml_string, "refactored_code") or "",
                "review": CodeProcessor.extract_content(xml_string, "review") or "",
                "merge_status": CodeProcessor.extract_content(xml_string, "merge_status") or "incomplete",
                "tests": CodeProcessor.extract_content(xml_string, "tests") or ""
            }

    @staticmethod
    def wrap_in_xml(original_code: str, refactored_code: str, review: str, merge_status: str = "incomplete", tests: str = "") -> str:
        root = etree.Element("code_review")
        etree.SubElement(root, "review").text = etree.CDATA(review)
        etree.SubElement(root, "original_code").text = etree.CDATA(original_code)
        etree.SubElement(root, "refactored_code").text = etree.CDATA(refactored_code)
        etree.SubElement(root, "merge_status").text = merge_status
        etree.SubElement(root, "tests").text = etree.CDATA(tests)
        return etree.tostring(root, pretty_print=True, encoding="unicode")

class CodeMerger:
    def __init__(self, api: AnthropicAPI):
        self.api = api

    async def merge_changes(self, original_code: str, refactored_code: str, review: str, tests: str = "") -> APIResponse:
        system_prompt = """
        You are an expert Python developer tasked with merging code changes.
        The code you receive will be wrapped in XML tags. Analyze the content within these tags.
        Merge the changes, ensuring the final code is complete, fully functional, and incorporates all improvements.
        If the merge is incomplete, explain what needs to be done in the next turn.
        Your response should be in the following XML format:

        <code_review>
            <review>
            [Your review of the changes and merge process]
            </review>
    
            <refactored_code>
            [Merged code here]
            </refactored_code>
            
            <merge_status>
            [complete/incomplete]
            </merge_status>

            <tests>
            [Updated or new tests here]
            </tests>
        </code_review>
        """

        prompt = f"""
        Please merge the following code changes:

        Original Code:
        {original_code}

        Refactored Code:
        {refactored_code}

        Review:
        {review}

        Tests:
        {tests}

        Merge the changes, ensuring the final code is complete and fully functional.
        If the merge is incomplete, explain what needs to be done in the next turn.
        Update or create new tests as necessary.
        """

        return await self.api.call_api(prompt, system_prompt)

    async def finalize_merge(self, original_file: str, refactored_file: str, review_file: str, tests_file: str, output_file: str, max_turns: int = 3) -> None:
        original_code = await FileUtils.read_file(original_file)
        refactored_code = await FileUtils.read_file(refactored_file)
        review = await FileUtils.read_file(review_file)
        tests = await FileUtils.read_file(tests_file) if tests_file else ""

        for turn in range(max_turns):
            try:
                merged_response = await self.merge_changes(original_code, refactored_code, review, tests)
                merged_data = XMLHandler.extract_from_xml(merged_response.content)

                if merged_data["merge_status"].lower() == "complete":
                    await FileUtils.write_file(output_file, merged_data["refactored_code"])
                    if tests_file:
                        await FileUtils.write_file(tests_file, merged_data["tests"])
                    logger.info(f"Merge completed in {turn + 1} turns. Output saved to {output_file}")
                    logger.info(f"Final review:\n{merged_data['review']}")
                    return

                logger.info(f"Merge incomplete after turn {turn + 1}. Continuing...")
                original_code = merged_data["original_code"]
                refactored_code = merged_data["refactored_code"]
                review = merged_data["review"]
                tests = merged_data["tests"]
            except Exception as e:
                logger.exception(f"Error during merge (turn {turn + 1})")
                logger.info("Attempting to continue with the merge process...")

        logger.warning(f"Merge incomplete after {max_turns} turns. Please review the latest output:")
        logger.info(merged_data["review"])
        await FileUtils.write_file(output_file, merged_data["refactored_code"])
        if tests_file:
            await FileUtils.write_file(tests_file, merged_data["tests"])
        logger.info(f"Latest merged code saved to {output_file}")

        os.rename(output_file, original_file)
        logger.info(f"Renamed {output_file} to {original_file}")

class FileUtils:
    @staticmethod
    async def read_file(file_path: str) -> str:
        """Read file and wrap its content with XML tags based on the file name."""
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            wrapped_content = f"<{os.path.basename(file_path)}><![CDATA[{content}]]></{os.path.basename(file_path)}>"
            return wrapped_content

    @staticmethod
    async def write_file(file_path: str, content: str) -> None:
        """Write content to file"""
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(content)

    @staticmethod
    async def read_files(file_paths: List[str]) -> str:
        """Read multiple files and wrap each individual file's content with XML tags"""
        contents = await asyncio.gather(*[FileUtils.read_file(f) for f in file_paths if not os.path.basename(f).startswith('.')])
        return "\n".join(contents)

def get_system_prompt(generate_features: bool, generate_tests: bool, user_prompt: Optional[str] = None) -> str:
    base_prompt = """
    You are an expert Python developer. The code and documentation you will receive is wrapped in XML tags.
    Each file's content is enclosed in tags named after the file (e.g., <filename.py>).
    Please analyze the content within these tags.
    Always include your response in the following format:
    <REVIEW>
    [Your review here]
    </REVIEW>
    <REFACTORED_CODE>
    [Your refactored code here]