import os
import sys
import argparse
import requests
import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from typing import Dict, List, Optional

def call_anthropic_api(prompt: str, system_prompt: str) -> str:
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15",
        },
        json={
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 8000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    response.raise_for_status()
    return response.json()['content'][0]['text']

def extract_content(text: str, tag: str) -> Optional[str]:
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1).strip() if match else None

def process_file(file_path: str, project_files: List[str], doc_files: List[str], save_intermediate: bool) -> Dict[str, str]:
    with open(file_path, 'r') as f:
        source_code = f.read()
    
    full_project = "\n".join(open(f, 'r').read() for f in project_files)
    docs_content = "\n".join(open(f, 'r').read() for f in doc_files)
    
    system_prompt = """
    You are an expert Python developer. Refactor and improve the provided code.
    Always include your response in the following format:
    <REFACTORED_CODE>
    [Your refactored code here]
    </REFACTORED_CODE>
    <REVIEW>
    [Your review here]
    </REVIEW>
    """
    prompt = f"Project context:\n{full_project}\n\nAdditional docs:\n{docs_content}\n\nRefactor this code:\n{source_code}"
    
    response = call_anthropic_api(prompt, system_prompt)
    
    refactored_code = extract_content(response, 'REFACTORED_CODE')
    review = extract_content(response, 'REVIEW')
    
    if not refactored_code or not review:
        print(f"Error: Failed to extract refactored code or review for {file_path}")
        print("API Response:")
        print(response)
        return {}
    
    output_dir = os.path.join(os.path.dirname(file_path), '.refactoring')
    os.makedirs(output_dir, exist_ok=True)
    
    refactored_file = os.path.join(output_dir, f"refactored_{os.path.basename(file_path)}")
    review_file = os.path.join(output_dir, f"review_{os.path.basename(file_path)}.txt")
    
    with open(refactored_file, 'w') as f:
        f.write(refactored_code)
    
    with open(review_file, 'w') as f:
        f.write(review)
    
    if save_intermediate:
        with open(os.path.join(output_dir, f"intermediate_{os.path.basename(file_path)}"), 'w') as f:
            f.write(response)
    
    print(f"Successfully processed {file_path}")
    return {"original": file_path, "refactored": refactored_file, "review": review_file}


def extract_from_xml(xml_string: str) -> Dict[str, str]:
    try:
        # Attempt to parse the XML
        root = ET.fromstring(xml_string)
        return {
            "original_code": root.find("original_code").text.strip() if root.find("original_code") is not None else "",
            "refactored_code": root.find("refactored_code").text.strip() if root.find("refactored_code") is not None else "",
            "review": root.find("review").text.strip() if root.find("review") is not None else "",
            "merge_status": root.find("merge_status").text.strip() if root.find("merge_status") is not None else "incomplete"
        }
    except ET.ParseError:
        # If parsing fails, attempt to extract content using regex
        print("XML parsing failed. Attempting to extract content using regex.")
        return {
            "original_code": extract_content(xml_string, "original_code") or "",
            "refactored_code": extract_content(xml_string, "refactored_code") or "",
            "review": extract_content(xml_string, "review") or "",
            "merge_status": extract_content(xml_string, "merge_status") or "incomplete"
        }

def wrap_in_xml(original_code: str, refactored_code: str, review: str, merge_status: str = "incomplete") -> str:
    return f"""
    <code_review>
        <original_code>
        <![CDATA[
        {original_code}
        ]]>
        </original_code>
        <refactored_code>
        <![CDATA[
        {refactored_code}
        ]]>
        </refactored_code>
        <review>
        <![CDATA[
        {review}
        ]]>
        </review>
        <merge_status>{escape(merge_status)}</merge_status>
    </code_review>
    """
def merge_changes(original_code: str, refactored_code: str, review: str) -> str:
    system_prompt = """
    You are an expert Python developer tasked with merging code changes.
    Analyze the original code, the refactored code, and the review.
    Merge the changes, ensuring the final code is complete, fully functional, and incorporates all improvements.
    If the merge is incomplete, explain what needs to be done in the next turn.
    Your response should be in the following XML format:

    <code_review>
        <original_code>
        [Original code here]
        </original_code>
        <refactored_code>
        [Merged code here]
        </refactored_code>
        <review>
        [Your review of the changes and merge process]
        </review>
        <merge_status>[complete/incomplete]</merge_status>
    </code_review>
    """

    prompt = f"""
    Please merge the following code changes:

    Original Code:
    ```python
    {original_code}
    ```

    Refactored Code:
    ```python
    {refactored_code}
    ```

    Review:
    {review}

    Merge the changes, ensuring the final code is complete and fully functional.
    If the merge is incomplete, explain what needs to be done in the next turn.
    """

    return call_anthropic_api(prompt, system_prompt)

def finalize_merge(original_file: str, refactored_file: str, review_file: str, output_file: str, max_turns: int = 3) -> None:
    with open(original_file, 'r') as f:
        original_code = f.read()
    with open(refactored_file, 'r') as f:
        refactored_code = f.read()
    with open(review_file, 'r') as f:
        review = f.read()

    for turn in range(max_turns):
        try:
            merged_xml = merge_changes(original_code, refactored_code, review)
            merged_data = extract_from_xml(merged_xml)

            if merged_data["merge_status"].lower() == "complete":
                with open(output_file, 'w') as f:
                    f.write(merged_data["refactored_code"])
                print(f"Merge completed in {turn + 1} turns. Output saved to {output_file}")
                print(f"Final review:\n{merged_data['review']}")
                return

            print(f"Merge incomplete after turn {turn + 1}. Continuing...")
            original_code = merged_data["original_code"]
            refactored_code = merged_data["refactored_code"]
            review = merged_data["review"]
        except Exception as e:
            print(f"Error during merge (turn {turn + 1}): {str(e)}")
            print("Attempting to continue with the merge process...")

    print(f"Merge incomplete after {max_turns} turns. Please review the latest output:")
    print(merged_data["review"])
    with open(output_file, 'w') as f:
        f.write(merged_data["refactored_code"])
    print(f"Latest merged code saved to {output_file}")
def main():
    parser = argparse.ArgumentParser(description="Improve and merge code using Claude API")
    parser.add_argument("--project", required=True, help="Project directory or file")
    parser.add_argument("--target", nargs='+', help="Target file(s) or directory to improve")
    parser.add_argument("--docs", nargs='+', help="Additional document files")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results")
    args = parser.parse_args()

    project_files = []
    for root, _, files in os.walk(args.project):
        project_files.extend(os.path.join(root, f) for f in files if f.endswith(('.py', '.txt', '.md')))

    doc_files = args.docs or []

    targets = args.target or [args.project]
    for target in targets:
        if os.path.isfile(target):
            result = process_file(target, project_files, doc_files, args.save_intermediate)
            if result:
                finalize_merge(result["original"], result["refactored"], result["review"], 
                               os.path.join(os.path.dirname(target), f"final_{os.path.basename(target)}"))
        elif os.path.isdir(target):
            for root, _, files in os.walk(target):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        result = process_file(file_path, project_files, doc_files, args.save_intermediate)
                        if result:
                            finalize_merge(result["original"], result["refactored"], result["review"], 
                                           os.path.join(root, f"final_{file}"))
        else:
            print(f"Error: {target} is not a valid file or directory")

if __name__ == "__main__":
    main()