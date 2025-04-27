import time
from pathlib import Path
import polars as pl
from git import Repo
from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract, extract_metadata
import re
import yaml


def clone_langchain_docs():
    """Clone or pull the latest LangChain docs"""
    repo_path = Path("./langchain-docs")
    if not repo_path.exists():
        print("Cloning LangChain repository...")
        Repo.clone_from("https://github.com/langchain-ai/langchain.git", repo_path)
    else:
        print("Pulling latest changes...")
        repo = Repo(repo_path)
        repo.remotes.origin.pull()
    return repo_path


def extract_yaml_and_content(file_path: Path) -> dict:
    """Extract YAML frontmatter and content from markdown file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split YAML frontmatter from content
        yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
        yaml_match = yaml_pattern.match(content)

        if yaml_match:
            # Extract YAML and content
            yaml_content = yaml_match.group(1)
            main_content = content[yaml_match.end():]
            try:
                metadata = yaml.safe_load(yaml_content)
            except yaml.YAMLError as e:
                print(f"Error parsing YAML in {file_path}: {e}")
                metadata = {}
        else:
            metadata = {}
            main_content = content

        # Convert file path to URL
        relative_path = str(file_path).split("docs/docs/")[1]
        url = f"https://python.langchain.com/{relative_path.replace('.mdx', '')}"

        # Extract title from metadata or first heading
        title = metadata.get('title', '')
        if not title:
            # Try to find first heading if title not in metadata
            heading_match = re.search(r'^#\s+(.+)$', main_content, re.MULTILINE)
            if heading_match:
                title = heading_match.group(1)
        
        return {
            "url": url,
            "title": title,
            "description": metadata.get('description', ''),
            "body": main_content.strip()
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def create_langchain_dataset() -> pl.DataFrame:
    """Create a dataset from LangChain documentation"""
    # Clone or update the repo
    repo_path = clone_langchain_docs()

    # Get all markdown files from the docs directory
    docs_path = repo_path / "docs/docs"
    markdown_files = list(docs_path.rglob("*.mdx"))

    print(f"Found {len(markdown_files)} documentation files")

    # Process each file
    data = []
    for file_path in tqdm(markdown_files, desc="Processing files"):
        content = extract_yaml_and_content(file_path)
        if content:
            data.append(content)

    # Create DataFrame
    df = pl.DataFrame(data)

    # Save the dataset
    output_path = "langchain_docs.csv"
    df.write_csv(output_path)
    print(f"Dataset saved to {output_path}")

    return df


def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    """
    Get the sitemap from the given url
    """
    urls = sitemap_search(sitemap_url)
    print(f"Found {len(urls)} urls in {sitemap_url}")
    return urls


def create_dataset(list_of_websites: list) -> pl.DataFrame:
    """
    Create a dataset from the given list of websites
    """

    data = []

    for website in tqdm(list_of_websites, desc="Websites"):
        urls = get_urls_from_sitemap(website)
        for url in tqdm(urls, desc="Urls"):
            html = fetch_url(url)
            print(html)
            body = extract(html)

            try:
                metadata = extract_metadata(html)
                title = metadata.title
                description = metadata.description
            except Exception as e:
                print(f"Error extracting metadata from {url}: {e}")
                metadata = ""
                title = ""
                description = ""

            d = {
                "url": url,
                "body": body,
                "description": description,
                "title": title,
            }

            data.append(d)
            time.sleep(1)

    df = pl.DataFrame(data)
    df = df.drop()
    df = df.drop_nulls()
    return df


if __name__ == "__main__":
    list_of_websites = [
        # "https://verycontent.it"
        "https://python.langchain.com/docs"
    ]
    # df = create_dataset(list_of_websites=list_of_websites)
    # df.write_csv("dataset.csv")
    df = create_langchain_dataset()
    df.write_csv("langchain_docs.csv")
