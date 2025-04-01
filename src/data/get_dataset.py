import time
import polars as pl
from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract, extract_metadata


def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    """
    Get the sitemap from the given url
    """
    urls = sitemap_search(sitemap_url)
    return urls


def create_dataset(list_of_websites: list) -> pl.DataFrame:
    """
    Create a dataset from the given list of websites
    """

    data = []

    for website in tqdm(list_of_websites, desc="Websites"):
        urls = get_urls_from_sitemap(website)
        print(f"Found {len(urls)} urls in {website}")
        for url in tqdm(urls[:10], desc="Urls"):
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
            time.sleep(.5)

    df = pl.DataFrame(data)
    df = df.drop()
    df = df.drop_nulls()
    return df


if __name__ == "__main__":
    list_of_websites = [
        "https://python.langchain.com/"
    ]
    df = create_dataset(list_of_websites=list_of_websites)
    df.write_csv("dataset.csv")
