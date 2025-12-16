import requests
from bs4 import BeautifulSoup


def extract_text_from_url(url):
    """Fetch and extract readable text from a webpage."""
    response = requests.get(url)
    response.raise_for_status()  # Raise error if request failed
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style tags
    for script in soup(["script", "style"]):
        script.decompose()

    text = soup.get_text(separator="\n")

    # Remove extra whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
