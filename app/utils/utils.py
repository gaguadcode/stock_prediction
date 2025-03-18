import re
from urllib.parse import urlparse, urlunparse

def anonymize_database_url(database_url: str) -> str:
    """
    Anonymizes the database URL by masking the username and password.
    """
    
    parsed_url = urlparse(database_url)

    # Mask username and password
    netloc = f"{parsed_url.hostname}:{parsed_url.port}" if parsed_url.port else parsed_url.hostname
    sanitized_url = urlunparse((parsed_url.scheme, netloc, parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment))

    return sanitized_url

    