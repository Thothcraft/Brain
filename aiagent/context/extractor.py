#!/usr/bin/env python3

"""
HTML Extraction Utilities Module

This module provides functions for extracting structured data from HTML content.
It can identify form fields, search results, and other webpage elements.
It also includes functionality for performing web searches.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import requests


def extract_fields_from_page(html_content: str) -> Dict[str, Any]:
    """Extract form fields and structured data from a webpage's HTML content.

    This function identifies and extracts important elements from webpages such as:
    - Form fields (inputs, textareas, selects)
    - Search results (when on search engine pages)
    - Article content and key sections
    - Structured data (tables, lists)

    Args:
        html_content (str): The HTML content of the webpage.

    Returns:
        Dict[str, Any]: A dictionary containing structured information about the page:
            - form_fields: List of input elements and their attributes
            - search_results: If the page appears to be a search results page, the list of results
            - page_type: Detected type of page (e.g., "search", "article", "form", etc.)
            - key_elements: Important elements on the page

    Example:
        >>> html = "<html><body><form><input type='text' name='username'></form></body></html>"
        >>> fields = extract_fields_from_page(html)
        >>> print(f"Found {len(fields.get('form_fields', []))} form fields")
        Found 1 form fields
    """
    logging.info("Extracting structured data from HTML content")

    # Initialize results structure
    results = {
        "form_fields": [],
        "search_results": [],
        "page_type": "unknown",
        "key_elements": [],
    }

    # Return empty results if no HTML content provided
    if not html_content:
        logging.warning("No HTML content provided for extraction")
        return results

    try:
        # Try to import BeautifulSoup
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        # Extract form fields
        form_fields = []
        for field in soup.find_all(["input", "textarea", "select"]):
            field_info = {
                "type": field.name,
                "id": field.get("id", ""),
                "name": field.get("name", ""),
                "placeholder": field.get("placeholder", ""),
                "value": field.get("value", ""),
                "required": field.has_attr("required"),
            }
            # Add field type for input elements
            if field.name == "input":
                field_info["input_type"] = field.get("type", "text")
            form_fields.append(field_info)
        results["form_fields"] = form_fields

        # Detect if this is a search results page
        # Check for common search result patterns
        is_search_page = False

        # Check URL in meta tags
        meta_url = soup.find("meta", property="og:url")
        if meta_url and meta_url.get("content"):
            url = meta_url.get("content", "").lower()
            if any(
                search_term in url for search_term in ["search", "query", "q=", "find"]
            ):
                is_search_page = True

        # Check for common search result containers
        search_containers = soup.select(
            "div.g, div.result, div.search-result, div.serp-item, .searchResult"
        )
        if search_containers and len(search_containers) > 3:
            is_search_page = True

            # Extract search results
            search_results = []
            for result in search_containers[:10]:  # Limit to first 10 results
                # Try to extract title, URL, and snippet
                title_elem = result.find(["h3", "h2", ".title"])
                link_elem = result.find("a")
                snippet_elem = result.find(
                    ["span.snippet", "div.snippet", "p", ".description"]
                )

                result_info = {
                    "title": title_elem.get_text().strip() if title_elem else "",
                    "url": link_elem.get("href") if link_elem else "",
                    "snippet": snippet_elem.get_text().strip() if snippet_elem else "",
                }
                search_results.append(result_info)

            results["search_results"] = search_results
            results["page_type"] = "search"

        # If not a search page, determine other page types
        if not is_search_page:
            # Check if it's a form-heavy page
            if len(form_fields) > 5:
                results["page_type"] = "form"
            # Check if it's an article
            elif soup.find(["article", ".article", "main"]) and soup.find(["h1", "h2"]):
                results["page_type"] = "article"

                # Extract article content
                article_element = soup.find(["article", ".article", "main"])
                if article_element:
                    title = soup.find("h1")
                    # Get key sections (h2, h3 headers)
                    sections = []
                    for header in article_element.find_all(["h2", "h3"]):
                        sections.append(
                            {"header": header.get_text().strip(), "level": header.name}
                        )

                    results["key_elements"] = {
                        "title": title.get_text().strip() if title else "",
                        "sections": sections,
                    }

        logging.info(
            f"Extracted {len(form_fields)} form fields. Page type: {results['page_type']}"
        )
        if results["page_type"] == "search":
            logging.info(f"Found {len(results['search_results'])} search results")

    except ImportError:
        logging.error("BeautifulSoup is not installed. Unable to parse HTML.")
    except Exception as e:
        logging.error(f"Error extracting fields from HTML: {e}")

    return results


def fill_fields(html_content: str, fields: List[Dict]) -> str:
    """
    Fill in the form fields with the provided values.
    
    Args:
        html_content (str): The HTML content of the page
        fields (List[Dict]): List of form fields with their values
    
    Returns:
        str: HTML content with form fields filled in
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        for field in fields:
            input_element = soup.find(field["type"], {"name": field["name"]})
            if input_element:
                input_element["value"] = field["value"]
        return str(soup)
    except ImportError:
        logging.error("BeautifulSoup is not installed. Unable to fill form fields.")
    except Exception as e:
        logging.error(f"Error filling form fields: {e}")
    return html_content


def beautifulJsonPrint(json_data: Dict) -> None:
    """Print JSON data in a formatted, human-readable format.
    
    This utility function formats JSON data with proper indentation
    for improved readability during debugging and development.
    
    Args:
        json_data (Dict): The JSON data to format and print
        
    Returns:
        None: This function prints to stdout but does not return a value
    """
    print(json.dumps(json_data, indent=2))




if __name__ == "__main__":
    ## test extract_fields_from_page using soup
    
    # get wikipedia html content
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    html_content = requests.get(url).text

    # extract fields from page
    fields = extract_fields_from_page(html_content)
    fields["form_fields"][0]["value"] = "test"
    beatifulJsonPrint(fields)

    

    