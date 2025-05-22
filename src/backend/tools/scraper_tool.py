import requests
import time
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from crewai_tools import BaseTool
from pydantic import BaseModel, Field


class ScrapingOptions(BaseModel):
    """Configuration options for web scraping."""
    use_selenium: bool = Field(default=False, description="Use Selenium for JavaScript-heavy sites")
    wait_time: int = Field(default=10, description="Maximum wait time for page load (seconds)")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    delay_between_requests: float = Field(default=1.0, description="Delay between requests (seconds)")
    extract_links: bool = Field(default=True, description="Extract all links from the page")
    extract_images: bool = Field(default=True, description="Extract image URLs")
    extract_tables: bool = Field(default=True, description="Extract table data")
    headers: Dict[str, str] = Field(
        default_factory=lambda: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        },
        description="HTTP headers to use for requests"
    )


class ScrapingResult(BaseModel):
    """Result object containing scraped data."""
    url: str
    title: str
    content: str
    links: List[Dict[str, str]]
    images: List[Dict[str, str]]
    tables: List[List[List[str]]]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class ScraperTool(BaseTool):
    """
    AI-powered web scraping tool that can handle both static and dynamic content.
    Supports multiple extraction methods and respectful scraping practices.
    """
    
    name: str = "Web Scraper Tool"
    description: str = """
    Advanced web scraping tool capable of extracting content from websites.
    Features:
    - Static content scraping with requests + BeautifulSoup
    - Dynamic content scraping with Selenium for JavaScript sites
    - Intelligent content extraction and cleaning
    - Rate limiting and respectful scraping
    - Comprehensive error handling
    
    Use this tool to scrape websites and extract structured information.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the scraper."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _run(self, url: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Main execution method for the scraper tool.
        
        Args:
            url: The URL to scrape
            options: Optional scraping configuration
            
        Returns:
            JSON string containing the scraping results
        """
        try:
            # Parse options
            scraping_options = ScrapingOptions(**(options or {}))
            
            # Perform scraping
            result = self.scrape_url(url, scraping_options)
            
            return result.model_dump_json(indent=2)
            
        except Exception as e:
            self.logger.error(f"Error in scraper tool: {str(e)}")
            error_result = ScrapingResult(
                url=url,
                title="",
                content="",
                links=[],
                images=[],
                tables=[],
                metadata={},
                success=False,
                error_message=str(e)
            )
            return error_result.model_dump_json(indent=2)
    
    def scrape_url(self, url: str, options: ScrapingOptions) -> ScrapingResult:
        """
        Main scraping method that chooses between static and dynamic scraping.
        
        Args:
            url: The URL to scrape
            options: Scraping configuration options
            
        Returns:
            ScrapingResult object containing all extracted data
        """
        self.logger.info(f"Starting scrape of: {url}")
        
        # Validate URL
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL provided: {url}")
        
        # Choose scraping method
        if options.use_selenium:
            return self._scrape_with_selenium(url, options)
        else:
            return self._scrape_with_requests(url, options)
    
    def _scrape_with_requests(self, url: str, options: ScrapingOptions) -> ScrapingResult:
        """
        Scrape using requests + BeautifulSoup for static content.
        
        Args:
            url: The URL to scrape
            options: Scraping configuration
            
        Returns:
            ScrapingResult with extracted data
        """
        for attempt in range(options.max_retries):
            try:
                self.logger.info(f"Attempt {attempt + 1} - Scraping with requests: {url}")
                
                # Add delay for respectful scraping
                if attempt > 0:
                    time.sleep(options.delay_between_requests * attempt)
                
                # Make request
                response = self.session.get(
                    url,
                    headers=options.headers,
                    timeout=options.wait_time
                )
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract all content
                return self._extract_content_from_soup(soup, url, options)
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt == options.max_retries - 1:
                    raise e
                
            except Exception as e:
                self.logger.error(f"Unexpected error during scraping: {str(e)}")
                raise e
    
    def _scrape_with_selenium(self, url: str, options: ScrapingOptions) -> ScrapingResult:
        """
        Scrape using Selenium for JavaScript-heavy sites.
        
        Args:
            url: The URL to scrape
            options: Scraping configuration
            
        Returns:
            ScrapingResult with extracted data
        """
        driver = None
        try:
            self.logger.info(f"Scraping with Selenium: {url}")
            
            # Configure Chrome options
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'--user-agent={options.headers.get("User-Agent", "")}')
            
            # Create driver
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(options.wait_time)
            
            # Load page
            driver.get(url)
            
            # Wait for page to fully load
            WebDriverWait(driver, options.wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            time.sleep(2)
            
            # Get page source and parse
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            return self._extract_content_from_soup(soup, url, options)
            
        except TimeoutException:
            self.logger.error(f"Timeout loading page: {url}")
            raise TimeoutException(f"Page load timeout for {url}")
            
        except WebDriverException as e:
            self.logger.error(f"WebDriver error: {str(e)}")
            raise WebDriverException(f"WebDriver error for {url}: {str(e)}")
            
        finally:
            if driver:
                driver.quit()
    
    def _extract_content_from_soup(self, soup: BeautifulSoup, url: str, options: ScrapingOptions) -> ScrapingResult:
        """
        Extract all content from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup parsed HTML
            url: Original URL
            options: Scraping configuration
            
        Returns:
            ScrapingResult with all extracted data
        """
        try:
            # Extract basic content
            title = self._extract_title(soup)
            content = self.extract_text_content(soup)
            
            # Extract structured data based on options
            links = self.extract_links(soup, url) if options.extract_links else []
            images = self._extract_images(soup, url) if options.extract_images else []
            tables = self._extract_tables(soup) if options.extract_tables else []
            
            # Extract metadata
            metadata = self._extract_metadata(soup)
            
            return ScrapingResult(
                url=url,
                title=title,
                content=content,
                links=links,
                images=images,
                tables=tables,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting content: {str(e)}")
            raise e
    
    def extract_text_content(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text content from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Cleaned text content
        """
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """
        Extract all links from the page.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
            
        Returns:
            List of link dictionaries with text and href
        """
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text(strip=True)
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Only include HTTP/HTTPS links
            if absolute_url.startswith(('http://', 'https://')):
                links.append({
                    'text': text,
                    'href': absolute_url
                })
        
        return links
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)
        
        return "No title found"
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract image information."""
        images = []
        for img in soup.find_all('img'):
            src = img.get('src')
            alt = img.get('alt', '')
            
            if src:
                absolute_url = urljoin(base_url, src)
                images.append({
                    'src': absolute_url,
                    'alt': alt
                })
        
        return images
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """Extract table data."""
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    row_data.append(cell.get_text(strip=True))
                if row_data:
                    table_data.append(row_data)
            if table_data:
                tables.append(table_data)
        
        return tables
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract page metadata."""
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        # Count various elements
        metadata.update({
            'paragraph_count': len(soup.find_all('p')),
            'heading_count': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
            'link_count': len(soup.find_all('a')),
            'image_count': len(soup.find_all('img')),
            'table_count': len(soup.find_all('table'))
        })
        
        return metadata
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def handle_javascript_content(self, url: str) -> str:
        """
        Handle JavaScript-heavy content by automatically using Selenium.
        
        Args:
            url: URL to scrape
            
        Returns:
            JSON string with scraping results
        """
        options = ScrapingOptions(use_selenium=True)
        result = self.scrape_url(url, options)
        return result.model_dump_json(indent=2)