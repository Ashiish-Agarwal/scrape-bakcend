"""
Content Processor Tool for AI Web Scraper

This tool handles content processing, cleaning, structuring, and analysis
of scraped web content. It prepares raw scraped data for AI querying.
"""

import re
import html
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
from datetime import datetime
import hashlib

from crewai_tools import BaseTool
from pydantic import BaseModel, Field


class ContentMetadata(BaseModel):
    """Metadata structure for processed content."""
    title: str = ""
    description: str = ""
    author: str = ""
    publish_date: str = ""
    keywords: List[str] = []
    content_type: str = ""
    word_count: int = 0
    reading_time: int = 0  # in minutes
    language: str = "en"
    headings: List[str] = []
    links_count: int = 0
    images_count: int = 0


class ProcessedContent(BaseModel):
    """Structure for fully processed content."""
    url: str
    title: str
    clean_text: str
    summary: str
    key_points: List[str]
    categories: List[str]
    metadata: ContentMetadata
    sections: Dict[str, str]
    processed_at: str
    content_hash: str


class ProcessorTool(BaseTool):
    """
    Tool for processing and analyzing scraped web content.
    
    Handles content cleaning, structuring, categorization, and summarization
    to prepare data for AI-powered querying.
    """
    
    name: str = "Content Processor"
    description: str = (
        "Processes and analyzes scraped web content. "
        "Cleans raw HTML content, extracts key information, "
        "categorizes content, and generates summaries."
    )
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Common words to exclude from keyword extraction
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
            'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
            'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'should',
            'could', 'can', 'may', 'might', 'must', 'shall'
        }
        
        # Content categories for classification
        self.content_categories = {
            'news': ['news', 'article', 'report', 'breaking', 'update', 'story'],
            'tutorial': ['tutorial', 'guide', 'how-to', 'step', 'instructions', 'learn'],
            'blog': ['blog', 'post', 'opinion', 'thoughts', 'personal', 'diary'],
            'documentation': ['docs', 'documentation', 'manual', 'reference', 'api'],
            'product': ['product', 'review', 'specification', 'feature', 'price'],
            'research': ['research', 'study', 'analysis', 'data', 'survey', 'findings'],
            'entertainment': ['entertainment', 'movie', 'music', 'game', 'celebrity'],
            'technology': ['tech', 'software', 'hardware', 'programming', 'code'],
            'business': ['business', 'finance', 'market', 'economy', 'company'],
            'health': ['health', 'medical', 'fitness', 'wellness', 'doctor'],
            'education': ['education', 'school', 'university', 'course', 'learning']
        }

    def _run(self, raw_content: str, url: str = "", scrape_metadata: Dict[str, Any] = None) -> str:
        """
        Main execution method for content processing.
        
        Args:
            raw_content: Raw scraped content to process
            url: Original URL of the content
            scrape_metadata: Additional metadata from scraping
            
        Returns:
            JSON string of processed content
        """
        try:
            self.logger.info(f"Starting content processing for URL: {url}")
            
            # Parse input if it's JSON string
            if isinstance(raw_content, str) and raw_content.startswith('{'):
                try:
                    content_data = json.loads(raw_content)
                    text_content = content_data.get('content', raw_content)
                    url = content_data.get('url', url)
                    scrape_metadata = content_data.get('metadata', scrape_metadata or {})
                except json.JSONDecodeError:
                    text_content = raw_content
            else:
                text_content = raw_content
            
            # Step 1: Clean the content
            cleaned_content = self.clean_content(text_content)
            
            # Step 2: Extract metadata
            metadata = self.extract_metadata(cleaned_content, scrape_metadata or {})
            
            # Step 3: Structure content into sections
            sections = self.structure_content(cleaned_content)
            
            # Step 4: Extract key information
            key_info = self.extract_key_information(cleaned_content)
            
            # Step 5: Categorize content
            categories = self.categorize_content(cleaned_content, metadata)
            
            # Step 6: Generate summary
            summary = self.generate_summary(cleaned_content)
            
            # Step 7: Create processed content structure
            processed = ProcessedContent(
                url=url,
                title=metadata.title or self._extract_title_from_content(cleaned_content),
                clean_text=cleaned_content,
                summary=summary,
                key_points=key_info.get('key_points', []),
                categories=categories,
                metadata=metadata,
                sections=sections,
                processed_at=datetime.now().isoformat(),
                content_hash=self._generate_content_hash(cleaned_content)
            )
            
            result = processed.model_dump()
            self.logger.info(f"Content processing completed. Word count: {metadata.word_count}")
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            return json.dumps({
                "error": f"Content processing failed: {str(e)}",
                "url": url,
                "processed_at": datetime.now().isoformat()
            })

    def clean_content(self, raw_content: str) -> str:
        """
        Clean and normalize raw content text.
        
        Args:
            raw_content: Raw text content to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not raw_content:
            return ""
        
        # Decode HTML entities
        content = html.unescape(raw_content)
        
        # Remove HTML tags if any remain
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Remove script and style content
        content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)  # Multiple spaces to single space
        content = re.sub(r'\n\s*\n', '\n\n', content)  # Clean up line breaks
        
        # Remove excessive punctuation
        content = re.sub(r'[.]{3,}', '...', content)
        content = re.sub(r'[!]{2,}', '!', content)
        content = re.sub(r'[?]{2,}', '?', content)
        
        # Remove special characters but keep essential punctuation
        content = re.sub(r'[^\w\s\.\,\!\?\;\:\(\)\-\"\'\n]', ' ', content)
        
        # Remove URLs
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        
        # Clean up email addresses
        content = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', content)
        
        # Remove phone numbers
        content = re.sub(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', '[PHONE]', content)
        
        # Final cleanup
        content = content.strip()
        content = re.sub(r'\s+', ' ', content)
        
        return content

    def extract_metadata(self, content: str, scrape_metadata: Dict[str, Any]) -> ContentMetadata:
        """
        Extract metadata from content and scraping data.
        
        Args:
            content: Cleaned content text
            scrape_metadata: Metadata from scraping process
            
        Returns:
            ContentMetadata object
        """
        metadata = ContentMetadata()
        
        # Extract from scrape metadata
        metadata.title = scrape_metadata.get('title', '')
        metadata.description = scrape_metadata.get('description', '')
        metadata.author = scrape_metadata.get('author', '')
        metadata.publish_date = scrape_metadata.get('publish_date', '')
        metadata.content_type = scrape_metadata.get('content_type', 'webpage')
        
        # Calculate word count and reading time
        words = content.split()
        metadata.word_count = len(words)
        metadata.reading_time = max(1, metadata.word_count // 200)  # Assume 200 WPM reading speed
        
        # Extract headings
        metadata.headings = self._extract_headings(content)
        
        # Extract keywords
        metadata.keywords = self._extract_keywords(content)
        
        # Count links and images from scrape metadata
        metadata.links_count = scrape_metadata.get('links_count', 0)
        metadata.images_count = scrape_metadata.get('images_count', 0)
        
        # Detect language (simple heuristic)
        metadata.language = self._detect_language(content)
        
        return metadata

    def structure_content(self, content: str) -> Dict[str, str]:
        """
        Structure content into logical sections.
        
        Args:
            content: Cleaned content text
            
        Returns:
            Dictionary of content sections
        """
        sections = {}
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return {"full_content": content}
        
        # Identify introduction (first paragraph)
        if paragraphs:
            sections['introduction'] = paragraphs[0]
        
        # Identify conclusion (last paragraph if it contains concluding words)
        concluding_words = ['conclusion', 'summary', 'finally', 'in conclusion', 'to summarize']
        if len(paragraphs) > 1:
            last_para = paragraphs[-1].lower()
            if any(word in last_para for word in concluding_words):
                sections['conclusion'] = paragraphs[-1]
                main_content = paragraphs[1:-1]
            else:
                main_content = paragraphs[1:]
        else:
            main_content = []
        
        # Group main content
        if main_content:
            sections['main_content'] = '\n\n'.join(main_content)
        
        # Full content for reference
        sections['full_content'] = content
        
        return sections

    def extract_key_information(self, content: str) -> Dict[str, Any]:
        """
        Extract key information points from content.
        
        Args:
            content: Cleaned content text
            
        Returns:
            Dictionary containing key information
        """
        key_info = {
            'key_points': [],
            'important_phrases': [],
            'numbers_and_stats': [],
            'dates': [],
            'names': []
        }
        
        sentences = self._split_into_sentences(content)
        
        # Extract key points (sentences with important indicators)
        importance_indicators = [
            'important', 'key', 'main', 'significant', 'crucial', 'essential',
            'primary', 'major', 'critical', 'fundamental', 'note that',
            'remember', 'keep in mind', 'it should be noted'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in importance_indicators):
                if len(sentence.split()) > 5:  # Avoid very short sentences
                    key_info['key_points'].append(sentence.strip())
        
        # Limit key points to most relevant
        key_info['key_points'] = key_info['key_points'][:10]
        
        # Extract numbers and statistics
        number_pattern = r'\b\d+(?:\.\d+)?(?:%|\s*percent|\s*million|\s*billion|\s*thousand)?\b'
        key_info['numbers_and_stats'] = list(set(re.findall(number_pattern, content, re.IGNORECASE)))
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD or YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',  # Month DD, YYYY
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b'  # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, content, re.IGNORECASE)
            key_info['dates'].extend(dates)
        
        key_info['dates'] = list(set(key_info['dates']))[:10]
        
        # Extract names (capitalized words that might be names)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_names = re.findall(name_pattern, content)
        # Filter out common words that are capitalized
        common_caps = {'The', 'This', 'That', 'These', 'Those', 'And', 'But', 'Or', 'So', 'For'}
        key_info['names'] = [name for name in potential_names if name not in common_caps][:15]
        
        return key_info

    def categorize_content(self, content: str, metadata: ContentMetadata) -> List[str]:
        """
        Categorize content based on keywords and context.
        
        Args:
            content: Cleaned content text
            metadata: Content metadata
            
        Returns:
            List of content categories
        """
        categories = []
        content_lower = content.lower()
        title_lower = metadata.title.lower()
        
        # Check each category
        for category, keywords in self.content_categories.items():
            score = 0
            
            # Check in content
            for keyword in keywords:
                content_matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', content_lower))
                title_matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', title_lower))
                
                score += content_matches + (title_matches * 3)  # Title matches weighted more
            
            # Normalize score by content length
            if metadata.word_count > 0:
                normalized_score = score / (metadata.word_count / 100)
                if normalized_score > 0.5:  # Threshold for category inclusion
                    categories.append(category)
        
        # If no categories found, try to infer from content structure
        if not categories:
            if any(word in content_lower for word in ['how', 'step', 'tutorial', 'guide']):
                categories.append('tutorial')
            elif any(word in content_lower for word in ['news', 'reported', 'according to']):
                categories.append('news')
            elif any(word in content_lower for word in ['opinion', 'think', 'believe', 'feel']):
                categories.append('blog')
            else:
                categories.append('general')
        
        return categories[:3]  # Limit to top 3 categories

    def generate_summary(self, content: str, max_sentences: int = 3) -> str:
        """
        Generate a summary of the content.
        
        Args:
            content: Content to summarize
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Content summary
        """
        if not content:
            return ""
        
        sentences = self._split_into_sentences(content)
        
        if len(sentences) <= max_sentences:
            return content
        
        # Score sentences based on various factors
        sentence_scores = {}
        word_freq = self._calculate_word_frequency(content)
        
        for i, sentence in enumerate(sentences):
            score = 0
            words = sentence.lower().split()
            
            # Score based on word frequency
            for word in words:
                if word in word_freq and word not in self.stop_words:
                    score += word_freq[word]
            
            # Bonus for position (first and last sentences often important)
            if i == 0:
                score *= 1.5
            elif i == len(sentences) - 1:
                score *= 1.2
            
            # Bonus for sentence length (avoid very short sentences)
            if len(words) > 10:
                score *= 1.1
            
            # Penalty for very long sentences
            if len(words) > 30:
                score *= 0.8
            
            sentence_scores[sentence] = score
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if any(sentence == s[0] for s in top_sentences):
                summary_sentences.append(sentence)
                if len(summary_sentences) >= max_sentences:
                    break
        
        return ' '.join(summary_sentences)

    def _extract_title_from_content(self, content: str) -> str:
        """Extract title from content if not provided."""
        sentences = self._split_into_sentences(content)
        if sentences:
            # Use first sentence as title if it's not too long
            first_sentence = sentences[0]
            if len(first_sentence.split()) <= 15:
                return first_sentence
        return "Untitled Content"

    def _extract_headings(self, content: str) -> List[str]:
        """Extract potential headings from content."""
        lines = content.split('\n')
        headings = []
        
        for line in lines:
            line = line.strip()
            # Potential heading: short line, starts with capital, no ending punctuation
            if (line and 
                len(line.split()) <= 10 and 
                line[0].isupper() and 
                not line.endswith(('.', '!', '?'))):
                headings.append(line)
        
        return headings[:10]  # Limit to 10 headings

    def _extract_keywords(self, content: str, max_keywords: int = 15) -> List[str]:
        """Extract keywords from content."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        word_freq = {}
        
        for word in words:
            if word not in self.stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]

    def _detect_language(self, content: str) -> str:
        """Simple language detection (placeholder for more sophisticated detection)."""
        # This is a very basic implementation
        # In a production system, you'd use a proper language detection library
        common_english_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']
        
        content_lower = content.lower()
        english_word_count = sum(1 for word in common_english_words if word in content_lower)
        
        # Simple threshold-based detection
        total_common_words = len(common_english_words)
        if english_word_count / total_common_words > 0.3:
            return "en"
        else:
            return "unknown"

    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _calculate_word_frequency(self, content: str) -> Dict[str, int]:
        """Calculate word frequency in content."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        word_freq = {}
        
        for word in words:
            if word not in self.stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return word_freq

    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for content deduplication."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()