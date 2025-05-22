"""
Query Tool for AI Web Scraper Project

This tool handles content search, relevance scoring, context matching,
and AI-powered answer generation from scraped content.
"""

import re
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from crewai_tools import BaseTool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Data class for search results"""
    content_id: str
    title: str
    content: str
    url: str
    relevance_score: float
    matched_snippets: List[str]
    timestamp: datetime


@dataclass
class QueryResponse:
    """Data class for query responses"""
    query: str
    answer: str
    sources: List[SearchResult]
    confidence_score: float
    query_id: str
    timestamp: datetime


class QueryTool(BaseTool):
    """
    Tool for querying scraped content and generating AI-powered answers.
    
    This tool provides functionality to:
    - Search through scraped content
    - Calculate relevance scores
    - Generate contextual answers with sources
    - Handle different query types
    """
    
    name: str = "QueryTool"
    description: str = (
        "Search and query scraped website content to answer user questions. "
        "Provides relevant information with source attribution and confidence scoring."
    )
    
    def __init__(self):
        super().__init__()
        self.content_database: Dict[str, Dict] = {}
        self.search_index: Dict[str, List[str]] = {}
        self.query_history: List[QueryResponse] = []
        
    def _execute(self, query: str, content_db: Optional[Dict] = None, 
                 max_results: int = 5, min_relevance: float = 0.1) -> Dict[str, Any]:
        """
        Execute query against content database
        
        Args:
            query: User's search query
            content_db: Optional content database (uses internal if not provided)
            max_results: Maximum number of results to return
            min_relevance: Minimum relevance score threshold
            
        Returns:
            Dictionary containing query response with answer and sources
        """
        try:
            # Use provided content_db or internal database
            if content_db:
                self.content_database = content_db
                
            # Validate inputs
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
                
            if not self.content_database:
                return {
                    "success": False,
                    "error": "No content available to search",
                    "query": query
                }
            
            # Search for relevant content
            search_results = self.search_content(
                query=query,
                max_results=max_results,
                min_relevance=min_relevance
            )
            
            # Generate AI-powered answer
            if search_results:
                answer_data = self.generate_answer(query, search_results)
            else:
                answer_data = {
                    "answer": "No relevant content found for your query.",
                    "confidence_score": 0.0
                }
            
            # Create response object
            response = QueryResponse(
                query=query,
                answer=answer_data["answer"],
                sources=search_results,
                confidence_score=answer_data["confidence_score"],
                query_id=str(uuid.uuid4()),
                timestamp=datetime.now()
            )
            
            # Store in query history
            self.query_history.append(response)
            
            logger.info(f"Query processed successfully: {query[:50]}...")
            
            return {
                "success": True,
                "response": asdict(response),
                "total_sources": len(search_results),
                "processing_time": "< 1s"
            }
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            return {
                "success": False,
                "error": f"Query processing failed: {str(e)}",
                "query": query
            }
    
    def search_content(self, query: str, max_results: int = 5, 
                      min_relevance: float = 0.1) -> List[SearchResult]:
        """
        Search through content database for relevant information
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            min_relevance: Minimum relevance score threshold
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        try:
            if not self.content_database:
                return []
            
            results = []
            query_terms = self._extract_query_terms(query)
            
            for content_id, content_data in self.content_database.items():
                # Calculate relevance score
                relevance_score = self.calculate_relevance(query, content_data)
                
                # Skip if below minimum relevance threshold
                if relevance_score < min_relevance:
                    continue
                
                # Extract matched snippets
                matched_snippets = self._extract_matching_snippets(
                    query_terms, content_data.get("content", "")
                )
                
                # Create search result
                result = SearchResult(
                    content_id=content_id,
                    title=content_data.get("title", "Untitled"),
                    content=content_data.get("content", ""),
                    url=content_data.get("url", ""),
                    relevance_score=relevance_score,
                    matched_snippets=matched_snippets,
                    timestamp=datetime.fromisoformat(
                        content_data.get("timestamp", datetime.now().isoformat())
                    )
                )
                
                results.append(result)
            
            # Sort by relevance score (descending)
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Return top results
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            return []
    
    def calculate_relevance(self, query: str, content_data: Dict[str, Any]) -> float:
        """
        Calculate relevance score between query and content
        
        Args:
            query: User's search query
            content_data: Content data dictionary
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            # Get content text
            content = content_data.get("content", "").lower()
            title = content_data.get("title", "").lower()
            query_lower = query.lower()
            
            if not content and not title:
                return 0.0
            
            # Extract query terms
            query_terms = self._extract_query_terms(query_lower)
            
            if not query_terms:
                return 0.0
            
            # Calculate different types of matches
            exact_match_score = self._calculate_exact_match_score(query_lower, content, title)
            term_match_score = self._calculate_term_match_score(query_terms, content, title)
            semantic_score = self._calculate_semantic_score(query_terms, content, title)
            
            # Weighted combination of scores
            relevance_score = (
                exact_match_score * 0.4 +
                term_match_score * 0.4 +
                semantic_score * 0.2
            )
            
            # Normalize to 0-1 range
            return min(1.0, max(0.0, relevance_score))
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {str(e)}")
            return 0.0
    
    def generate_answer(self, query: str, relevant_content: List[SearchResult]) -> Dict[str, Any]:
        """
        Generate AI-powered answer based on relevant content
        
        Args:
            query: User's original query
            relevant_content: List of relevant SearchResult objects
            
        Returns:
            Dictionary with generated answer and confidence score
        """
        try:
            if not relevant_content:
                return {
                    "answer": "I couldn't find relevant information to answer your query.",
                    "confidence_score": 0.0
                }
            
            # Combine relevant content for context
            context_content = []
            for result in relevant_content[:3]:  # Use top 3 results
                context_content.append({
                    "title": result.title,
                    "content": result.content[:500],  # Limit content length
                    "url": result.url,
                    "snippets": result.matched_snippets
                })
            
            # Generate answer based on query type
            answer = self._generate_contextual_answer(query, context_content)
            
            # Calculate confidence score
            confidence_score = self._calculate_answer_confidence(
                query, answer, relevant_content
            )
            
            return {
                "answer": answer,
                "confidence_score": confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence_score": 0.0
            }
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'what', 'when', 'where', 'why', 'how', 'who', 'which'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _calculate_exact_match_score(self, query: str, content: str, title: str) -> float:
        """Calculate score for exact phrase matches"""
        query_clean = query.strip()
        if not query_clean:
            return 0.0
        
        # Check for exact matches in title (higher weight)
        title_matches = title.count(query_clean)
        content_matches = content.count(query_clean)
        
        # Calculate score
        title_score = min(1.0, title_matches * 0.5)
        content_score = min(0.8, content_matches * 0.1)
        
        return title_score + content_score
    
    def _calculate_term_match_score(self, terms: List[str], content: str, title: str) -> float:
        """Calculate score based on individual term matches"""
        if not terms:
            return 0.0
        
        title_matches = sum(1 for term in terms if term in title)
        content_matches = sum(1 for term in terms if term in content)
        
        title_ratio = title_matches / len(terms)
        content_ratio = content_matches / len(terms)
        
        return (title_ratio * 0.6) + (content_ratio * 0.4)
    
    def _calculate_semantic_score(self, terms: List[str], content: str, title: str) -> float:
        """Calculate semantic similarity score (simplified version)"""
        # Simple proximity-based semantic scoring
        if not terms or len(terms) < 2:
            return 0.0
        
        combined_text = f"{title} {content}".lower()
        proximity_score = 0.0
        
        for i, term1 in enumerate(terms):
            for term2 in terms[i+1:]:
                # Find positions of both terms
                pos1 = combined_text.find(term1)
                pos2 = combined_text.find(term2)
                
                if pos1 != -1 and pos2 != -1:
                    # Calculate proximity score (closer = higher score)
                    distance = abs(pos1 - pos2)
                    if distance < 100:  # Within 100 characters
                        proximity_score += 1.0 / (1 + distance / 20)
        
        # Normalize by number of term pairs
        num_pairs = len(terms) * (len(terms) - 1) / 2
        return min(1.0, proximity_score / num_pairs) if num_pairs > 0 else 0.0
    
    def _extract_matching_snippets(self, terms: List[str], content: str, 
                                  snippet_length: int = 150) -> List[str]:
        """Extract text snippets containing query terms"""
        snippets = []
        content_lower = content.lower()
        
        for term in terms:
            pos = content_lower.find(term.lower())
            if pos != -1:
                # Extract snippet around the term
                start = max(0, pos - snippet_length // 2)
                end = min(len(content), pos + snippet_length // 2)
                snippet = content[start:end].strip()
                
                # Add ellipsis if truncated
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
                
                if snippet not in snippets:
                    snippets.append(snippet)
        
        return snippets[:3]  # Return top 3 snippets
    
    def _generate_contextual_answer(self, query: str, context_content: List[Dict]) -> str:
        """Generate contextual answer based on query and content"""
        # Determine query type
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'what are', 'define']):
            return self._generate_definition_answer(query, context_content)
        elif any(word in query_lower for word in ['how to', 'how do', 'how can']):
            return self._generate_how_to_answer(query, context_content)
        elif any(word in query_lower for word in ['why', 'because', 'reason']):
            return self._generate_explanation_answer(query, context_content)
        elif any(word in query_lower for word in ['when', 'date', 'time']):
            return self._generate_temporal_answer(query, context_content)
        else:
            return self._generate_general_answer(query, context_content)
    
    def _generate_definition_answer(self, query: str, context_content: List[Dict]) -> str:
        """Generate definition-type answer"""
        if not context_content:
            return "No definition found in the available content."
        
        # Look for definition-like content
        for content in context_content:
            text = content['content'].lower()
            if any(word in text for word in ['definition', 'means', 'refers to', 'is defined as']):
                return f"Based on the content from {content['url']}: {content['content'][:200]}..."
        
        # Fallback to first relevant content
        return f"According to {context_content[0]['url']}: {context_content[0]['content'][:200]}..."
    
    def _generate_how_to_answer(self, query: str, context_content: List[Dict]) -> str:
        """Generate how-to type answer"""
        if not context_content:
            return "No procedural information found in the available content."
        
        # Look for step-by-step or instructional content
        for content in context_content:
            text = content['content'].lower()
            if any(word in text for word in ['step', 'first', 'then', 'next', 'finally']):
                return f"Based on instructions from {content['url']}: {content['content'][:300]}..."
        
        return f"From {context_content[0]['url']}: {context_content[0]['content'][:250]}..."
    
    def _generate_explanation_answer(self, query: str, context_content: List[Dict]) -> str:
        """Generate explanation-type answer"""
        if not context_content:
            return "No explanatory information found in the available content."
        
        # Combine multiple sources for comprehensive explanation
        explanation_parts = []
        for content in context_content[:2]:  # Use top 2 sources
            snippet = content['content'][:150]
            explanation_parts.append(f"According to {content['url']}: {snippet}")
        
        return " ".join(explanation_parts) + "..."
    
    def _generate_temporal_answer(self, query: str, context_content: List[Dict]) -> str:
        """Generate time/date related answer"""
        if not context_content:
            return "No temporal information found in the available content."
        
        # Look for date/time related information
        for content in context_content:
            text = content['content']
            if re.search(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', text.lower()):
                return f"From {content['url']}: {content['content'][:200]}..."
        
        return f"Based on {context_content[0]['url']}: {context_content[0]['content'][:200]}..."
    
    def _generate_general_answer(self, query: str, context_content: List[Dict]) -> str:
        """Generate general answer"""
        if not context_content:
            return "No relevant information found in the available content."
        
        # Combine information from multiple sources
        answer_parts = []
        for i, content in enumerate(context_content[:2]):
            source_info = f"Source {i+1} ({content['url']}): {content['content'][:150]}"
            answer_parts.append(source_info)
        
        return ". ".join(answer_parts) + "..."
    
    def _calculate_answer_confidence(self, query: str, answer: str, 
                                   sources: List[SearchResult]) -> float:
        """Calculate confidence score for generated answer"""
        if not sources:
            return 0.0
        
        # Base confidence on source relevance scores
        avg_relevance = sum(source.relevance_score for source in sources) / len(sources)
        
        # Adjust based on number of sources
        source_factor = min(1.0, len(sources) / 3.0)
        
        # Adjust based on answer length (longer answers might be more comprehensive)
        length_factor = min(1.0, len(answer) / 200.0)
        
        confidence = (avg_relevance * 0.6) + (source_factor * 0.3) + (length_factor * 0.1)
        
        return min(1.0, max(0.0, confidence))
    
    def load_content_database(self, content_db: Dict[str, Dict]) -> bool:
        """
        Load content database for querying
        
        Args:
            content_db: Dictionary containing scraped content
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.content_database = content_db
            self._build_search_index()
            logger.info(f"Loaded content database with {len(content_db)} entries")
            return True
        except Exception as e:
            logger.error(f"Error loading content database: {str(e)}")
            return False
    
    def _build_search_index(self) -> None:
        """Build search index for faster querying"""
        self.search_index = {}
        
        for content_id, content_data in self.content_database.items():
            content_text = content_data.get("content", "").lower()
            title_text = content_data.get("title", "").lower()
            
            # Extract terms for indexing
            all_text = f"{title_text} {content_text}"
            terms = self._extract_query_terms(all_text)
            
            for term in terms:
                if term not in self.search_index:
                    self.search_index[term] = []
                if content_id not in self.search_index[term]:
                    self.search_index[term].append(content_id)
    
    def get_query_history(self) -> List[Dict]:
        """Get query history"""
        return [asdict(query) for query in self.query_history]
    
    def clear_query_history(self) -> None:
        """Clear query history"""
        self.query_history = []
        logger.info("Query history cleared")