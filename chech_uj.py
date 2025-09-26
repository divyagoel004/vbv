import streamlit as st
import json
from autogen import AssistantAgent, config_list_from_json
from base64 import b64decode
import requests
import torch
from io import BytesIO
import numpy as np
import os
import re

from agent import search_and_embed
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter
from PIL import Image
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from time import sleep
from collections import defaultdict
import ast
from cleanup_images import start_cleanup_daemon

start_cleanup_daemon()
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# New imports for DuckDuckGo search and vector database
import hashlib
from urllib.parse import urljoin, urlparse
import trafilatura
import asyncio
import aiohttp
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any
import uuid
from base import serper_search, query_vector_db, clear_vector_db, get_filtered_sources, rebuild_vector_db_with_filtered_sources

# Define the structure to parse
class CodeSnippet(BaseModel):
    code: str = Field(..., description="Concise Python code snippet")
parser = PydanticOutputParser(pydantic_object=CodeSnippet)

# API Keys (only Gemini needed now)
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Streamlit page config for wide layout
st.set_page_config(layout="wide", page_title="AI Presentation Studio", page_icon="🎯")
from langfuse import Langfuse

# Initialize Langfuse client
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com",
)

# Enhanced CSS for subtopic interface
st.markdown("""
<style>
/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container styling */
.main-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 20px;
}

/* Subtopic container */
.subtopic-container {
    background: white;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    margin: 15px 0;
    padding: 20px;
    border-left: 4px solid #667eea;
}

.subtopic-title {
    font-size: 18px;
    font-weight: 600;
    color: #333;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 2px solid #f0f0f0;
}

.content-type-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
    margin: 15px 0;
}

.content-type-card {
    background: #f8f9ff;
    border: 2px solid #e0e6ff;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}

.content-type-card:hover {
    border-color: #667eea;
    background: #e8ecff;
    transform: translateY(-2px);
}

.content-type-card.selected {
    border-color: #667eea;
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
}

.component-tag {
    display: inline-block;
    background: #667eea;
    color: white;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 11px;
    margin: 2px;
}

/* Step indicators */
.step-indicator {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    padding: 15px 20px;
    border-radius: 10px;
    margin: 10px 0;
    font-weight: 600;
    text-align: center;
}

.step-indicator.step-completed {
    background: linear-gradient(45deg, #11998e, #38ef7d);
}

/* Editable content areas */
.editable-content {
    border: 2px dashed #ddd;
    border-radius: 8px;
    padding: 10px;
    margin: 8px 0;
    min-height: 40px;
    cursor: pointer;
}

.editable-content:hover {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.05);
}

/* Slide container */
.slide-container {
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    margin: 20px 0;
    overflow: hidden;
    transition: all 0.3s ease;
    position: relative;
}

.slide-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 30px 60px rgba(0,0,0,0.15);
}

.slide-header {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    font-size: 20px !important;   
    padding: 20px 25px !important;
    font-weight: 700;
    position: relative;
}

.slide-number {
    position: absolute;
    top: 15px;
    right: 25px;
    background: rgba(255,255,255,0.2);
    padding: 5px 12px;
    border-radius: 15px;
    font-size: 14px;
}

.slide-content {
    padding: 30px;
    min-height: 400px;
    font-size: 16px;
    line-height: 1.6;
}

.progress-bar {
    height: 4px;
    background: linear-gradient(45deg, #667eea, #764ba2);
    border-radius: 2px;
    margin: 20px 0;
    transition: width 0.3s ease;
}

.fade-in {
    animation: fadeIn 0.6s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

</style>
""", unsafe_allow_html=True)


research_sources_css = """
<style>
.research-sources-container {
    background: #f8f9ff;
    border: 2px solid #e0e6ff;
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
}

.source-card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    color: #333; /* Ensure dark text color */
}

.source-card:hover {
    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}

.source-card.excluded {
    opacity: 0.5;
    border-color: #ff6b6b;
    background: #fff5f5;
    color: #666; /* Darker text for excluded items */
}

.source-title {
    font-size: 16px;
    font-weight: 600;
    color: #333 !important; /* Force dark text */
    margin-bottom: 8px;
}

.source-url {
    font-size: 12px;
    color: #666 !important; /* Force visible text */
    margin-bottom: 10px;
    word-break: break-all;
}

.source-type-badge {
    display: inline-block;
    background: #667eea;
    color: white !important; /* White text on colored background */
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    margin-right: 5px;
}

.source-content-preview {
    background: #f8f9fa;
    border-left: 3px solid #667eea;
    padding: 10px;
    margin: 10px 0;
    font-size: 14px;
    max-height: 150px;
    overflow-y: auto;
    color: #333 !important; /* Force dark text */
}

.source-actions {
    text-align: right;
    margin-top: 10px;
}

/* Global text color fix for Streamlit elements */
.stMarkdown, .stText {
    color: #333 !important;
}

/* Fix for any other potential text visibility issues */
* {
    color: inherit;
}

/* Specifically target white backgrounds to ensure dark text */
[style*="background: white"], 
[style*="background-color: white"],
.white-bg {
    color: #333 !important;
}
</style>"""

# Initialize session state with new subtopic variables
if 'slides' not in st.session_state:
    st.session_state.slides = []
if 'current_slide' not in st.session_state:
    st.session_state.current_slide = 0
if 'presentation_mode' not in st.session_state:
    st.session_state.presentation_mode = False
if 'editing_section' not in st.session_state:
    st.session_state.editing_section = None
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = {}
if 'comments' not in st.session_state:
    st.session_state.comments = {}
if 'generation_step' not in st.session_state:
    st.session_state.generation_step = 'topic_subtopic_input'  
if 'topic' not in st.session_state:
    st.session_state.topic = ""
if 'subtopics' not in st.session_state:
    st.session_state.subtopics = []
if 'subtopic_content_structure' not in st.session_state:
    st.session_state.subtopic_content_structure = {}
if 'subtopic_component_structure' not in st.session_state:
    st.session_state.subtopic_component_structure = {}
if 'depth_level' not in st.session_state:
    st.session_state.depth_level = "intermediate"
if 'research_sources' not in st.session_state:
    st.session_state.research_sources = []
if 'selected_sources' not in st.session_state:
    st.session_state.selected_sources = set()
# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Load configuration files
try:
    config_list = config_list_from_json(env_or_file="config.txt")
    llm_config = {
        "seed": 44,
        "config_list": config_list,
        "temperature": 0
    }

    with open("file.json", "r") as f:
        content_types_dict = json.load(f)

    with open("component.txt", "r") as f:
        available_components = [line.strip() for line in f]
except FileNotFoundError as e:
    st.error(f"Configuration file not found: {e}")
    st.stop()

def render_topic_subtopic_input_step():
    """Render the enhanced topic and subtopic input step"""
    st.markdown("""
    <div class="step-indicator">
        📝 Step 1: Define Your Topic and Subtopics
    </div>
    """, unsafe_allow_html=True)
    
    # Main topic input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic = st.text_input(
            "Enter your main presentation topic:",
            value=st.session_state.topic,
            placeholder="e.g., Generative AI, Machine Learning, Blockchain Technology",
            key="main_topic_input"
        )
    
    with col2:
        depth_level = st.selectbox(
            "Audience level:",
            ["beginner", "intermediate", "advanced", "expert"],
            index=["beginner", "intermediate", "advanced", "expert"].index(st.session_state.depth_level),
            key="depth_input"
        )
    
    # Subtopic input section
    st.markdown("### 📋 Define Your Subtopics")
    st.markdown("Add the key subtopics you want to cover in your presentation:")
    
    # Display current subtopics
    if st.session_state.subtopics:
        st.markdown("**Current Subtopics:**")
        for i, subtopic in enumerate(st.session_state.subtopics):
            col_subtopic, col_remove = st.columns([4, 1])
            with col_subtopic:
                st.markdown(f"**{i+1}.** {subtopic}")
            with col_remove:
                if st.button("❌", key=f"remove_subtopic_{i}", help="Remove this subtopic"):
                    st.session_state.subtopics.pop(i)
                    st.rerun()
    
    # Add new subtopic
    col_add, col_button = st.columns([3, 1])
    with col_add:
        new_subtopic = st.text_input(
            "Add a subtopic:",
            placeholder="e.g., AI in Healthcare, Market Trends, Historical Overview",
            key="new_subtopic_input"
        )
    
    with col_button:
        if st.button("➕ Add Subtopic", disabled=not new_subtopic.strip()):
            if new_subtopic.strip() and new_subtopic.strip() not in st.session_state.subtopics:
                st.session_state.subtopics.append(new_subtopic.strip())
                st.rerun()
    
    # Example subtopics suggestion
    if topic.strip() and not st.session_state.subtopics:
        st.markdown("### 💡 AI Suggested Subtopics")
        if st.button("🤖 Generate Subtopic Suggestions"):
            suggested_subtopics = generate_subtopic_suggestions(topic.strip())
            if suggested_subtopics:
                st.session_state.subtopics = suggested_subtopics
                st.rerun()
    
    # Navigation button
    if st.button("🚀 Proceed to Content Selection", 
                 type="primary", 
                 disabled=not (topic.strip() and st.session_state.subtopics)):
        if topic.strip() and st.session_state.subtopics:
            st.session_state.topic = topic.strip()
            st.session_state.depth_level = depth_level
            st.session_state.generation_step = 'subtopic_content_selection'
            st.rerun()


agentt = AssistantAgent(
    name="PPTTextGenerator",
    system_message="""
You are an elite PowerPoint Content Creator specializing in professional, presentation-ready slide content. 

**Your Mission:**
Create visually stunning, concise slide content that rivals Gamma AI and Manus AI quality.

**Content Structure:**
1. **Professional Definition**: 1-2 lines that clearly explain the topic
2. **Impactful Key Points**: 4-6 powerful bullet points (10-15 words each)

**Quality Standards:**
• Each key point must be actionable and specific
• Use dynamic, engaging language with strong verbs
• Include technical specifics when relevant  
• Ensure each point offers unique value
• Avoid generic or repetitive statements
• Focus on capabilities, benefits, and applications

**Formatting Excellence:**
• Write for immediate presentation use
• Ensure professional tone throughout
• Make content scannable and memorable
• Prioritize clarity and visual impact
• Use precise, technical terminology when appropriate

**Examples of Strong Key Points:**
❌ Weak: "FTK analyzes data"  
✅ Strong: "Process disk images with advanced file recovery capabilities"

❌ Weak: "Autopsy is useful"
✅ Strong: "Extract and analyze metadata from digital artifacts systematically"

**Topic-Specific Guidelines:**
• Forensics: Focus on investigation capabilities, evidence processing, technical analysis
• Technology: Emphasize features, performance, integration capabilities  
• Business: Highlight ROI, efficiency gains, competitive advantages

Always deliver content ready for high-end presentation platforms.
""",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

content_generator = AssistantAgent(
    name="Content-generator",
    system_message='''You are a professional content generator for presentations with access to comprehensive research from vector database.

You are an expert at generating one of the following content types at a time:
- Text
- Code Snippet
- Mathematical Equation
- Table

Instructions:
- You will be provided with researched knowledge base and validated sources from vector database
- Use this comprehensive information to create accurate, up-to-date, and comprehensive content
- Include relevant examples and latest trends from the research
- Leverage full content from academic papers, documentation, and expert sources
- Only generate the requested type. Do not include anything else.
- Do not provide explanations or extra content.
- Your output must be in a format suitable for direct use in a slide.
- When generating text, incorporate findings from the comprehensive knowledge base
- For code snippets, use best practices and latest standards found in research
- For equations, ensure mathematical accuracy using verified sources
- Include citations when using specific information from sources

❌ Do NOT include:
- Headings or labels (like "Here is your text")
- Multiple content types at once
- Reasoning or notes

✅ Only respond with the requested content enhanced by comprehensive research findings.
''',
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

def categorize_source_type(url, title="", content=""):
    """Categorize source type based on URL and content"""
    url_lower = url.lower()
    title_lower = title.lower()
    
    if any(domain in url_lower for domain in ['arxiv.org', 'ieee.org', 'acm.org', 'springer.com']):
        return 'Academic Paper'
    elif any(domain in url_lower for domain in ['wikipedia.org', 'wiki']):
        return 'Wikipedia'
    elif any(domain in url_lower for domain in ['github.com', 'gitlab.com']):
        return 'Code Repository'
    elif any(domain in url_lower for domain in ['medium.com', 'blog', 'dev.to']):
        return 'Blog'
    elif any(domain in url_lower for domain in ['stackoverflow.com', 'stackexchange.com']):
        return 'Q&A Forum'
    elif any(word in title_lower for word in ['tutorial', 'guide', 'how to']):
        return 'Tutorial'
    elif any(domain in url_lower for domain in ['news', 'reuters.com', 'bbc.com']):
        return 'News Article'
    elif any(domain in url_lower for domain in ['docs.', 'documentation']):
        return 'Documentation'
    else:
        return 'Web Article'

def render_research_sources_section():
    """Render the research sources section with filtering and exclusion options"""
    
    st.markdown(research_sources_css, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="research-sources-container">
        <h2 style="color: #667eea; margin-bottom: 20px;">📚 Research Sources Used</h2>
        <p>Review the sources used to generate your presentation. You can exclude sources and regenerate content if needed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.research_sources:
        st.info("No research sources available. Generate a presentation first to see sources.")
        return
    
    # Source type filtering
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        source_types = list(set([source.get('type', 'Unknown') for source in st.session_state.research_sources]))
        selected_types = st.multiselect(
            "Filter by source type:",
            source_types,
            default=source_types,
            key="source_type_filter"
        )
    
    with col2:
        search_term = st.text_input(
            "Search sources:",
            placeholder="Search by title or URL...",
            key="source_search"
        )
    
    with col3:
        show_excluded = st.checkbox("Show excluded sources", key="show_excluded")
    
    # Summary statistics
    total_sources = len(st.session_state.research_sources)
    excluded_count = len(st.session_state.selected_sources)
    active_count = total_sources - excluded_count
    
    st.markdown(f"""
    **Summary:** {total_sources} total sources | {active_count} active | {excluded_count} excluded
    """)
    
    # Filter sources
    filtered_sources = []
    for i, source in enumerate(st.session_state.research_sources):
        # Type filter
        if source.get('type', 'Unknown') not in selected_types:
            continue
            
        # Search filter
        if search_term:
            if search_term.lower() not in source.get('title', '').lower() and \
               search_term.lower() not in source.get('url', '').lower():
                continue
        
        # Exclusion filter
        is_excluded = i in st.session_state.selected_sources
        if is_excluded and not show_excluded:
            continue
            
        filtered_sources.append((i, source, is_excluded))
    
    # Display sources
    for source_idx, source, is_excluded in filtered_sources:
        render_source_card(source_idx, source, is_excluded)
    
    # Bulk actions
    if st.session_state.research_sources:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Select All", key="select_all_sources"):
                st.session_state.selected_sources = set(range(len(st.session_state.research_sources)))
                st.rerun()
        
        with col2:
            if st.button("Clear Selection", key="clear_all_sources"):
                st.session_state.selected_sources = set()
                st.rerun()
        
        with col3:
            if st.button("Regenerate with Selected Sources", type="primary", key="regenerate_filtered"):
                regenerate_presentation_with_filtered_sources()
def regenerate_presentation_with_filtered_sources():
    """Regenerate presentation using only non-excluded sources"""
    try:
        # Get excluded indices from selected_sources
        excluded_indices = st.session_state.selected_sources
        
        if not st.session_state.research_sources or len(excluded_indices) == len(st.session_state.research_sources):
            st.error("No sources selected. Please include at least one source.")
            return
        
        active_count = len(st.session_state.research_sources) - len(excluded_indices)
        st.info(f"Regenerating presentation with {active_count} selected sources...")
        
        # Rebuild vector database with filtered sources
        rebuild_vector_db_with_filtered_sources(excluded_indices)
        
        # Update knowledge base with filtered sources
        filtered_sources = get_filtered_sources(excluded_indices)
        filtered_knowledge = {}
        
        for source in filtered_sources:
            topic = source.get('topic', 'general')
            content = source.get('content', '')
            if topic not in filtered_knowledge:
                filtered_knowledge[topic] = []
            if content:
                filtered_knowledge[topic].append(content)
        
        # Store the filtered knowledge
        st.session_state.knowledge_base = filtered_knowledge
        
        # Regenerate slides
        slides = generate_subtopic_based_slides(
            st.session_state.topic,
            st.session_state.subtopics,
            st.session_state.subtopic_component_structure,
            st.session_state.depth_level
        )
        
        st.session_state.slides = slides
        st.session_state.current_slide = 0
        
        st.success("Presentation regenerated successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error regenerating presentation: {str(e)}")

def render_source_card(source_idx, source, is_excluded):
    """Render an individual source card"""
    
    card_class = "source-card excluded" if is_excluded else "source-card"
    
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Source type badge
        source_type = source.get('type', 'Unknown')
        st.markdown(f'<span class="source-type-badge">{source_type}</span>', unsafe_allow_html=True)
        
        # Title and URL
        title = source.get('title', 'Untitled')
        url = source.get('url', '')
        
        st.markdown(f'<div class="source-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="source-url">{url}</div>', unsafe_allow_html=True)
        
        # Content preview
        content = source.get('content', '')
        if content:
            preview = content[:300] + "..." if len(content) > 300 else content
            st.markdown(f"""
            <div class="source-content-preview">
                <strong>Content Preview:</strong><br>
                {preview}
            </div>
            """, unsafe_allow_html=True)
        
        # Metadata
        if 'date' in source:
            st.caption(f"Date: {source['date']}")
        if 'word_count' in source:
            st.caption(f"Word count: {source['word_count']}")
    
    with col2:
        # Toggle exclusion
        if is_excluded:
            if st.button("Include", key=f"include_{source_idx}", help="Include this source"):
                st.session_state.selected_sources.discard(source_idx)
                st.rerun()
        else:
            if st.button("Exclude", key=f"exclude_{source_idx}", help="Exclude this source"):
                st.session_state.selected_sources.add(source_idx)
                st.rerun()
        
        # View full content
        if st.button("View Full", key=f"view_{source_idx}", help="View full content"):
            st.session_state[f"show_full_{source_idx}"] = not st.session_state.get(f"show_full_{source_idx}", False)
            st.rerun()
    
    # Show full content if requested
    if st.session_state.get(f"show_full_{source_idx}", False):
        with st.expander("Full Content", expanded=True):
            st.text_area(
                "Full extracted content:",
                value=source.get('content', 'No content available'),
                height=300,
                key=f"full_content_{source_idx}",
                disabled=True
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

def regenerate_presentation_with_filtered_sources():
    """Regenerate presentation using only non-excluded sources"""
    try:
        # Filter out excluded sources
        active_sources = [
            source for i, source in enumerate(st.session_state.research_sources)
            if i not in st.session_state.selected_sources
        ]
        
        if not active_sources:
            st.error("No sources selected. Please include at least one source.")
            return
        
        st.info(f"Regenerating presentation with {len(active_sources)} selected sources...")
        
        # Update knowledge base with filtered sources
        filtered_knowledge = {}
        for source in active_sources:
            if 'topic' in source:
                topic = source['topic']
                if topic not in filtered_knowledge:
                    filtered_knowledge[topic] = []
                filtered_knowledge[topic].append(source['content'])
        
        # Store the filtered knowledge
        st.session_state.knowledge_base = filtered_knowledge
        
        # Trigger regeneration
        st.session_state.generation_step = 'generating'
        st.rerun()
        
    except Exception as e:
        st.error(f"Error regenerating presentation: {str(e)}")

# Modified research function to store sources
def enhanced_serper_search(query, num_results=10):
    """Enhanced search function that stores source metadata"""
    try:
        # Your existing serper_search logic here
        # After extracting content, store source information
        
        source_info = {
            'title': 'Extracted Title',
            'url': 'https://example.com',
            'content': 'Extracted content...',
            'type': categorize_source_type('https://example.com', 'Extracted Title', 'content'),
            'date': 'extraction_date',
            'word_count': len('content'.split()),
            'topic': query
        }
        
        st.session_state.research_sources.append(source_info)
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")

# Update your main display function to include research sources
def enhanced_display_generated_presentation():
    """Enhanced presentation display with research sources tab"""
    
    # Create tabs for presentation and sources
    tab1, tab2 = st.tabs(["Presentation", "Research Sources"])
    
    with tab1:
        # Your existing presentation display code
        display_generated_presentation()
    
    with tab2:
        render_research_sources_section()
def generate_subtopic_suggestions(topic):
    """Generate subtopic suggestions using AI"""
    try:
        prompt = f"""
        For the topic "{topic}", suggest 4-6 relevant subtopics that would make a comprehensive presentation.
        
        Consider:
        - Key aspects and components of {topic}
        - Different perspectives (technical, practical, historical, future)
        - Applications and use cases
        - Challenges and opportunities
        
        Return only a comma-separated list of subtopics, no explanations.
        Example format: "Subtopic 1, Subtopic 2, Subtopic 3"
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT"], temperature=0.7)
        )
        
        suggestions = [s.strip() for s in response.text.split(",")]
        return suggestions[:6] if suggestions else []
        
    except Exception as e:
        print(f"Error generating subtopic suggestions: {e}")
        return []
def render_subtopic_content_selection_step():
    """Render the subtopic-based content selection step"""
    st.markdown(f"""
    <div class="step-indicator step-completed">
        ✅ Topic: {st.session_state.topic}
    </div>
    <div class="step-indicator step-completed">
        ✅ Subtopics: {len(st.session_state.subtopics)} defined
    </div>
    <div class="step-indicator">
        🎯 Step 2: Select Content Types for Each Subtopic
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### AI-Powered Content Type Selection")
    st.markdown("✨ Our AI will suggest relevant content types for each subtopic. You can customize these selections:")

    # Auto-generate content types for all subtopics if not already done
    if not st.session_state.subtopic_content_structure:
        if st.button("🤖 Generate Content Structure", type="primary"):
            generate_content_structure_for_subtopics()
            st.rerun()

    # Display and allow editing of content structure
    if st.session_state.subtopic_content_structure:
        for subtopic_idx, subtopic in enumerate(st.session_state.subtopics):
            # ✅ Ensure the subtopic key exists
            st.session_state.subtopic_content_structure.setdefault(subtopic, [])

            st.markdown(f"""
            <div class="subtopic-container">
                <div class="subtopic-title">📌 {subtopic}</div>
            """, unsafe_allow_html=True)

            # Get current content types for this subtopic
            current_content_types = st.session_state.subtopic_content_structure[subtopic]

            # Display current content types with option to remove
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Selected Content Types:**")
                content_types_to_remove = []

                for ct_idx, content_type in enumerate(current_content_types):
                    col_ct, col_remove = st.columns([3, 1])
                    with col_ct:
                        st.markdown(f"""
                        <div class="content-type-card selected">
                            {content_type}
                        </div>
                        """, unsafe_allow_html=True)

                    with col_remove:
                        if st.button("❌", key=f"remove_ct_{subtopic_idx}_{ct_idx}", help="Remove content type"):
                            content_types_to_remove.append(content_type)

                # Remove marked content types
                for ct in content_types_to_remove:
                    st.session_state.subtopic_content_structure[subtopic].remove(ct)
                    if content_types_to_remove:
                        st.rerun()

            with col2:
                st.markdown("**Add Content Type:**")
                available_types = [ct for ct in content_types_dict.keys() 
                                   if ct not in current_content_types]

                if available_types:
                    new_content_type = st.selectbox(
                        "Choose:",
                        [""] + available_types,
                        key=f"add_ct_{subtopic_idx}"
                    )

                    if st.button("➕ Add", key=f"add_ct_btn_{subtopic_idx}") and new_content_type:
                        # ✅ Safe append without KeyError
                        st.session_state.subtopic_content_structure.setdefault(subtopic, []).append(new_content_type)
                        st.rerun()
                else:
                    st.info("All types selected!")

            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Navigation buttons
        col_back, col_next = st.columns([1, 1])
        
        with col_back:
            if st.button("⬅️ Back to Topic Input", type="secondary"):
                st.session_state.generation_step = 'topic_subtopic_input'
                st.rerun()
        
        with col_next:
            if st.button("➡️ Customize Components", type="primary"):
                # Generate initial component structure
                generate_component_structure_for_subtopics()
                st.session_state.generation_step = 'subtopic_component_selection'
                st.rerun()

def generate_content_structure_for_subtopics():
    """Generate AI-powered content type suggestions for each subtopic"""
    st.session_state.subtopic_content_structure = {}
    
    for subtopic in st.session_state.subtopics:
        try:
            prompt = f"""
            For the subtopic "{subtopic}" under the main topic "{st.session_state.topic}", 
            select 3-5 most relevant content types from this list: {list(content_types_dict.keys())}
            
            Consider:
            - The specific nature of this subtopic
            - How it fits within the main topic "{st.session_state.topic}"
            - Audience level: {st.session_state.depth_level}
            - Educational flow and structure
            
            Return only a comma-separated list of content types, no explanations.
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["TEXT"], temperature=0.7)
            )
            
            suggested_types = [ct.strip() for ct in response.text.split(",")]
          
            if response and hasattr(response, 'text') and response.text:
                suggested_types = [ct.strip() for ct in response.text.split(",")]
                valid_types = [ct for ct in suggested_types if ct in content_types_dict]
                
                if valid_types:
                    st.session_state.subtopic_content_structure[subtopic] = valid_types[:5]
                else:
                    # Fallback to default selection
                    st.session_state.subtopic_content_structure[subtopic] = list(content_types_dict.keys())[:3]
            else:
                # Handle empty response
                print(f"Warning: Empty response for subtopic: {subtopic}")
                st.session_state.subtopic_content_structure[subtopic] = list(content_types_dict.keys())[:3]
                
        except Exception as e:
            print(f"Error generating content types for {subtopic}: {e}")
            # Fallback
            st.session_state.subtopic_content_structure[subtopic] = list(content_types_dict.keys())[:3]

def generate_component_structure_for_subtopics():
    """Generate component structure for each subtopic's content types"""
    st.session_state.subtopic_component_structure = {}
    
    for subtopic, content_types in st.session_state.subtopic_content_structure.items():
        st.session_state.subtopic_component_structure[subtopic] = {}
        
        for content_type in content_types:
            try:
                prompt = f"""
                For subtopic "{subtopic}" under main topic "{st.session_state.topic}", 
                and content type "{content_type}", select 3-5 most suitable components from: {available_components}
                
                Consider:
                - The specific subtopic context
                - The content type requirements
                - Audience level: {st.session_state.depth_level}
                - Visual and textual balance
                
                Return only a comma-separated list of components, no explanations.
                """
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_modalities=["TEXT"], temperature=0.7)
                )
                
                suggested_components = [comp.strip() for comp in response.text.split(",")]
                valid_components = [comp for comp in suggested_components if comp in available_components]
                
                if valid_components:
                    st.session_state.subtopic_component_structure[subtopic][content_type] = valid_components[:5]
                else:
                    # Fallback to default from content_types_dict
                    st.session_state.subtopic_component_structure[subtopic][content_type] = content_types_dict.get(content_type, ["Text"])[:3]
                    
            except Exception as e:
                print(f"Error generating components for {subtopic} - {content_type}: {e}")
                # Fallback
                st.session_state.subtopic_component_structure[subtopic][content_type] = content_types_dict.get(content_type, ["Text"])[:3]

def render_subtopic_component_selection_step():
    """Render the component-selection step – crash-proof."""
    st.markdown(f"""
    <div class="step-indicator step-completed">✅ Topic: {st.session_state.topic}</div>
    <div class="step-indicator step-completed">✅ Subtopics: {len(st.session_state.subtopics)} defined</div>
    <div class="step-indicator step-completed">✅ Content Types: Selected for each subtopic</div>
    <div class="step-indicator">🔧 Step 3: Customize Components for Each Content Type</div>
    """, unsafe_allow_html=True)

    st.markdown("### Fine-tune Components for Each Subtopic")

    # Ensure the nested dictionary exists for every subtopic/content-type pair
    for sub in st.session_state.subtopics:
        st.session_state.subtopic_component_structure.setdefault(sub, {})
        for ct in st.session_state.subtopic_content_structure.get(sub, []):
            st.session_state.subtopic_component_structure[sub].setdefault(ct, [])

    # ------------------------------------------------------------------
    # UI – identical logic as before, but now **never** crashes on KeyError
    # ------------------------------------------------------------------
    for subtopic in st.session_state.subtopics:
        st.markdown(f'<div class="subtopic-container"><div class="subtopic-title">📌 {subtopic}</div>',
                    unsafe_allow_html=True)

        for content_type in st.session_state.subtopic_content_structure.get(subtopic, []):
            st.markdown(f"#### {content_type}")
            col_cur, col_mod = st.columns([2, 1])

            # Current list (guaranteed to exist now)
            current = st.session_state.subtopic_component_structure[subtopic][content_type]

            # Remove buttons
            with col_cur:
                st.markdown("**Current components:**")
                to_remove = []
                for i, comp in enumerate(current):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f'<span class="component-tag">{comp}</span>', unsafe_allow_html=True)
                    with c2:
                        if st.button("❌", key=f"rm_{subtopic}_{content_type}_{i}"):
                            to_remove.append(comp)
                for comp in to_remove:
                    current.remove(comp)
                    st.rerun()

            # Add buttons
            with col_mod:
                st.markdown("**Add component:**")
                avail = [c for c in available_components if c not in current]
                if avail:
                    new_comp = st.selectbox("Choose:", [""] + avail, key=f"add_{subtopic}_{content_type}")
                    if st.button("➕ Add", key=f"add_btn_{subtopic}_{content_type}") and new_comp:
                        current.append(new_comp)
                        st.rerun()
                else:
                    st.info("All types selected!")

        st.markdown("</div>", unsafe_allow_html=True)

    # Navigation
    if st.button("⬅️ Back to Content Selection", type="secondary"):
        st.session_state.generation_step = 'subtopic_content_selection'
        st.rerun()

    if st.button("🎬 Generate Presentation", type="primary"):
        st.session_state.generation_step = 'generating'
        st.rerun()

def render_generation_step():
    """Render the generation step with progress for subtopic-based presentation"""
    st.markdown(f"""
    <div class="step-indicator step-completed">
        ✅ Topic: {st.session_state.topic}
    </div>
    <div class="step-indicator step-completed">
        ✅ Subtopics: {len(st.session_state.subtopics)} defined
    </div>
    <div class="step-indicator step-completed">
        ✅ Content Structure: Customized
    </div>
    <div class="step-indicator step-completed">
        ✅ Components: Fine-tuned
    </div>
    <div class="step-indicator">
        ⚡ Step 4: Generating Your Presentation
    </div>
    """, unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Clear previous research sources and vector DB
        if 'research_sources' not in st.session_state:
            st.session_state.research_sources = []
        else:
            st.session_state.research_sources.clear()
        
        clear_vector_db()
        
        # Step 1: Research main topic
        status_text.text(f"🔍 Researching main topic: {st.session_state.topic}...")
        progress_bar.progress(15)
        serper_search(st.session_state.topic, 20)  # This will populate research_sources
        
        # Step 2: Research each subtopic
        if st.session_state.subtopics:
            subtopic_progress = 50 // len(st.session_state.subtopics)
            
            for i, subtopic in enumerate(st.session_state.subtopics):
                status_text.text(f"🔍 Researching subtopic: {subtopic}...")
                current_progress = 15 + (i + 1) * subtopic_progress
                progress_bar.progress(min(current_progress, 65))
                
                # Search for each subtopic in context of main topic
                search_query = f"{st.session_state.topic} {subtopic}"
                serper_search(search_query, 15)  # This adds to research_sources
        
        # Step 3: Build knowledge base
        status_text.text("📚 Building knowledge base from research...")
        progress_bar.progress(70)
        
        # The knowledge base is already built in the vector store by serper_search
        # Just update our session state knowledge base for compatibility
        build_knowledge_base_from_research_sources()
        
        # Step 4: Generate slides
        status_text.text("🎨 Generating presentation slides...")
        progress_bar.progress(85)
        
        slides = generate_subtopic_based_slides(
            st.session_state.topic,
            st.session_state.subtopics,
            st.session_state.subtopic_component_structure,
            st.session_state.depth_level
        )
        
        progress_bar.progress(100)
        status_text.text(f"✅ Presentation generated successfully! Found {len(st.session_state.research_sources)} research sources.")
        
        st.session_state.slides = slides
        st.session_state.current_slide = 0
        st.session_state.generation_step = 'completed'
        
        sleep(2)  # Show completion message
        st.rerun()
        
    except Exception as e:
        st.error(f"Error generating presentation: {str(e)}")
        status_text.text("❌ Generation failed. Please try again.")
        
        if st.button("🔄 Retry Generation"):
            st.rerun()
        
        if st.button("⬅️ Back to Components"):
            st.session_state.generation_step = 'subtopic_component_selection'
            st.rerun()
def build_knowledge_base_from_research_sources():
    """Build knowledge base from collected research sources for compatibility"""
    st.session_state.knowledge_base = {}
    
    if 'research_sources' not in st.session_state:
        return
    
    for source in st.session_state.research_sources:
        topic = source.get('topic', 'general')
        content = source.get('content', '')
        
        if topic not in st.session_state.knowledge_base:
            st.session_state.knowledge_base[topic] = []
        
        if content and content not in st.session_state.knowledge_base[topic]:
            st.session_state.knowledge_base[topic].append(content)
def generate_subtopic_based_slides(topic, subtopics, subtopic_component_structure, depth_level="intermediate"):
    """Generate slides based on subtopic structure"""
    # This function would integrate with your existing slide generation logic
    # but organized by subtopics rather than single content structure
    
    slides = []
    slide_number = 1
    
    # Create title slide
    title_slide_content = [{
        "type": "text",
        "content": f"# {topic}\n\n**Subtopics:**\n" + "\n".join([f"• {st}" for st in subtopics]),
        "editable": True,
        "component_name": "title"
    }]
    
    title_slide = Slide(f"{topic}", title_slide_content, slide_number)
    slides.append(title_slide)
    slide_number += 1
    
    # Generate slides for each subtopic
    for subtopic in subtopics:
        subtopic_components = subtopic_component_structure.get(subtopic, {})
        
        for content_type, components in subtopic_components.items():
            # Generate slide title
            slide_title = f"{subtopic} - {content_type}"
            
            # Generate content sections (integrate with your existing content generation)
            content_sections = []
            
            for component in components:
                # Here you would call your existing content generation functions
                # but with subtopic context
                section = generate_content_for_subtopic_component(
                    component, content_type, subtopic, topic, depth_level,content_generator
                )
                content_sections.append(section)
            
            # Create slide
            slide = Slide(slide_title, content_sections, slide_number)
            slides.append(slide)
            slide_number += 1
    
    return slides

import os
import re
import json
import time
import requests
import numpy as np
from io import BytesIO
from base64 import b64decode
from PIL import Image, UnidentifiedImageError
from langchain.prompts import PromptTemplate
import streamlit as st
def generate_mermaid_diagram(payload: dict, vm_ip: str = "http://127.0.0.1:5500") -> str:
    url = f"http://{vm_ip}/render-mermaid/"
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def clean_latex_equation(math_expr: str) -> str:
    """
    Clean and normalize LaTeX equation string
    """
    # Remove common wrapper symbols
    expr = math_expr.strip()
    expr = re.sub(r'^\$+|\$+$', '', expr)  # Remove leading/trailing $
    expr = re.sub(r'^\\begin{equation}|\\end{equation}$', '', expr)  # Remove equation environment
    expr = re.sub(r'^\\begin{align}|\\end{align}$', '', expr)  # Remove align environment
    
    # Fix common LaTeX issues
    expr = re.sub(r'\\text{([^}]*)}', r'\\mathrm{\1}', expr)  # Replace \text with \mathrm
    expr = re.sub(r'\\textrm{([^}]*)}', r'\\mathrm{\1}', expr)  # Replace \textrm with \mathrm
    
    # Clean up whitespace
    expr = re.sub(r'\s+', ' ', expr).strip()
    
    return expr

def render_latex_to_image(math_expr: str, output_path="equation.png") -> str:
    """
    Render a LaTeX equation as a high-quality PNG image
    
    Args:
        math_expr: LaTeX equation string (without $ delimiters)
        output_path: Output file path
        
    Returns:
        Absolute path to the generated image
    """
    try:
        # Clean the equation
        expr = clean_latex_equation(math_expr)
        
        # Create figure with high DPI for crisp rendering
        fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
        ax.axis('off')
        
        # Render the equation
        ax.text(
            0.5, 0.5,
            f"${expr}$",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=24,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
        
        # Set tight layout
        plt.tight_layout(pad=0.2)
        
        # Save with high quality
        abs_path = os.path.abspath(output_path)
        fig.savefig(
            abs_path, 
            bbox_inches='tight', 
            transparent=False,  # White background for better visibility
            dpi=300,
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)
        
        return abs_path
        
    except Exception as e:
        print(f"ERROR: Failed to render LaTeX equation: {str(e)}")
        # Create a simple text fallback image
        return create_fallback_equation_image(math_expr, output_path)

def create_fallback_equation_image(math_expr: str, output_path: str) -> str:
    """
    Create a simple text-based fallback when LaTeX rendering fails
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 2), dpi=150)
        ax.axis('off')
        
        # Display as plain text
        ax.text(
            0.5, 0.5,
            f"Equation: {math_expr}",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8)
        )
        
        plt.tight_layout()
        abs_path = os.path.abspath(output_path)
        fig.savefig(abs_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close(fig)
        
        return abs_path
        
    except Exception as e:
        print(f"ERROR: Even fallback image creation failed: {str(e)}")
        return ""
    
def generate_content_for_subtopic_component_enhanced(
    component: str,
    content_type: str,
    subtopic: str,
    main_topic: str,
    depth_level: str,
    content_generator,
    trace_id=None
):
    """
    Generate content using the vector database for research context
    """
    
    # Query vector database for relevant context
    query = f"{main_topic} {subtopic} {content_type} {component}"
    research_context_results = query_vector_db(query, top_k=5, chunk_limit=300)
    
    # Combine research context
    research_context = "\n\n".join(research_context_results) if research_context_results else ""
    
    # Use the existing content generation logic but with enhanced context
    return generate_content_for_subtopic_component(
        component, content_type, subtopic, main_topic, depth_level, content_generator, trace_id
    )

def get_top_image_contexts(topic, content_type, component, top_k=10):
    """
    Retrieve top-k image paths based on similarity between text and image embeddings.
    Enhanced with vector database integration.
    """

    # Choose correct embedding and path file based on content_type
    if component.lower() == "real-world photo":
        emb_file = "embeddings_dino/real_images.npy"
        path_file = "embeddings_dino/real_images_paths.txt"
        query = f"Extract the relevant data for generating {component} on {content_type} of {topic}"
    else:
        emb_file = "embeddings_dino/diagrams.npy"
        path_file = "embeddings_dino/diagrams_paths.txt"
        query = f"Extract the relevant data on {content_type} of {topic} for generating {component}"

    # Check if embedding and path files exist
    if not (os.path.exists(emb_file) and os.path.exists(path_file)):
        return ""

    # Load image embeddings and image paths
    image_emb = np.load(emb_file)
    with open(path_file, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]

    if len(image_paths) != len(image_emb):
        raise ValueError("Mismatch between image paths and embedding vectors.")

    # Load sentence transformer for encoding the query
    text_encoder = SentenceTransformer("all-mpnet-base-v2")
    query_emb = text_encoder.encode(query, normalize_embeddings=True)

    # Convert to tensors
    query_emb = torch.tensor(query_emb, dtype=torch.float32)
    image_emb = torch.tensor(image_emb, dtype=torch.float32)

    # Compute cosine similarity
    scores = torch.nn.functional.cosine_similarity(query_emb, image_emb)
    top_indices = torch.topk(scores, k=min(top_k, len(scores))).indices.numpy()

    # Fetch the corresponding top image paths
    top_contexts = [image_paths[i] for i in top_indices]
    return "\n".join(top_contexts)
# def generate_content_for_subtopic_component(
#     component: str,
#     content_type: str,
#     subtopic: str,
#     main_topic: str,
#     depth_level: str,
#     # Langfuse instance
#     content_generator,        # LLM client
#     trace_id=None             # Optional trace ID for Langfuse
# ):
#     """
#     Generate content for a specific component within a sub-topic context.
#     Leverages Langfuse-prompts for all diagram, image, equation, graph and code components.
#     """

#     # ------------------------------------------------------------
#     # Helper: image download + validation
#     # ------------------------------------------------------------
#     import time
#     trace = langfuse_client.trace(
#         name="generate_presentation_slides",
#         input={
#             "topic": subtopic,
#             "depth_level": depth_level
#         },
#         metadata={
#             "function": "generate_presentation_slides",
#             "timestamp": time.time()
#         }
#     )
#     def _download_and_save_image(url: str, prefix: str = "asset") -> str:
#         """Download an image, validate, save locally and return absolute path."""
#         try:
#             img_data = requests.get(url, timeout=10).content
#             img_buffer = BytesIO(img_data)

#             # Validate image
#             with Image.open(img_buffer) as test_img:
#                 test_img.verify()

#             # Re-open for saving
#             img_buffer.seek(0)
#             img = Image.open(img_buffer)

#             if img.mode != "RGB":
#                 img = img.convert("RGB")

#             filename = f"{prefix}_{int(time.time())}.png"
#             img_path = os.path.abspath(filename)
#             img.save(img_path, "PNG", optimize=True)

#             if os.path.exists(img_path):
#                 return img_path
#         except Exception as e:
#             print(f"DEBUG: Image download/save failed: {e}")
#         return None

#     # ------------------------------------------------------------
#     # Shared research context block
#     # ------------------------------------------------------------
#     research_context = f"""
#     Main Topic: {main_topic}
#     Subtopic:   {subtopic}
#     Content Type: {content_type}
#     Audience Level: {depth_level}
#     """

#     # ------------------------------------------------------------
#     # TEXT component
#     # ------------------------------------------------------------
#     if component.lower() == "text":
#         try:
#             # Fetch prompt from Langfuse
#             prompt_obj = langfuse_client.get_prompt(
#                 name="text_general_generation_prompt",
#                 label="production"
#             )
#             prompt = prompt_obj.compile(
#                 research_context=research_context,
#                 content_type=content_type,
#                 topic=subtopic
#             )

#             response = content_generator.generate_reply([
#                 {"role": "user", "content": prompt}
#             ])

#             return {
#                 "type": "text",
#                 "content": response.strip(),
#                 "editable": True,
#                 "component_name": component
#             }

#         except Exception as e:
#             return {
#                 "type": "text",
#                 "content": f"Content for {component} in {subtopic} - {content_type}",
#                 "editable": True,
#                 "component_name": component
#             }

#     # ------------------------------------------------------------
#     # CODE SNIPPET component
#     # ------------------------------------------------------------
#     elif component.lower() == "code snippet":
#         try:
#             prompt_obj = langfuse_client.get_prompt(
#                 name="code_snippet_generation_prompt",
#                 label="production"
#             )
#             template = PromptTemplate(
#                 template=prompt_obj.prompt,
#                 input_variables=["topic"],
#                 partial_variables={}
#             )
#             prompt = template.format(topic=f"{main_topic} - {subtopic}")

#             raw = content_generator.generate_reply([
#                 {"role": "user", "content": prompt}
#             ])

#             # Clean code
#             code_clean = re.sub(r"```(?:python)?\s*\n(.*?)```", r"\1", raw, flags=re.DOTALL).strip()

#             return {
#                 "type": "code",
#                 "content": code_clean,
#                 "editable": True,
#                 "component_name": component
#             }

#         except Exception as e:
#             return {
#                 "type": "code",
#                 "content": f"# Code example for {subtopic}\nprint('{main_topic} - {subtopic}')",
#                 "editable": True,
#                 "component_name": component
#             }

#     # ------------------------------------------------------------
#     # MATHEMATICAL EQUATIONS component
#     # ------------------------------------------------------------
#     elif component.lower() == "mathematical equations":
#         try:
#             prompt_obj = langfuse_client.get_prompt(
#                 name="mathematical_equation_generation_prompt",
#                 label="production"
#             )
#             prompt = prompt_obj.compile(topic=f"{main_topic} - {subtopic}")

#             out = content_generator.generate_reply([
#                 {"role": "user", "content": prompt}
#             ])

#             lines = out.strip().split("\n")
#             latex_equation = ""
#             description = ""
#             for line in lines:
#                 if line.startswith("EQUATION:"):
#                     latex_equation = line.replace("EQUATION:", "").strip()
#                 elif line.startswith("DESCRIPTION:"):
#                     description = line.replace("DESCRIPTION:", "").strip()

#             if not latex_equation:
#                 latex_equation = out.strip()

#             # Render to PNG (assumes render_latex_to_image exists)
#             img_path = render_latex_to_image(
#                 latex_equation,
#                 output_path=f"equation_{int(time.time())}.png"
#             )

#             sections = []
#             if os.path.exists(img_path):
#                 sections.append({
#                     "type": "image",
#                     "content": img_path,
#                     "editable": True,
#                     "component_name": component
#                 })
#                 # if description:
#                 #     sections.append({
#                 #         "type": "text",
#                 #         "content": description,
#                 #         "editable": True,
#                 #         "component_name": f"{component}_description"
#                 #     })
#             return sections if sections else {
#                 "type": "text",
#                 "content": f"Mathematical equations related to {subtopic}",
#                 "editable": True,
#                 "component_name": component
#             }

#         except Exception as e:
#             return {
#                 "type": "text",
#                 "content": f"Mathematical equations related to {subtopic}",
#                 "editable": True,
#                 "component_name": component
#             }

#     # ------------------------------------------------------------
#     # DIAGRAM / FLOW / TABLE / ILLUSTRATION component
#     # ------------------------------------------------------------
#     # elif any(k in component.lower() for k in ["diagram", "flow", "table", "illustration"]):
#     #     try:
#     #         # 1. Fetch prompt
#     #         prompt_obj = langfuse_client.get_prompt(
#     #             name="diagram_generation_prompt",
#     #             label="production"
#     #         )
#     #         prompt = prompt_obj.compile(
#     #             topic=f"{main_topic}",
#     #             component=component,
#     #             context_block="",           # no image context here; pass empty
#     #             research_context=research_context
#     #         )

#     #         # 2. Get mermaid JSON
#     #         response = content_generator.generate_reply([
#     #             {"role": "user", "content": prompt}
#     #         ])

#     #         # Clean markdown fences
#     #         raw = response.strip()
#     #         if raw.startswith("```") and raw.endswith("```"):
#     #             raw = "\n".join([ln for ln in raw.splitlines() if not ln.strip().startswith("```")]).strip()

#     #         data = json.loads(raw)
#     #         mermaid_code = data["code"]

#     #         # 3. Generate diagram PNG (assumes generate_mermaid_diagram exists)
#     #         svg_b64 = generate_mermaid_diagram({"code": mermaid_code})
#     #         if svg_b64:
#     #             img_data = BytesIO(b64decode(svg_b64))
#     #             img = Image.open(img_data)

#     #             # Resize
#     #             max_w, min_w, max_h = 700, 300, 900
#     #             w, h = img.size
#     #             aspect = h / w
#     #             new_w = max(min(w, max_w), min_w)
#     #             new_h = new_w * aspect
#     #             if new_h > max_h:
#     #                 new_h = max_h
#     #                 new_w = new_h / aspect
#     #             img = img.resize((int(new_w), int(new_h)), Image.Resampling.LANCZOS)

#     #             filename = f"diagram_{int(time.time())}.png"
#     #             img_path = os.path.abspath(filename)
#     #             img.save(img_path, "PNG", optimize=True)

#     #             if os.path.exists(img_path):
#     #                 return {
#     #                     "type": "image",
#     #                     "content": img_path,
#     #                     "editable": True,
#     #                     "component_name": component,
#     #                     "mermaid_code": mermaid_code
#     #                 }

#     #     except Exception as e:
#     #         pass  # Fallback below

#     #     # Fallback text
#     #     return {
#     #         "type": "text",
#     #         "content": f"{component} for {subtopic}",
#     #         "editable": True,
#     #         "component_name": component
#     #     }

#     # # ------------------------------------------------------------
#     # # REAL-WORLD PHOTO component
#     # # ------------------------------------------------------------
#     # elif component.lower() == "real-world photo":
#     #     try:
#     #         prompt_obj = langfuse_client.get_prompt(
#     #             name="real_world_photo_generation",
#     #             label="production"
#     #         )
#     #         prompt = prompt_obj.compile(
#     #             topic=f"{main_topic} - {subtopic}",
#     #             context_block=""
#     #         )

#     #         response = content_generator.generate_reply([
#     #             {"role": "user", "content": prompt}
#     #         ])

#     #         # If the model returned an image inline, handle it
#     #         # (Assuming the generator supports images; adapt if not)
#     #         # For demonstration, we fall back to a placeholder
#     #         return {
#     #             "type": "text",
#     #             "content": f"Real-world photo placeholder for {subtopic}",
#     #             "editable": True,
#     #             "component_name": component
#     #         }

#     #     except Exception as e:
#     #         return {
#     #             "type": "text",
#     #             "content": f"Real-world photo placeholder for {subtopic}",
#     #             "editable": True,
#     #             "component_name": component
#     #         }
#     elif "diagram" in component.lower()  or "flow" in component.lower() or "table" in component.lower() or "illustration" in component.lower():
#                 # Generate mermaid diagram
#                 heading = f"**{component}**"
#                 import json
#                 context_block = get_top_image_contexts(main_topic, content_type, component, top_k=10)
                
#                 try:
#                     temp=langfuse_client.get_prompt(
#                     name="diagram_generation_prompt",label="production"
#                     )
#                     mer = temp.compile(
#                         topic = str(main_topic),
#                         component = str(component),
#                         context_block = str(context_block),
#                         research_context = str(research_context)
#                     )
                    
#                     # mer = f"""
#                     #     You are an expert in generating Mermaid diagrams in JSON format.

#                     #     Topic: "{topic}"
#                     #     Diagram Type: "{component}" or relevant type
#                     #     Reference diagram Context:\n{context_block}
#                     #     Reference text Context:\n {research_context}

#                     #     Now, generate a JSON output that corresponds to a mermaid diagram representing the above topic, following the referenced structure and insights.

#                     #     Task:
#                     #     – Generate a meaningful Mermaid diagram that accurately represents the topic based on the provided reference context  
#                     #     – Use less than 15  nodes, layout direction, and branches necessary
#                     #     – Focus on clarity, compactness, and contextual relevance
#                     #     – Ensure node labels are contextual and non-repetitive

#                     #     Output Format:
#                     #     Return ONLY a valid JSON object in the following format:  
#                     #     {{ "code": "<MERMAID_CODE>" }}

#                     #     Do NOT:
#                     #     - Repeat earlier structures
#                     #     - Use markdown or ``` syntax
#                     #     - Provide explanation
#                     #     - Overcomplicate with too many branches.
                        
#                     #     Return only valid Mermaid code without markdown blocks.
#                     #     """
                    
#                     # # Log the prompt
#                     # langfuse.create_prompt(
#                     #     name="diagram_generation_prompt",
#                     #     prompt=mer,
#                     #     type="text"
#                     # )
                    
#                     diagram_generation_span = langfuse_client.span(
#                         trace_id=trace.id,
#                         name="diagram_generation",
#                         input={
#                             "prompt": mer,
#                             "context_block_length": len(str(context_block)) if context_block else 0
#                         }
#                     )
                    
#                     response = client.models.generate_content(
#                         model="gemini-2.5-flash",
#                         contents=mer,
#                         config=types.GenerateContentConfig(response_modalities=["TEXT"], temperature=0.8, top_p=0.9)
#                     )
#                     response_text = response.text.strip()

# # Remove markdown fences if present
#                     if response_text.startswith("```") and response_text.endswith("```"):
#                         response_text = "\n".join(line for line in response_text.splitlines() if not line.strip().startswith("```")).strip()

#                     try:
#                         mermaid_json = json.loads(response_text)
#                         if not isinstance(mermaid_json, dict) or "code" not in mermaid_json:
#                             raise ValueError("Invalid JSON structure or missing 'code' field")
                        
#                         mermaid_code = mermaid_json["code"]

#                     except json.JSONDecodeError as json_err:
#                         print(f"DEBUG: JSON parsing failed: {json_err}")
#                         print(f"DEBUG: Raw model output: {response_text!r}")

#                         # Try extracting code using regex if JSON is malformed
                        
#                         code_match = re.search(r'"code"\s*:\s*"([^"]+)"', response_text)
#                         if code_match:
#                             mermaid_code = code_match.group(1)
#                         else:
#                             raise ValueError("Could not extract 'code' from model output")

#                     # mermaid_json = json.loads(response.text)
#                     # mermaid_code = mermaid_json['code']
                    
#                     # Generate diagram image
#                     svg = generate_mermaid_diagram({"code": mermaid_code})
                    
#                     if svg:
#                         sections.append({
#                             "type": "text",
#                             "content": heading,
#                             "editable": False,
#                             "component_name": "heading"
#                         })
                        
#                         import time
#                         timestamp = int(time.time())
#                         img_filename = f"diagram_{timestamp}_{component.replace(' ', '_')}.png"
#                         img_path = os.path.abspath(img_filename)
                        
#                         print(f"DEBUG: Saving diagram to: {img_path}")
                        
#                         try:
#                             image_data = BytesIO(b64decode(svg))
#                             img = Image.open(image_data)
                            
#                             # Apply sizing logic
#                             max_width, min_width, max_height = 700, 300, 900
#                             img_w, img_h = img.size
#                             aspect = img_h / img_w
#                             scaled_w = max(min(img_w, max_width), min_width)
#                             scaled_h = scaled_w * aspect
#                             if scaled_h > max_height:
#                                 scaled_h = max_height
#                                 scaled_w = scaled_h / aspect
                                
#                             img = img.resize((int(scaled_w), int(scaled_h)), Image.Resampling.LANCZOS)
                            
#                             # Save with error handling
#                             img.save(img_path, 'PNG', optimize=True)
                            
#                             # Verify file was created
#                             if os.path.exists(img_path):
#                                 print(f"DEBUG: Successfully saved diagram: {img_path} (size: {os.path.getsize(img_path)} bytes)")
                                
#                                 diagram_generation_span.update(
#                                     output={
#                                         "mermaid_code": mermaid_code,
#                                         "image_path": img_path,
#                                         "image_size": os.path.getsize(img_path),
#                                         "generation_success": True
#                                     }
#                                 )
                                
#                                 sections.append({
#                                     "type": "image",
#                                     "content": img_path,
#                                     "editable": True,
#                                     "component_name": component,
#                                     "mermaid_code": mermaid_code
#                                 })
#                             else:
#                                 raise Exception("File was not created successfully")
                                
#                         except Exception as save_error:
#                             print(f"DEBUG: Error saving diagram: {save_error}")
#                             raise save_error
                            
#                     else:
#                         raise Exception("SVG generation returned None")
                    
#                     diagram_generation_span.end()
                        
#                 except Exception as e:
#                     print(f"DEBUG: Diagram generation failed: {e}")
                    
#                     if 'diagram_generation_span' in locals():
#                         diagram_generation_span.update(
#                             output={"error": str(e), "generation_success": False}
#                         )
#                         diagram_generation_span.end()
                    
#                     # Fallback to text
#                     # temp=langfuse.get_prompt(
#                     # name="diagram_fallback_prompt",label="production"
#                     # )
#                     # fallback_prompt = temp.compile(
#                     #     component = str(component),
#                     #     topic = str(topic),
#                     #     knowledge_base = str(knowledge_base)
#                     # )
                    
#                     # fallback_prompt = f"""
#                     #  Summarize what the **{component}** for "{topic}" should ideally contain.

#                     # Use this context:
#                     # {knowledge_base}

#                     # Instructions:
#                     # - Start with a short **heading** that reflects the {component} type and topic
#                     # - Then provide 3–5 bullet points
#                     # - Each bullet should describe one clear, useful idea
#                     # - Use plain English (no jargon, no markdown, no fluff)
#                     # - No explanation or preamble
#                     # """
                    
#                     # Log fallback prompt
#                     # langfuse.create_prompt(
#                     #     name="diagram_fallback_prompt",
#                     #     prompt=fallback_prompt,
#                     #     type="text"
#                     # )
                    
#                     # fallback_span = langfuse.span(
#                     #     trace_id=trace.id,
#                     #     name="diagram_fallback_generation",
#                     #     input={"fallback_reason": str(e), "prompt": fallback_prompt}
#                     # )
                    
#                     # fallback_content = content_enrichment_agent.generate_reply([
#                     #     {
#                     #         "role": "user", 
#                     #         "content": fallback_prompt
#                     #     }
#                     # ])
                    
#                     # fallback_span.update(output={"fallback_content": fallback_content})
#                     # fallback_span.end()
                    
#                     # content_sections.append({
#                     #     "type": "text",
#                     #     "content": fallback_content,
#                     #     "editable": True,
#                     #     "component_name": component
#                     # })

#             # For real-world photo generation  
#     elif component.lower() == "real-world photo":
#                 try:
#                     unique_seed = np.random.randint(1000, 9999)
#                     timestamp = int(time.time())
#                     temp=langfuse_client.get_prompt(
#                     name="real_world_photo_generation",label="production"
#                     )
#                     photo_prompt = temp.compile(
#                         topic= str(main_topic),
#                         context_block = str(context_block)
#                     )
                    
#                     # photo_prompt = (
#                     #     f"Generate a realistic, high-resolution photograph that visually represents the real-world application of '{topic}'.\n"
#                     #     f"The image should resemble an authentic, unstaged snapshot of a practical scenario where '{topic}' is being used or demonstrated in action.\n"
#                     #     f"Avoid logos, branding, or any text overlays.\n"
#                     #     f"Refer to the following similar real-world scenes for inspiration:\n\n{context_block}\n\n"
#                     #     f"Ensure the final output maintains a natural look and clearly communicates the essence of '{topic}' without needing any explicit labels."
#                     # )
                    
#                     # # Log the prompt
#                     # langfuse.create_prompt(
#                     #     name="real_world_photo_generation_prompt",
#                     #     prompt=photo_prompt,
#                     #     type="text"
#                     # )
                    
#                     photo_generation_span = langfuse_client.span(
#                         trace_id=trace.id,
#                         name="real_world_photo_generation",
#                         input={
#                             "prompt": photo_prompt,
#                             "unique_seed": unique_seed,
#                             "timestamp": timestamp
#                         }
#                     )
                    
#                     response = client.models.generate_content(
#                         model="gemini-2.0-flash-preview-image-generation",
#                         contents=photo_prompt,
#                         config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
#                     )

#                     photo_generated = False
#                     for idx, part in enumerate(response.candidates[0].content.parts):
#                         if part.inline_data:
#                             # Create unique filename with absolute path
#                             img_filename = f"photo_{timestamp}_{unique_seed}_{idx}.png"
#                             img_path = os.path.abspath(img_filename)
                            
#                             print(f"DEBUG: Saving photo to: {img_path}")
                            
#                             try:
#                                 photo_data = BytesIO(part.inline_data.data)
#                                 photo_img = Image.open(photo_data)
                                
#                                 # Convert to RGB if needed
#                                 if photo_img.mode != 'RGB':
#                                     photo_img = photo_img.convert('RGB')
                                
#                                 # Save with error handling
#                                 photo_img.save(img_path, 'PNG', optimize=True)
                                
#                                 # Verify file was created
#                                 if os.path.exists(img_path):
#                                     print(f"DEBUG: Successfully saved photo: {img_path} (size: {os.path.getsize(img_path)} bytes)")
                                    
#                                     photo_generation_span.update(
#                                         output={
#                                             "image_path": img_path,
#                                             "image_size": os.path.getsize(img_path),
#                                             "generation_success": True,
#                                             "image_index": idx
#                                         }
#                                     )
                                    
#                                     sections.append({
#                                         "type": "image",
#                                         "content": img_path,
#                                         "editable": True,
#                                         "component_name": component
#                                     })
#                                     photo_generated = True
#                                     break
#                                 else:
#                                     raise Exception("Photo file was not created successfully")
                                    
#                             except Exception as save_error:
#                                 print(f"DEBUG: Error saving photo: {save_error}")
#                                 continue
                    
#                     if not photo_generated:
#                         raise Exception("No photo was generated from the response")
                    
#                     photo_generation_span.end()
                
                        
#                 except Exception as e:
#                     print(f"DEBUG: Photo generation failed: {e}")
                    
#                     if 'photo_generation_span' in locals():
#                         photo_generation_span.update(
#                             output={"error": str(e), "generation_success": False}
#                         )
#                         photo_generation_span.end()
#     # ------------------------------------------------------------
#     # GRAPH component
#     # ------------------------------------------------------------
#     elif component.lower() == "graph":
#         try:
#             # Step 1: graph idea prompt
#             prompt_obj = langfuse_client.get_prompt(
#                 name="graph_generation_prompt",
#                 label="production"
#             )
#             prompt = prompt_obj.compile(
#                 content_type=content_type,
#                 topic=f"{main_topic} - {subtopic}"
#             )

#             response = content_generator.generate_reply([
#                 {"role": "user", "content": prompt}
#             ])

#             raw = response.strip()
#             if raw.startswith("```") and raw.endswith("```"):
#                 raw = "\n".join([ln for ln in raw.splitlines() if not ln.strip().startswith("```")]).strip()

#             info = json.loads(raw)
#             graph_topic = info["graph_topic"]
#             graph_type = info["graph_type"]

#             # Step 2: Google image search via Serper
#             SERPER_API_KEY = os.getenv("SERPER_API_KEY")
#             if not SERPER_API_KEY:
#                 raise ValueError("SERPER_API_KEY missing")

#             query = f"{graph_topic} {graph_type} chart"
#             url = "https://google.serper.dev/images"
#             headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
#             payload = {"q": query, "num": 5}

#             resp = requests.post(url, headers=headers, json=payload)
#             data = resp.json()

#             if "images" in data and data["images"]:
#                 img_url = data["images"][0]["imageUrl"]
#                 img_path = _download_and_save_image(img_url, prefix="graph")
#                 if img_path:
#                     return {
#                         "type": "image",
#                         "content": img_path,
#                         "editable": True,
#                         "component_name": component,
#                         "graph_topic": graph_topic,
#                         "graph_type": graph_type
#                     }

#         except Exception as e:
#             print(f"DEBUG: Graph generation failed: {e}")

#     # ------------------------------------------------------------
#     # Fallback for anything else
#     # ------------------------------------------------------------
#     return {
#         "type": "text",
#         "content": f"**{component}**\n\nContent for {component} in {subtopic} - {content_type}",
#         "editable": True,
#         "component_name": component
    # }
# Slide class for better organization (same as original)
def format_presentation_content(raw_content, content_type, topic):
    """
    Format content for professional presentation with emojis and better structure
    """
    # Enhanced topic-specific emojis
    emoji_map = {
        "forensic": "🔍",
        "security": "🛡️", 
        "cyber": "🔐",
        "digital": "💻",
        "investigation": "🕵️",
        "analysis": "📊",
        "tools": "🛠️",
        "technology": "⚙️",
        "business": "📈",
        "marketing": "🎯",
        "finance": "💰",
        "healthcare": "🏥",
        "education": "📚",
        "environment": "🌱",
        "innovation": "💡",
        "data": "📊",
        "ai": "🤖",
        "default": "✨"
    }
    
    # Get appropriate emoji based on topic keywords
    topic_emoji = get_topic_emoji(topic.lower(), emoji_map)
    
    # Enhanced parsing to handle poorly formatted content
    content = raw_content.strip()
    
    # Remove redundant topic repetition
    if topic.lower() in content.lower():
        content = remove_redundant_title(content, topic)
    
    # Split and parse content more intelligently
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    definition = ""
    key_points = []
    
    # Extract definition and key points
    current_section = "definition"
    for line in lines:
        if "key points:" in line.lower():
            current_section = "points"
            continue
        elif line.startswith(('•', '-', '*', '▶️', '🔹')):
            clean_point = clean_bullet_point(line)
            if clean_point and len(clean_point) > 10:  # Ensure meaningful content
                key_points.append(enhance_key_point(clean_point))
        elif current_section == "definition" and not line.startswith(('•', '-', '*')):
            if not definition and len(line) > 15:  # Avoid short fragments
                definition = line
    
    # Generate professional definition if missing
    if not definition:
        definition = generate_smart_definition(topic, content_type)
    
    # Ensure we have quality key points
    if len(key_points) < 4:
        key_points = generate_enhanced_key_points(topic, content_type, key_points)
    
    # Format the final output with enhanced structure
    formatted_output = f"🎯 **{topic}**\n\n"
    formatted_output += f"_{definition}_\n\n"

    
    for i, point in enumerate(key_points[:6], 1):
        point_emoji = get_enhanced_point_emoji(i, content_type, topic)
        formatted_output += f"{point_emoji} **{point}**\n\n"
    
    return formatted_output.strip()
def clean_bullet_point(line):
    """Clean and enhance bullet points"""
    # Remove various bullet symbols and clean
    cleaned = line.lstrip('•-*▶️🔹🔸 ').strip()
    
    # Capitalize first letter if needed
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    return cleaned
def enhance_key_point(point):
    """Enhance key points for better presentation"""
    # Ensure proper capitalization and punctuation
    if not point.endswith('.') and len(point) > 20:
        point += ''  # Don't add period for short points
    
    # Make it more action-oriented if possible
    action_words = ['analyze', 'examine', 'detect', 'identify', 'support', 'enable']
    for word in action_words:
        if word in point.lower() and not point.startswith(word.capitalize()):
            # Already has action word, keep as is
            break
    
    return point
def generate_smart_definition(topic, content_type):
    """Generate intelligent definition based on topic"""
    forensic_definitions = {
       "":""
    }
    
    topic_lower = topic.lower()
    for key, definition in forensic_definitions.items():
        if key in topic_lower:
            return definition
    
    return f"Comprehensive overview of {topic.lower()} and their applications in digital investigation and analysis"

def generate_enhanced_key_points(topic, content_type, existing_points):
    """Generate enhanced key points when original content is insufficient"""
    enhanced_points = []
    
    if "forensic" in topic.lower():
        forensic_points = [
            "Process disk images with advanced file recovery capabilities",
            "Extract and analyze metadata from digital artifacts systematically", 
            "Examine volatile memory for hidden processes and malware",
            "Generate comprehensive forensic reports with chain of custody",
            "Support multiple file systems including NTFS, FAT, and HFS+",
            "Integrate with cloud platforms for modern digital investigations"
        ]
        enhanced_points = forensic_points
    
    # Combine existing with enhanced, removing duplicates
    all_points = existing_points + enhanced_points
    unique_points = []
    seen = set()
    
    for point in all_points:
        if point.lower()[:20] not in seen:  # Check first 20 chars to avoid near-duplicates
            unique_points.append(point)
            seen.add(point.lower()[:20])
    
    return unique_points[:6]

def get_topic_emoji(topic, emoji_map):
    """Get appropriate emoji based on topic keywords"""
    for keyword, emoji in emoji_map.items():
        if keyword in topic:
            return emoji
    return emoji_map["default"]

def get_enhanced_point_emoji(index, content_type, topic):
    """Get contextually relevant emojis for key points"""
    
    # Forensic/Security specific emojis
    forensic_emojis = ["🔍", "🛡️", "💾", "🔐", "📱", "⚡"]
    
    # Content type specific emojis  
    content_emojis = {
        "overview": ["🎯", "📋", "🔍", "⚙️", "💻", "🛠️"],
        "applications": ["🔧", "⚡", "🎯", "🚀", "💼", "🔥"], 
        "benefits": ["✅", "💪", "📈", "🎉", "⭐", "🏆"],
        "features": ["🔹", "⚙️", "🎛️", "🔍", "📱", "🖥️"],
        "challenges": ["⚠️", "🚧", "⛔", "📉", "🔴", "❗"],
        "solutions": ["💡", "🔑", "🛠️", "✨", "🎯", "🔧"],
        "trends": ["📊", "📈", "🔥", "🚀", "⏰", "📅"],
        "tools": ["🛠️", "🔧", "⚙️", "💻", "🔍", "📊"],
        "default": ["🚀", "💎", "⚡", "🎯", "✨", "🔥"]
    }
    
    # Use forensic emojis if topic is forensic-related
    if any(word in topic.lower() for word in ["forensic", "security", "investigation", "cyber"]):
        return forensic_emojis[(index - 1) % len(forensic_emojis)]
    
    # Get emoji set based on content type
    for key, emojis in content_emojis.items():
        if key in content_type.lower():
            return emojis[(index - 1) % len(emojis)]
    
    return content_emojis["default"][(index - 1) % len(content_emojis["default"])]

def generate_fallback_content(component, subtopic, content_type):
    """Generate professional fallback content with emojis"""
    topic_emoji = "✨"
    
    return f"""{topic_emoji} **{subtopic}**



**Key Points:**
🔹 Comprehensive overview and analysis
⚡ Strategic implementation approach  
📈 Measurable outcomes and benefits
🎯 Industry best practices integration
💡 Innovation-driven solutions
🚀 Future-ready methodology"""
def remove_redundant_title(content, topic):
    """Remove redundant topic repetitions from content"""
    lines = content.split('\n')
    cleaned_lines = []
    topic_lower = topic.lower()
    
    for line in lines:
        if line.strip() and not (topic_lower in line.lower() and len(line.strip()) < len(topic) + 20):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)
def get_topic_emoji(topic, emoji_map):
    """Get appropriate emoji based on topic keywords"""
    for keyword, emoji in emoji_map.items():
        if keyword in topic:
            return emoji
    return emoji_map["default"]

def get_point_emoji(index, content_type):
    """Get rotating emojis for key points based on content type"""
    content_emojis = {
        "applications": ["🔧", "⚡", "🎯", "🚀", "💼", "🔥"],
        "benefits": ["✅", "💪", "📈", "🎉", "⭐", "🏆"],
        "features": ["🔹", "⚙️", "🎛️", "🔍", "📱", "🖥️"],
        "challenges": ["⚠️", "🚧", "⛔", "📉", "🔴", "❗"],
        "solutions": ["💡", "🔑", "🛠️", "✨", "🎯", "🔧"],
        "trends": ["📊", "📈", "🔥", "🚀", "⏰", "📅"],
        "statistics": ["📊", "📈", "🔢", "📉", "💹", "📋"],
        "default": ["▶️", "🔸", "💫", "⚡", "🎯", "✨"]
    }
    
    # Get emoji set based on content type
    for key, emojis in content_emojis.items():
        if key in content_type.lower():
            return emojis[(index - 1) % len(emojis)]
    
    return content_emojis["default"][(index - 1) % len(content_emojis["default"])]

def generate_fallback_content(component, subtopic, content_type):
    """Generate professional fallback content with emojis"""
    topic_emoji = "✨"
    
    return f"""{topic_emoji} **{subtopic}**

_Professional content for {content_type.lower()} presentation_

**Key Points:**
🔹 Comprehensive overview and analysis
⚡ Strategic implementation approach  
📈 Measurable outcomes and benefits
🎯 Industry best practices integration
💡 Innovation-driven solutions
🚀 Future-ready methodology"""
def generate_content_for_subtopic_component(
    component: str,
    content_type: str,
    subtopic: str,
    main_topic: str,
    depth_level: str,
    content_generator,
    trace_id=None
):
    """
    Generate content for a specific component within a sub-topic context.
    Fixed version with proper variable initialization.
    """
    
    # Initialize sections list at the beginning
    sections = []
    
    # Helper function for image download and validation
    def _download_and_save_image(url: str, prefix: str = "asset") -> str:
        """Download an image, validate, save locally and return absolute path."""
        try:
            img_data = requests.get(url, timeout=10).content
            img_buffer = BytesIO(img_data)

            # Validate image
            with Image.open(img_buffer) as test_img:
                test_img.verify()

            # Re-open for saving
            img_buffer.seek(0)
            img = Image.open(img_buffer)

            if img.mode != "RGB":
                img = img.convert("RGB")

            filename = f"{prefix}_{int(time.time())}.png"
            img_path = os.path.abspath(filename)
            img.save(img_path, "PNG", optimize=True)

            if os.path.exists(img_path):
                return img_path
        except Exception as e:
            print(f"DEBUG: Image download/save failed: {e}")
        return None

    # Shared research context block
    research_context = f"""
    Main Topic: {main_topic}
    Subtopic:   {subtopic}
    Content Type: {content_type}
    Audience Level: {depth_level}
    """

    # Initialize Langfuse trace
    import time
    trace = langfuse_client.trace(
        name="generate_presentation_slides",
        input={
            "topic": subtopic,
            "depth_level": depth_level
        },
        metadata={
            "function": "generate_presentation_slides",
            "timestamp": time.time()
        }
    )

    # TEXT component
    if component.lower() == "text":
        try:
                # prompt_obj = langfuse_client.get_prompt(
                #     name="text_general_generation_prompt",
                #     label="production"
                # )
                # prompt = prompt_obj.compile(
                #     research_context=research_context,
                #     content_type=content_type,
                #     topic=subtopic
                # )
                prompt = f"""
                        Use this research context only as a reference:

                            {research_context}

                            Create clean, slide-ready content.

                            Instructions:
                            - Start with a short **Definition** of **{topic}** (max 2 lines, aim for 1–1.5)
                            - Add 2-3 crisp **Key Points** (each ≤ 1.5 lines, ideally 1)
                            - List 2-3 precise **Real-World Applications** (≤ 1.5 lines each)
                            - Use external knowledge, not just input
                            - Avoid fluff, rewording, or long phrases
                            - Tone: clear, minimal, and professional ({depth_level} audience)
                            - Format: {component} → {content_type}
                        """
                response = agentt.generate_reply(
                [{"role": "user", "content": prompt}],
                # Add these parameters if supported by your agent
                temperature=0.8,  # Increase randomness
                top_p=0.9,       # Nucleus sampling
                max_tokens=2000   # Ensure sufficient length
            )

                # Post-process the response for better presentation
                formatted_content = response.strip()

                return {
                    "type": "text",
                    "content": formatted_content,
                    "editable": True,
                    "component_name": component
                }

        except Exception as e:
        # Enhanced fallback content with emojis and better structure
            fallback_content = generate_fallback_content(component, subtopic, content_type)
            return {
                "type": "text",
                "content": fallback_content,
                "editable": True,
                "component_name": component
            }

    # CODE SNIPPET component
    elif component.lower() == "code snippet":
        try:
                temp=langfuse_client.get_prompt(
                    name="code_snippet_generation_prompt",label="production"
                )
                m=temp.prompt
                template = PromptTemplate(
                #     template="""
                # Generate a concise Python code snippet for the topic: '{topic}'.

                # Requirements:
                # - Fully runnable (no syntax errors)
                # - ≤ 30 lines
                # - 4-space indentation
                # - Use functions/classes if useful
                # - Avoid markdown formatting like backticks

                # {format_instructions}
                # """,
                template=m,
                    input_variables=["topic"],
                    partial_variables={"format_instructions": parser.get_format_instructions()}
                )

                prompt = template.format(topic=subtopic)
                
                # Log the prompt
                # langfuse.create_prompt(
                #     name="code_snippet_generation_prompt",
                #     prompt=prompt,
                #     type="text"
                # )
                
                code_generation_span = langfuse_client.span(
                    trace_id=trace.id,
                    name="code_snippet_generation",
                    input={"prompt": prompt}
                )
                
                raw_output = content_generator.generate_reply([{
                    "role": "user",
                    "content": prompt
                }])

                # Try structured parsing
                try:
                    parsed = parser.parse(raw_output)
                    code_clean = parsed.code
                    parsing_success = True
                except Exception as parse_error:
                    # Fallback to regex extraction
                    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw_output, re.DOTALL)
                    code_clean = match.group(1).strip() if match else raw_output.strip()
                    parsing_success = False
                
                code_generation_span.update(
                    output={
                        "raw_output": raw_output,
                        "code_clean": code_clean,
                        "parsing_success": parsing_success
                    }
                )
                code_generation_span.end()

                return {
                "type": "code",
                "content": code_clean,
                "editable": True,
                "component_name": component
            }

        except Exception as e:
            
            return {
                "type": "code",
                "content": f"",
                "editable": True,
                "component_name": ""
            }

    # MATHEMATICAL EQUATIONS component
    elif component.lower() == "mathematical equations":
        try:
            prompt_obj = langfuse_client.get_prompt(
                name="mathematical_equation_generation_prompt",
                label="production"
            )
            prompt = prompt_obj.compile(topic=f"{main_topic} - {subtopic}")

            out = content_generator.generate_reply([
                {"role": "user", "content": prompt}
            ])

            lines = out.strip().split("\n")
            latex_equation = ""
            description = ""
            for line in lines:
                if line.startswith("EQUATION:"):
                    latex_equation = line.replace("EQUATION:", "").strip()
                elif line.startswith("DESCRIPTION:"):
                    description = line.replace("DESCRIPTION:", "").strip()

            if not latex_equation:
                latex_equation = out.strip()

            # Render to PNG
            img_path = render_latex_to_image(
                latex_equation,
                output_path=f"equation_{int(time.time())}.png"
            )

            if os.path.exists(img_path):
                sections.append({
                    "type": "image",
                    "content": img_path,
                    "editable": True,
                    "component_name": component
                })
                
            return sections if sections else {
                "type": "text",
                "content": f"",
                "editable": True,
                "component_name": ""
            }

        except Exception as e:
            
            return {
                "type": "text",
                "content": "",
                "editable": True,
                "component_name": ""
            }

    # DIAGRAM / FLOW / TABLE / ILLUSTRATION components
    elif any(k in component.lower() for k in ["diagram", "flow", "table", "illustration"]):
        heading = f"**{component}**"
        context_block = get_top_image_contexts(main_topic, content_type, component, top_k=10)
        
        try:
            temp = langfuse_client.get_prompt(
                name="diagram_generation_prompt", 
                label="production"
            )
            topic=f"{main_topic} - {subtopic}"
            mer = temp.compile(
                topic=str(topic),
                component=str(component),
                context_block=str(context_block),
                research_context=str(research_context)
            )
            
            diagram_generation_span = langfuse_client.span(
                trace_id=trace.id,
                name="diagram_generation",
                input={
                    "prompt": mer,
                    "context_block_length": len(str(context_block)) if context_block else 0
                }
            )
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=mer,
                config=types.GenerateContentConfig(response_modalities=["TEXT"], temperature=0.8, top_p=0.9)
            )
            response_text = response.text.strip()

            # Remove markdown fences if present
            if response_text.startswith("```") and response_text.endswith("```"):
                response_text = "\n".join(line for line in response_text.splitlines() if not line.strip().startswith("```")).strip()

            try:
                mermaid_json = json.loads(response_text)
                if not isinstance(mermaid_json, dict) or "code" not in mermaid_json:
                    raise ValueError("Invalid JSON structure or missing 'code' field")
                
                mermaid_code = mermaid_json["code"]

            except json.JSONDecodeError as json_err:
                print(f"DEBUG: JSON parsing failed: {json_err}")
                print(f"DEBUG: Raw model output: {response_text!r}")

                # Try extracting code using regex if JSON is malformed
                code_match = re.search(r'"code"\s*:\s*"([^"]+)"', response_text)
                if code_match:
                    mermaid_code = code_match.group(1)
                else:
                    raise ValueError("Could not extract 'code' from model output")

            # Generate diagram image
            svg = generate_mermaid_diagram({"code": mermaid_code})
            
            if svg:
                sections.append({
                    "type": "text",
                    "content": heading,
                    "editable": False,
                    "component_name": "heading"
                })
                
                timestamp = int(time.time())
                img_filename = f"diagram_{timestamp}_{component.replace(' ', '_')}.png"
                img_path = os.path.abspath(img_filename)
                
                print(f"DEBUG: Saving diagram to: {img_path}")
                
                try:
                    image_data = BytesIO(b64decode(svg))
                    img = Image.open(image_data)
                    
                    # Apply sizing logic
                    max_width, min_width, max_height = 700, 300, 900
                    img_w, img_h = img.size
                    aspect = img_h / img_w
                    scaled_w = max(min(img_w, max_width), min_width)
                    scaled_h = scaled_w * aspect
                    if scaled_h > max_height:
                        scaled_h = max_height
                        scaled_w = scaled_h / aspect
                        
                    img = img.resize((int(scaled_w), int(scaled_h)), Image.Resampling.LANCZOS)
                    
                    # Save with error handling
                    img.save(img_path, 'PNG', optimize=True)
                    
                    # Verify file was created
                    if os.path.exists(img_path):
                        print(f"DEBUG: Successfully saved diagram: {img_path} (size: {os.path.getsize(img_path)} bytes)")
                        
                        diagram_generation_span.update(
                            output={
                                "mermaid_code": mermaid_code,
                                "image_path": img_path,
                                "image_size": os.path.getsize(img_path),
                                "generation_success": True
                            }
                        )
                        
                        sections.append({
                            "type": "image",
                            "content": img_path,
                            "editable": True,
                            "component_name": component,
                            "mermaid_code": mermaid_code
                        })
                    else:
                        raise Exception("File was not created successfully")
                        
                except Exception as save_error:
                    print(f"DEBUG: Error saving diagram: {save_error}")
                    raise save_error
                    
            else:
                raise Exception("SVG generation returned None")
            
            diagram_generation_span.end()
            
            # Return sections if we have any, otherwise fallback
            return sections if sections else {
                "type": "text",
                "content": f"",
                "editable": True,
                "component_name": ""
            }
                
        except Exception as e:
            print(f"DEBUG: Diagram generation failed: {e}")
            
            if 'diagram_generation_span' in locals():
                diagram_generation_span.update(
                    output={"error": str(e), "generation_success": False}
                )
                diagram_generation_span.end()
            
            # Fallback to text
            return {
                "type": "text",
                "content": f"",
                "editable": True,
                "component_name": ""
            }

    # REAL-WORLD PHOTO component
    elif component.lower() == "real-world photo":
        try:
            unique_seed = np.random.randint(1000, 9999)
            timestamp = int(time.time())
            
            temp = langfuse_client.get_prompt(
                name="real_world_photo_generation",
                label="production"
            )
            photo_prompt = temp.compile(
                topic=str(main_topic),
                context_block=str(context_block) if 'context_block' in locals() else ""
            )
            
            photo_generation_span = langfuse_client.span(
                trace_id=trace.id,
                name="real_world_photo_generation",
                input={
                    "prompt": photo_prompt,
                    "unique_seed": unique_seed,
                    "timestamp": timestamp
                }
            )
            
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=photo_prompt,
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
            )

            photo_generated = False
            for idx, part in enumerate(response.candidates[0].content.parts):
                if part.inline_data:
                    img_filename = f"photo_{timestamp}_{unique_seed}_{idx}.png"
                    img_path = os.path.abspath(img_filename)
                    
                    print(f"DEBUG: Saving photo to: {img_path}")
                    
                    try:
                        photo_data = BytesIO(part.inline_data.data)
                        photo_img = Image.open(photo_data)
                        
                        # Convert to RGB if needed
                        if photo_img.mode != 'RGB':
                            photo_img = photo_img.convert('RGB')
                        
                        # Save with error handling
                        photo_img.save(img_path, 'PNG', optimize=True)
                        
                        # Verify file was created
                        if os.path.exists(img_path):
                            print(f"DEBUG: Successfully saved photo: {img_path} (size: {os.path.getsize(img_path)} bytes)")
                            
                            photo_generation_span.update(
                                output={
                                    "image_path": img_path,
                                    "image_size": os.path.getsize(img_path),
                                    "generation_success": True,
                                    "image_index": idx
                                }
                            )
                            
                            sections.append({
                                "type": "image",
                                "content": img_path,
                                "editable": True,
                                "component_name": component
                            })
                            photo_generated = True
                            break
                        else:
                            raise Exception("Photo file was not created successfully")
                            
                    except Exception as save_error:
                        print(f"DEBUG: Error saving photo: {save_error}")
                        continue
            
            if not photo_generated:
                raise Exception("No photo was generated from the response")
            
            photo_generation_span.end()
            
            return sections if sections else {
                "type": "text",
                "content": f"",
                "editable": True,
                "component_name": ""
            }
                
        except Exception as e:
            print(f"DEBUG: Photo generation failed: {e}")
            
            if 'photo_generation_span' in locals():
                photo_generation_span.update(
                    output={"error": str(e), "generation_success": False}
                )
                photo_generation_span.end()
            
            return {
                "type": "text",
                "content": f"",
                "editable": True,
                "component_name": ""
            }

    # GRAPH component
    elif component.lower() == "graph":
        try:
            prompt_obj = langfuse_client.get_prompt(
                name="graph_generation_prompt",
                label="production"
            )
            prompt = prompt_obj.compile(
                content_type=content_type,
                topic=f"{main_topic} - {subtopic}"
            )

            response = content_generator.generate_reply([
                {"role": "user", "content": prompt}
            ])

            raw = response.strip()
            if raw.startswith("```") and raw.endswith("```"):
                raw = "\n".join([ln for ln in raw.splitlines() if not ln.strip().startswith("```")]).strip()

            info = json.loads(raw)
            graph_topic = info["graph_topic"]
            graph_type = info["graph_type"]

            # Google image search via Serper
            SERPER_API_KEY = os.getenv("SERPER_API_KEY")
            if not SERPER_API_KEY:
                raise ValueError("SERPER_API_KEY missing")

            query = f"{graph_topic} {graph_type} chart"
            url = "https://google.serper.dev/images"
            headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
            payload = {"q": query, "num": 5}

            resp = requests.post(url, headers=headers, json=payload)
            data = resp.json()

            if "images" in data and data["images"]:
                img_url = data["images"][0]["imageUrl"]
                img_path = _download_and_save_image(img_url, prefix="graph")
                if img_path:
                    return {
                        "type": "image",
                        "content": img_path,
                        "editable": True,
                        "component_name": component,
                        "graph_topic": graph_topic,
                        "graph_type": graph_type
                    }

        except Exception as e:
            print(f"DEBUG: Graph generation failed: {e}")

    # Fallback for anything else
    return {
        "type": "text",
        "content": f"",
        "editable": True,
        "component_name":""
    }
class Slide:
    def __init__(self, title, content_sections, slide_number):
        self.title = title
        self.content_sections = content_sections
        self.slide_number = slide_number
        self.comments = []
        self.research_sources = []

def render_slide_display(slide, slide_index):
    """Render a single slide with content"""
    
    # Calculate progress
    progress = ((slide_index + 1) / len(st.session_state.slides)) * 100
    
    # Slide container
    st.markdown(f"""
    <div class="slide-container fade-in">
        <div class="slide-header">
            {slide.title}
            <div class="slide-number">{slide.slide_number}/{len(st.session_state.slides)}</div>
        </div>
        <div class="progress-bar" style="width: {progress}%"></div>
    """, unsafe_allow_html=True)
    
    # Slide content
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    
    # Render each content section
    for section_index, section in enumerate(slide.content_sections):
        # Handle case where section might be a list (from mathematical equations, etc.)
        if isinstance(section, list):
            # If section is a list, process each item in the list
            for subsection in section:
                render_section_content(subsection)
        else:
            # Normal case where section is a dictionary
            render_section_content(section)
    
    st.markdown('</div></div>', unsafe_allow_html=True)
def render_section_content(section):
    """Helper function to render individual content sections"""
    try:
        if section["type"] == "text":
            st.markdown(section["content"])
        elif section["type"] == "code":
            st.code(section["content"], language="python")
        elif section["type"] == "image":
            try:
                image_path = section["content"]
                if os.path.exists(image_path):
                    st.image(image_path, caption=section.get('component_name', 'Generated Image'), use_container_width=True)
                else:
                    st.error(f"Image not found: {image_path}")
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        else:
            st.markdown(section["content"])
    except (KeyError, TypeError) as e:
        # Fallback for malformed sections
        st.error(f"Error rendering section: {e}")
        st.write("Raw section data:", section)
def display_generated_presentation():
    """Enhanced presentation display with research sources tab"""
    
    # Create tabs for presentation and sources
    tab1, tab2 = st.tabs(["📖 Presentation", "🔬 Research Sources"])
    
    with tab1:
        display_presentation_content()
    
    with tab2:
        render_research_sources_section()

def display_presentation_content():
    """Display the main presentation content"""
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🎮 Presentation Controls")
        
        # Research sources summary
        if 'research_sources' in st.session_state:
            total_sources = len(st.session_state.research_sources)
            excluded_count = len(st.session_state.selected_sources) if 'selected_sources' in st.session_state else 0
            active_count = total_sources - excluded_count
            
            st.markdown(f"""
            **📊 Research Summary:**
            - Total sources: {total_sources}
            - Active sources: {active_count}
            - Excluded sources: {excluded_count}
            """)
        
        # Navigation
        if st.session_state.slides:
            slide_options = [f"Slide {i+1}: {slide.title[:30]}..." if len(slide.title) > 30 
                           else f"Slide {i+1}: {slide.title}" 
                           for i, slide in enumerate(st.session_state.slides)]
            
            selected_slide = st.selectbox(
                "Navigate to slide:",
                range(len(st.session_state.slides)),
                format_func=lambda x: slide_options[x],
                index=st.session_state.current_slide
            )
            
            if selected_slide != st.session_state.current_slide:
                st.session_state.current_slide = selected_slide
                st.rerun()
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Previous", disabled=st.session_state.current_slide == 0):
                    st.session_state.current_slide -= 1
                    st.rerun()
            
            with col2:
                if st.button("➡️ Next", disabled=st.session_state.current_slide >= len(st.session_state.slides) - 1):
                    st.session_state.current_slide += 1
                    st.rerun()
            
            # Export options
            st.markdown("---")
            st.markdown("### 📤 Export Options")
            
            if st.button("📄 Export to PDF"):
                pdf_data = export_to_pdf(st.session_state.topic)
                if pdf_data:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_data,
                        file_name=f"{st.session_state.topic.replace(' ', '_')}_presentation.pdf",
                        mime="application/pdf"
                    )
            
            # Reset option
            st.markdown("---")
            if st.button("🔄 New Presentation", type="secondary"):
                # Clear everything including vector DB
                clear_vector_db()
                for key in ['slides', 'subtopics', 'subtopic_content_structure', 'subtopic_component_structure', 'topic', 'generation_step', 'research_sources', 'selected_sources', 'knowledge_base']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.generation_step = 'topic_subtopic_input'
                st.rerun()
    
    # Main content - Display current slide
    if st.session_state.slides:
        current_slide = st.session_state.slides[st.session_state.current_slide]
        render_slide_display(current_slide, st.session_state.current_slide)
       
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak,Preformatted
)
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_LEFT
def boldify(text: str) -> str:
    """
    Convert **text** to <b>text</b> for ReportLab Paragraphs.
    Keeps line-breaks as <br/>.
    """
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    return text.replace('\n', '<br/>')
def export_to_pdf(topic):
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # ---------- styles ----------
        title_style = ParagraphStyle(
            name='SlideTitle',
            fontSize=16,
            leading=20,
            spaceAfter=10,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            wordWrap='CJK'
        )
        body_style = ParagraphStyle(
            name='Body',
            parent=styles['Normal'],
            alignment=TA_LEFT,
            spaceAfter=6
        )
        code_style = ParagraphStyle(
            name='Code',
            fontSize=9,
            fontName='Courier',
            leftIndent=20,
            rightIndent=20,
            borderWidth=1,
            borderColor=colors.lightgrey,
            borderPadding=8,
            backColor=colors.lightgrey
        )

        # ---------- cover page ----------
        topic = getattr(st.session_state, "topic", "Untitled Topic")
        cover_title_style = ParagraphStyle(
            name='CoverTitle',
            fontSize=28,
            leading=36,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=40
        )
        story.append(Spacer(1, 200))  # Push title to middle
        story.append(Paragraph(topic, cover_title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Generated Slide Deck", title_style))
        story.append(PageBreak())

        # ---------- slides ----------
        for slide_idx, slide in enumerate(st.session_state.slides):
            story.append(Paragraph(slide.title, title_style))
            story.append(Spacer(1, 12))

            for section in slide.content_sections:
                # Handle case where section might be a list
                if isinstance(section, list):
                    # If section is a list, process each item in the list
                    for subsection in section:
                        process_pdf_section(subsection, story, body_style, code_style)
                else:
                    # Normal case where section is a dictionary
                    process_pdf_section(section, story, body_style, code_style)
                    
            story.append(PageBreak())

        # ---------- build PDF ----------
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

def process_pdf_section(section, story, body_style, code_style):
    """Helper function to process individual sections for PDF export"""
    try:
        if section["type"] == "text":
            story.append(Paragraph(boldify(section["content"]), body_style))
        elif section["type"] == "code":
            story.append(Preformatted(section["content"], code_style))
            story.append(Spacer(1, 6))
        elif section["type"] == "image":
            img_path = section["content"]
            if os.path.exists(img_path):
                try:
                    max_w, max_h = 5*inch, 4*inch
                    img = PILImage.open(img_path)
                    w, h = img.size
                    scale = min(max_w/w, max_h/h)
                    story.append(
                        RLImage(img_path,
                                width=w*scale,
                                height=h*scale)
                    )
                    story.append(Spacer(1, 6))
                except Exception:
                    story.append(Paragraph("[Image not rendered]", body_style))
    except (KeyError, TypeError) as e:
        # Fallback for malformed sections
        story.append(Paragraph(f"[Error rendering section: {e}]", body_style))
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; background: linear-gradient(45deg, #667eea, #764ba2); color: white; border-radius: 20px; margin-bottom: 30px;">
        <h1 style="margin: 0; font-size: 3em;">🎯 AI Presentation Creator</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">Create structured presentations with subtopic-based AI research</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content based on generation step
    if st.session_state.generation_step == 'topic_subtopic_input':
        render_topic_subtopic_input_step()
    elif st.session_state.generation_step == 'subtopic_content_selection':
        render_subtopic_content_selection_step()
    elif st.session_state.generation_step == 'subtopic_component_selection':
        render_subtopic_component_selection_step()
    elif st.session_state.generation_step == 'generating':
        render_generation_step()
    elif st.session_state.generation_step == 'completed':
        # Show completed presentation
        st.success("🎉 Presentation generated successfully!")
        
        # Add reset button at the top
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Create New Presentation"):
                # Reset all session state
                for key in ['slides', 'subtopics', 'subtopic_content_structure', 'subtopic_component_structure', 'topic', 'generation_step']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.generation_step = 'topic_subtopic_input'
                st.rerun()
        
        # Display slides
        display_generated_presentation()

if __name__ == "__main__":
    main()


