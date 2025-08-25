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
from base import serper_search, query_vector_db

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
}

.source-card:hover {
    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}

.source-card.excluded {
    opacity: 0.5;
    border-color: #ff6b6b;
    background: #fff5f5;
}

.source-title {
    font-size: 16px;
    font-weight: 600;
    color: #333;
    margin-bottom: 8px;
}

.source-url {
    font-size: 12px;
    color: #666;
    margin-bottom: 10px;
    word-break: break-all;
}

.source-type-badge {
    display: inline-block;
    background: #667eea;
    color: white;
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
}

.source-actions {
    text-align: right;
    margin-top: 10px;
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
            valid_types = [ct for ct in suggested_types if ct in content_types_dict]
            
            if valid_types:
                st.session_state.subtopic_content_structure[subtopic] = valid_types[:5]
            else:
                # Fallback to default selection
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
        # Step 1: Research and vector DB
        status_text.text("🔍 Researching topic and building knowledge base...")
        progress_bar.progress(15)
        serper_search(st.session_state.topic, 20)
        
        # Research each subtopic
        for i, subtopic in enumerate(st.session_state.subtopics):
            status_text.text(f"🔍 Researching subtopic: {subtopic}...")
            progress_bar.progress(15 + (i + 1) * 20 // len(st.session_state.subtopics))
            serper_search(f"{st.session_state.topic} {subtopic}", 15)
        
        # Step 2: Generate slides
        status_text.text("🎨 Generating presentation slides...")
        progress_bar.progress(60)
        
        slides = generate_subtopic_based_slides(
            st.session_state.topic,
            st.session_state.subtopics,
            st.session_state.subtopic_component_structure,
            st.session_state.depth_level
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Presentation generated successfully!")
        
        st.session_state.slides = slides
        st.session_state.current_slide = 0
        st.session_state.generation_step = 'completed'
        
        sleep(1)  # Brief pause to show completion
        st.rerun()
        
    except Exception as e:
        st.error(f"Error generating presentation: {str(e)}")
        status_text.text("❌ Generation failed. Please try again.")
        
        if st.button("🔄 Retry Generation"):
            st.rerun()
        
        if st.button("⬅️ Back to Components"):
            st.session_state.generation_step = 'subtopic_component_selection'
            st.rerun()

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
    
    title_slide = Slide(f"{topic} - Overview", title_slide_content, slide_number)
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
def generate_mermaid_diagram(payload: dict, vm_ip: str = "40.81.228.142:5500") -> str:
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
    


def generate_content_for_subtopic_component(
    component: str,
    content_type: str,
    subtopic: str,
    main_topic: str,
    depth_level: str,
    # Langfuse instance
    content_generator,        # LLM client
    trace_id=None             # Optional trace ID for Langfuse
):
    """
    Generate content for a specific component within a sub-topic context.
    Leverages Langfuse-prompts for all diagram, image, equation, graph and code components.
    """

    # ------------------------------------------------------------
    # Helper: image download + validation
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Shared research context block
    # ------------------------------------------------------------
    research_context = f"""
    Main Topic: {main_topic}
    Subtopic:   {subtopic}
    Content Type: {content_type}
    Audience Level: {depth_level}
    """

    # ------------------------------------------------------------
    # TEXT component
    # ------------------------------------------------------------
    if component.lower() == "text":
        try:
            # Fetch prompt from Langfuse
            prompt_obj = langfuse_client.get_prompt(
                name="text_general_generation_prompt",
                label="production"
            )
            prompt = prompt_obj.compile(
                research_context=research_context,
                content_type=content_type,
                topic=subtopic
            )

            response = content_generator.generate_reply([
                {"role": "user", "content": prompt}
            ])

            return {
                "type": "text",
                "content": response.strip(),
                "editable": True,
                "component_name": component
            }

        except Exception as e:
            return {
                "type": "text",
                "content": f"Content for {component} in {subtopic} - {content_type}",
                "editable": True,
                "component_name": component
            }

    # ------------------------------------------------------------
    # CODE SNIPPET component
    # ------------------------------------------------------------
    elif component.lower() == "code snippet":
        try:
            prompt_obj = langfuse_client.get_prompt(
                name="code_snippet_generation_prompt",
                label="production"
            )
            template = PromptTemplate(
                template=prompt_obj.prompt,
                input_variables=["topic"],
                partial_variables={}
            )
            prompt = template.format(topic=f"{main_topic} - {subtopic}")

            raw = content_generator.generate_reply([
                {"role": "user", "content": prompt}
            ])

            # Clean code
            code_clean = re.sub(r"```(?:python)?\s*\n(.*?)```", r"\1", raw, flags=re.DOTALL).strip()

            return {
                "type": "code",
                "content": code_clean,
                "editable": True,
                "component_name": component
            }

        except Exception as e:
            return {
                "type": "code",
                "content": f"# Code example for {subtopic}\nprint('{main_topic} - {subtopic}')",
                "editable": True,
                "component_name": component
            }

    # ------------------------------------------------------------
    # MATHEMATICAL EQUATIONS component
    # ------------------------------------------------------------
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

            # Render to PNG (assumes render_latex_to_image exists)
            img_path = render_latex_to_image(
                latex_equation,
                output_path=f"equation_{int(time.time())}.png"
            )

            sections = []
            if os.path.exists(img_path):
                sections.append({
                    "type": "image",
                    "content": img_path,
                    "editable": True,
                    "component_name": component
                })
                if description:
                    sections.append({
                        "type": "text",
                        "content": description,
                        "editable": True,
                        "component_name": f"{component}_description"
                    })
            return sections if sections else {
                "type": "text",
                "content": f"Mathematical equations related to {subtopic}",
                "editable": True,
                "component_name": component
            }

        except Exception as e:
            return {
                "type": "text",
                "content": f"Mathematical equations related to {subtopic}",
                "editable": True,
                "component_name": component
            }

    # ------------------------------------------------------------
    # DIAGRAM / FLOW / TABLE / ILLUSTRATION component
    # ------------------------------------------------------------
    elif any(k in component.lower() for k in ["diagram", "flow", "table", "illustration"]):
        try:
            # 1. Fetch prompt
            prompt_obj = langfuse_client.get_prompt(
                name="diagram_generation_prompt",
                label="production"
            )
            prompt = prompt_obj.compile(
                topic=f"{main_topic} - {subtopic}",
                component=component,
                context_block="",           # no image context here; pass empty
                research_context=research_context
            )

            # 2. Get mermaid JSON
            response = content_generator.generate_reply([
                {"role": "user", "content": prompt}
            ])

            # Clean markdown fences
            raw = response.strip()
            if raw.startswith("```") and raw.endswith("```"):
                raw = "\n".join([ln for ln in raw.splitlines() if not ln.strip().startswith("```")]).strip()

            data = json.loads(raw)
            mermaid_code = data["code"]

            # 3. Generate diagram PNG (assumes generate_mermaid_diagram exists)
            svg_b64 = generate_mermaid_diagram({"code": mermaid_code})
            if svg_b64:
                img_data = BytesIO(b64decode(svg_b64))
                img = Image.open(img_data)

                # Resize
                max_w, min_w, max_h = 700, 300, 900
                w, h = img.size
                aspect = h / w
                new_w = max(min(w, max_w), min_w)
                new_h = new_w * aspect
                if new_h > max_h:
                    new_h = max_h
                    new_w = new_h / aspect
                img = img.resize((int(new_w), int(new_h)), Image.Resampling.LANCZOS)

                filename = f"diagram_{int(time.time())}.png"
                img_path = os.path.abspath(filename)
                img.save(img_path, "PNG", optimize=True)

                if os.path.exists(img_path):
                    return {
                        "type": "image",
                        "content": img_path,
                        "editable": True,
                        "component_name": component,
                        "mermaid_code": mermaid_code
                    }

        except Exception as e:
            pass  # Fallback below

        # Fallback text
        return {
            "type": "text",
            "content": f"{component} for {subtopic}",
            "editable": True,
            "component_name": component
        }

    # ------------------------------------------------------------
    # REAL-WORLD PHOTO component
    # ------------------------------------------------------------
    elif component.lower() == "real-world photo":
        try:
            prompt_obj = langfuse_client.get_prompt(
                name="real_world_photo_generation",
                label="production"
            )
            prompt = prompt_obj.compile(
                topic=f"{main_topic} - {subtopic}",
                context_block=""
            )

            response = content_generator.generate_reply([
                {"role": "user", "content": prompt}
            ])

            # If the model returned an image inline, handle it
            # (Assuming the generator supports images; adapt if not)
            # For demonstration, we fall back to a placeholder
            return {
                "type": "text",
                "content": f"Real-world photo placeholder for {subtopic}",
                "editable": True,
                "component_name": component
            }

        except Exception as e:
            return {
                "type": "text",
                "content": f"Real-world photo placeholder for {subtopic}",
                "editable": True,
                "component_name": component
            }

    # ------------------------------------------------------------
    # GRAPH component
    # ------------------------------------------------------------
    elif component.lower() == "graph":
        try:
            # Step 1: graph idea prompt
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

            # Step 2: Google image search via Serper
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

    # ------------------------------------------------------------
    # Fallback for anything else
    # ------------------------------------------------------------
    return {
        "type": "text",
        "content": f"**{component}**\n\nContent for {component} in {subtopic} - {content_type}",
        "editable": True,
        "component_name": component
    }
# Slide class for better organization (same as original)
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
    
    st.markdown('</div></div>', unsafe_allow_html=True)

def display_generated_presentation():
    """Display the generated presentation with navigation and editing capabilities"""
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🎮 Presentation Controls")
        
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
                # Reset all session state
                for key in ['slides', 'subtopics', 'subtopic_content_structure', 'subtopic_component_structure', 'topic', 'generation_step']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.generation_step = 'topic_subtopic_input'
                st.rerun()
    
    # Main content - Display current slide
    if st.session_state.slides:
        current_slide = st.session_state.slides[st.session_state.current_slide]
        render_slide_display(current_slide, st.session_state.current_slide)
        
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
def export_to_pdf(topic):
    """Export the **sub-topic** presentation to PDF – no crashes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                        Spacer, PageBreak)
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.enums import TA_CENTER

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # cover
        cover = ParagraphStyle('cover', parent=styles['Title'], alignment=TA_CENTER)
        story.append(Spacer(1, 200))
        story.append(Paragraph(topic, cover))
        story.append(Spacer(1, 30))
        if st.session_state.subtopics:
            story.append(Paragraph("Subtopics:", styles['Heading2']))
            for st in st.session_state.subtopics:
                story.append(Paragraph(f"• {st}", styles['Normal']))
        story.append(PageBreak())

        # slides
        for slide in st.session_state.slides:
            story.append(Paragraph(slide.title, styles['Heading1']))
            story.append(Spacer(1, 12))
            for sec in slide.content_sections:
                if sec['type'] == 'text':
                    story.append(Paragraph(sec['content'], styles['Normal']))
                elif sec['type'] == 'code':
                    story.append(Paragraph(f"<font name='Courier'>{sec['content']}</font>",
                                           styles['Normal']))
                story.append(Spacer(1, 6))
            story.append(PageBreak())

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None

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