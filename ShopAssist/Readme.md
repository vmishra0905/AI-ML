ShopAssist AI - Intelligent Laptop Shopping Assistant
An AI-powered chatbot that helps users find the perfect laptop based on their requirements using Large Language Models (LLMs) and rule-based functions.

Project Overview
ShopAssist AI is an intelligent shopping assistant that combines the power of OpenAI's GPT models with structured data processing to provide personalized laptop recommendations. The system parses a laptop dataset and delivers accurate, relevant suggestions based on user requirements.

Problem Statement
Given a dataset containing laptop specifications (product names, specs, descriptions, etc.), build a chatbot that:

Parses the laptop dataset intelligently
Understands user requirements through conversation
Provides accurate laptop recommendations
Answers follow-up questions about recommended products

System Architecture
The ShopAssist AI system is built with three distinct layers:
1. User Intent Extraction Layer
Purpose: Understand what the user is looking for

Key Functions:

extract_user_requirement() - Extracts structured information from user queries
Identifies key specifications: processor, RAM, storage, price range, display, etc.
Converts natural language into structured JSON format

Example:

User: "I need a laptop for gaming under 60000"

Output: {

    "Core": "i7",

    "RAM Size": "16GB",

    "Graphics Processor": "NVIDIA GTX",

    "Price": "60000"

}
2. Product Comparison Layer
Purpose: Match user requirements with available products

Key Functions:

compare_laptops_with_user() - Scores each laptop against user requirements
recommendation_validation() - Filters top-scoring laptops
Ranks products based on specification matching
Returns top 3 recommendations with scores

Scoring Mechanism:

Each laptop gets a score (1-5) based on requirement match
Only laptops with score > 2 are recommended
Products ranked by relevance to user needs
3. Product Recommendation Layer
Purpose: Present recommendations and handle follow-up questions

Key Functions:

initialize_conv_reco() - Sets up recommendation conversation
Presents laptops in a structured format
Answers detailed questions about recommended products
Maintains conversation context

Output Format:

1. Lenovo ThinkPad: Ryzen 7, 16GB RAM, 14" IPS, NVIDIA GTX, Rs 60,000

2. MSI GL65: Intel Core i7, 16GB RAM, 15.6" IPS, NVIDIA GTX, Rs 55,000

3. Dell Inspiron: Intel Core i5, 8GB RAM, 15.6" LCD, Intel UHD, Rs 35,000
🎨 Key Features
✅ Intelligent Requirements Extraction
Natural language understanding
Handles vague queries ("good laptop", "gaming laptop")
Extracts multiple specifications simultaneously
Clarifies ambiguous requirements
✅ Smart Product Matching
Multi-criteria scoring system
Weighs different specifications appropriately
Filters irrelevant options
Ranks by best match
✅ Interactive Recommendations
Conversational interface
Detailed product information
Comparison capabilities
Follow-up question handling
✅ Safe & Reliable
Uses OpenAI Moderation API
Detects inappropriate queries
Handles prompt injection attempts
Provides consistent, accurate information
📊 Dataset Structure
The laptop dataset includes:

Field
Description
Example
Brand
Manufacturer
Dell, MSI, HP, Lenovo, ASUS
Model Name
Product model
Inspiron, GL65, EliteBook
Core
Processor type
i3, i5, i7, i9, Ryzen 7
CPU Manufacturer
Processor brand
Intel, AMD
Clock Speed
Processor speed
2.4 GHz, 2.6 GHz
RAM Size
Memory
4GB, 8GB, 16GB, 32GB, 64GB
Storage Type
Storage technology
HDD, SSD, HDD+SSD
Display Type
Screen technology
LCD, IPS, LED, OLED, TN
Display Size
Screen size
14", 15.6"
Graphics Processor
GPU
Intel UHD, NVIDIA GTX/RTX
Screen Resolution
Display resolution
1920x1080, 3840x2160
OS
Operating system
Windows 10/11, Linux
Laptop Weight
Weight
1.5 kg, 2.5 kg
Special Features
Extra features
Backlit Keyboard, Fingerprint Sensor
Warranty
Warranty period
1 year, 2 years, 3 years
Average Battery Life
Battery duration
4-8 hours
Price
Cost in Rs
25,000 - 200,000
Description
Detailed description
Full product description

🚀 How It Works
Step-by-Step Flow:
User Query

User: "I need a laptop for video editing under 100000"

Intent Extraction

{

  "Core": "i7 or above",

  "RAM Size": "16GB or more",

  "Graphics Processor": "Dedicated GPU",

  "Price": "100000"

}

Product Matching

System searches dataset
Scores each laptop (1-5)
Filters products with score > 2
Ranks by relevance

Recommendation

Based on your requirements, here are the top 3 laptops:

1. HP EliteBook: i7, 16GB RAM, Intel UHD, Rs 90,000

2. Lenovo ThinkPad: Ryzen 7, 16GB RAM, NVIDIA GTX, Rs 60,000

3. MSI GL65: i7, 16GB RAM, NVIDIA GTX, Rs 55,000

Follow-up Conversation

User: "Tell me more about the HP EliteBook"

Bot: "The HP EliteBook is a premium laptop designed for..."

Technical Implementation
Technologies Used:
OpenAI GPT-3.5/4 - Natural language processing
Python - Core programming language
Pandas - Data manipulation
JSON - Data interchange format
Google Colab - Development environment
Key Techniques:
1. Prompt Engineering
Clear Instructions: Structured task definitions
Role Assignment: Expert laptop advisor persona
Context Provision: Dataset and user profile context
Output Format: Structured JSON responses
Guidelines: Specific rules for handling queries
2. Chain-of-Thought Prompting
Step 1: Extract user requirements

Step 2: Search database for matches

Step 3: Score each laptop

Step 4: Rank and filter

Step 5: Present recommendations
3. Few-Shot Learning
Provides example inputs and outputs
Teaches desired response format
Improves accuracy and consistency
4. Self-Consistency
Multiple reasoning paths
Ensemble best outputs
Increases reliability
🎓 Educational Concepts Covered
This project demonstrates:
Prompt Engineering Principles:
Clear Instructions

Structured prompts (Task, Role, Context, Guidelines, Output Format)
Clear syntax and formatting
Conditioning on good performance

Enhanced LLM Reasoning

Chain-of-thought prompting
Few-shot learning with examples
Self-consistency techniques

Advanced Techniques

ReAct (Reasoning + Acting) prompting
System message optimization
Context management
System Design:
Modular Architecture

Separation of concerns
Clear layer responsibilities
Reusable components

Safety Mechanisms

Input validation
Moderation API integration
Prompt injection protection

User Experience

Conversational interface
Progressive disclosure
Context retention
📦 Setup & Installation
Prerequisites:
# Python 3.7+

# OpenAI API key

# Google Colab (recommended) or Jupyter Notebook
Installation:
# Install required packages

pip install openai==0.28 pandas
Configuration:
import openai

# Set your API key

openai.api_key = "your-api-key-here"

Use Cases
Primary Use Cases:

E-commerce platforms - Product recommendation systems

Sales assistance - Help customers find right products
Educational projects - Learn LLM application development

Research - Study conversational AI behavior
Industry Applications:
Online retail (Amazon, Flipkart)
Electronics stores
Corporate procurement
Customer service automation

Future Enhancements
Planned Features:
Multi-product category support (phones, tablets, etc.)
Price comparison across vendors
Image-based search
User preference learning
Review sentiment analysis
Budget optimization suggestions
Real-time price tracking
Voice interface integration
Technical Improvements:
Fine-tuned model for better accuracy
Vector database for faster search
Caching for repeated queries
A/B testing framework
User feedback loop
Multi-language support


Performance Metrics
Evaluation Criteria:
Relevance: How well recommendations match user needs
Accuracy: Correct specification extraction rate
Response Time: Time to generate recommendations
User Satisfaction: Feedback scores
Conversation Quality: Natural, helpful interactions
Expected Performance:
Requirement extraction accuracy: >90%
Recommendation relevance: >85%
Average response time: 2-5 seconds
User satisfaction: >4/5

Safety & Ethics
Safety Features:
Input moderation
Output filtering
Prompt injection detection
Rate limiting
Error handling
Ethical Considerations:
Transparent recommendations
No hidden biases
Clear pricing information
Honest product comparisons
Privacy protection


Acknowledgments
OpenAI for GPT models
Course instructors for project design
Community for feedback and suggestions

Support
For questions or issues:

Review the notebook comments
Check OpenAI documentation
Verify API key and version (0.28)
Ensure proper dataset format



Built with for intelligent shopping assistance

Empowering users to make informed purchase decisions through AI


