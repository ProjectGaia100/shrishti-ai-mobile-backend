"""
AI Service for Gemini Integration
Handles AI model interactions and conversation management
"""
import google.generativeai as genai
import logging
import time
from typing import Dict, Any, Optional, List
import json

class AIService:
    """Service class for Gemini AI operations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.initialized = False
        self.logger = logging.getLogger(__name__)
        
        # System prompt for disaster monitoring context
        self.system_prompt = """You are GEO VISION AI, an intelligent assistant specializing in disaster monitoring and satellite data analysis. 

Your capabilities include:
- Analyzing satellite imagery and remote sensing data
- Providing insights on natural disasters (floods, droughts, storms, landslides)
- Explaining vegetation indices (NDVI, NDWI, NBR) and their significance
- Offering guidance on disaster preparedness and response
- Interpreting geographic and meteorological data

Please provide accurate, helpful responses focused on disaster monitoring, satellite data analysis, and geographic information systems. Be concise but informative, and always prioritize safety in disaster-related advice."""

    def initialize(self) -> bool:
        """Initialize the Gemini AI model"""
        try:
            genai.configure(api_key=self.api_key)
            
            # Initialize the model with safety settings
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            
            # Try models in order of preference (gemini-2.5-flash first)
            model_names = [
                'gemini-2.5-flash',
                'gemini-2.0-flash',
                'gemini-1.5-flash',
            ]
            
            for model_name in model_names:
                try:
                    self.logger.info(f"Trying AI model: {model_name}")
                    self.model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                        system_instruction=self.system_prompt
                    )
                    # Quick test to verify the model works
                    test_resp = self.model.generate_content("Hello, respond with just OK")
                    self.logger.info(f"AI model configured: {model_name} - {test_resp.text}")
                    self.initialized = True
                    return True
                except Exception as model_err:
                    self.logger.warning(f"Model {model_name} failed: {model_err}")
                    continue
            
            self.logger.error("All AI models failed to initialize")
            return False
            
        except Exception as e:
            self.logger.error(f"AI model initialization failed: {str(e)}")
            return False
    
    def generate_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate AI response with retry mechanism
        
        Args:
            message: User message
            context: Optional context information (satellite data, location, etc.)
            
        Returns:
            Dictionary containing response and metadata
        """
        if not self.initialized:
            return {
                'status': 'error',
                'message': 'AI service not initialized'
            }
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Prepare the full prompt with context
                full_prompt = self._prepare_prompt(message, context)
                
                # Generate response
                response = self.model.generate_content(full_prompt)
                
                # Check if response was generated successfully
                if response.text:
                    return {
                        'status': 'success',
                        'message': response.text,
                        'model': 'gemini-1.5-flash',
                        'attempt': attempt + 1
                    }
                else:
                    self.logger.warning(f"Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                self.logger.error(f"AI generation error on attempt {attempt + 1}: {str(e)}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return {
                        'status': 'error',
                        'message': f'Failed to generate response after {max_retries} attempts: {str(e)}'
                    }
        
        return {
            'status': 'error',
            'message': 'Failed to generate response'
        }
    
    def _prepare_prompt(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare the full prompt with context information"""
        if not context:
            return message
            
        context_parts = []
        
        # Add satellite data context if available
        if 'satellite_data' in context:
            sat_data = context['satellite_data']
            if sat_data.get('status') == 'success':
                context_parts.append(f"Recent satellite data analysis for location ({sat_data.get('location', {}).get('latitude', 'N/A')}, {sat_data.get('location', {}).get('longitude', 'N/A')}):")
                context_parts.append(f"- Available images: {sat_data.get('count', 0)}")
                
                if 'latest_image' in sat_data:
                    latest = sat_data['latest_image']
                    context_parts.append(f"- Latest image date: {latest.get('date', 'N/A')}")
                    context_parts.append(f"- Cloud coverage: {latest.get('cloud_coverage', 'N/A')}%")
        
        # Add location context if available
        if 'location' in context:
            loc = context['location']
            context_parts.append(f"Location: {loc.get('latitude', 'N/A')}, {loc.get('longitude', 'N/A')}")
        
        # Add any indices data if available
        if 'indices' in context:
            indices = context['indices']
            context_parts.append("Current vegetation and water indices:")
            for key, value in indices.items():
                if value is not None:
                    context_parts.append(f"- {key}: {value:.3f}")
        
        # Combine context and message
        if context_parts:
            context_text = "\n".join(context_parts)
            return f"Context:\n{context_text}\n\nUser Question: {message}"
        
        return message
    
    def analyze_satellite_data(self, satellite_data: Dict[str, Any], user_query: str = None) -> Dict[str, Any]:
        """
        Analyze satellite data and provide insights
        
        Args:
            satellite_data: Satellite data from GEE service
            user_query: Optional specific query about the data
            
        Returns:
            AI analysis results
        """
        if not self.initialized:
            return {
                'status': 'error',
                'message': 'AI service not initialized'
            }
        
        try:
            # Prepare analysis prompt
            analysis_prompt = "Please analyze the following satellite data and provide insights on potential disaster indicators:\n\n"
            analysis_prompt += json.dumps(satellite_data, indent=2)
            
            if user_query:
                analysis_prompt += f"\n\nSpecific question: {user_query}"
            
            # Generate analysis
            response = self.model.generate_content(analysis_prompt)
            
            return {
                'status': 'success',
                'analysis': response.text,
                'data_analyzed': True
            }
            
        except Exception as e:
            self.logger.error(f"Satellite data analysis error: {str(e)}")
            return {
                'status': 'error',
                'message': f'Analysis failed: {str(e)}'
            }
    
    def get_disaster_insights(self, disaster_type: str, location_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get specific insights about a disaster type
        
        Args:
            disaster_type: Type of disaster (flood, drought, storm, landslide)
            location_data: Optional location-specific data
            
        Returns:
            Disaster-specific insights
        """
        if not self.initialized:
            return {
                'status': 'error',
                'message': 'AI service not initialized'
            }
        
        try:
            insights_prompt = f"""Provide comprehensive insights about {disaster_type} disasters, including:

1. Key indicators visible in satellite imagery
2. Typical precursor conditions
3. Monitoring strategies using remote sensing
4. Safety and preparedness recommendations

"""
            
            if location_data:
                insights_prompt += f"Consider this location context: {json.dumps(location_data, indent=2)}"
            
            response = self.model.generate_content(insights_prompt)
            
            return {
                'status': 'success',
                'insights': response.text,
                'disaster_type': disaster_type
            }
            
        except Exception as e:
            self.logger.error(f"Disaster insights error: {str(e)}")
            return {
                'status': 'error',
                'message': f'Insights generation failed: {str(e)}'
            }