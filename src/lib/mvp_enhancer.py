#!/usr/bin/env python3
"""
MVP Enhancer

This module takes user prompts and enhances them into comprehensive MVP specifications
that include all necessary components for a shareable NextJS application.

The enhancer makes user requests more technical and comprehensive without adding
features they didn't ask for - it just ensures proper technical implementation.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class MVPSpecification:
    """Enhanced MVP specification derived from user prompt."""
    original_prompt: str
    enhanced_prompt: str
    core_features: List[str]
    technical_requirements: List[str]
    ui_components: List[str]
    data_structure: Dict[str, any]
    routing_structure: List[str]
    styling_approach: str
    complexity_level: str  # 'simple', 'moderate', 'complex'
    estimated_components: int
    suggested_tech_stack: List[str]


class MVPEnhancer:
    """
    Enhances user prompts into comprehensive MVP specifications.
    
    Takes simple user ideas and transforms them into detailed technical
    specifications that include all necessary components for a complete,
    shareable NextJS application.
    """
    
    def __init__(self):
        """Initialize the MVP enhancer."""
        # Initialize OpenAI client for enhancement
        self.openai_client = None
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
            except ImportError:
                print("⚠️ OpenAI package not installed")
        
        print("🎯 MVP Enhancer initialized")
    
    def enhance_prompt_to_mvp(self, user_prompt: str) -> MVPSpecification:
        """
        Enhance a user prompt into a comprehensive MVP specification.
        
        Args:
            user_prompt: The original user idea/prompt
            
        Returns:
            MVPSpecification: Enhanced technical specification
        """
        print("🎯 Enhancing user prompt to MVP specification...")
        print(f"📝 Original prompt: {user_prompt}")
        
        # Use LLM to enhance the prompt
        enhancement_prompt = self._get_enhancement_prompt(user_prompt)
        
        enhanced_response = self._call_enhancement_llm(enhancement_prompt)
        
        if not enhanced_response:
            # Fallback to manual enhancement
            return self._create_fallback_mvp(user_prompt)
        
        # Parse the enhanced response
        mvp_spec = self._parse_mvp_specification(enhanced_response, user_prompt)
        
        print(f"✅ Enhanced to MVP with {len(mvp_spec.core_features)} core features")
        print(f"🏗️ Complexity level: {mvp_spec.complexity_level}")
        print(f"📊 Estimated components: {mvp_spec.estimated_components}")
        
        # 🖨️ PRINT THE ENHANCED PROMPT TO TERMINAL
        print("\n" + "=" * 80)
        print("🎯 ENHANCED MVP PROMPT:")
        print("=" * 80)
        print(mvp_spec.enhanced_prompt)
        print("\n📋 CORE FEATURES:")
        for i, feature in enumerate(mvp_spec.core_features, 1):
            print(f"  {i}. {feature}")
        print(f"\n🏗️ COMPLEXITY: {mvp_spec.complexity_level}")
        print(f"📊 ESTIMATED COMPONENTS: {mvp_spec.estimated_components}")
        print(f"🎨 STYLING: {mvp_spec.styling_approach}")
        print("=" * 80)
        
        return mvp_spec
    
    def _get_enhancement_prompt(self, user_prompt: str) -> str:
        """Generate the enhancement prompt for the LLM."""
        return f"""You are an expert product manager and technical architect. Your task is to transform a user's simple idea into a comprehensive FRONTEND-ONLY MVP specification for a NextJS application.

CRITICAL RULES:
1. DO NOT add features the user didn't ask for
2. DO make their request more technical and comprehensive
3. Focus on making it a COMPLETE, SHAREABLE FRONTEND application
4. NO authentication, NO database, NO backend APIs - FRONTEND ONLY
5. Use mock data, local storage, or component state for data management
6. Think about what makes a frontend app "production-ready" vs just a demo

USER'S ORIGINAL IDEA:
"{user_prompt}"

Transform this into a comprehensive FRONTEND MVP specification that includes:

1. ENHANCED PROMPT: Rewrite the user's idea as a detailed, frontend-focused specification
2. CORE FEATURES: List the essential frontend features (based on user's request only)
3. TECHNICAL REQUIREMENTS: What's needed technically (responsive design, state management, local storage, etc.)
4. UI COMPONENTS: Specific React components needed
5. DATA STRUCTURE: How data should be organized in component state or local storage
6. ROUTING STRUCTURE: What pages/routes are needed
7. STYLING APPROACH: Best approach for styling this type of app
8. TECH STACK: Recommended frontend technologies (within NextJS ecosystem)

Format your response as JSON:
{{
  "enhanced_prompt": "Detailed frontend-focused technical description...",
  "core_features": ["feature1", "feature2", ...],
  "technical_requirements": ["requirement1", "requirement2", ...],
  "ui_components": ["component1", "component2", ...],
  "data_structure": {{"model1": {{"field1": "type", ...}}, ...}},
  "routing_structure": ["/route1", "/route2", ...],
  "styling_approach": "tailwind/styled-components/css-modules/etc",
  "complexity_level": "simple/moderate/complex",
  "estimated_components": 5,
  "suggested_tech_stack": ["nextjs", "react", "typescript", "tailwindcss", ...]
}}

Example enhancement:
User says: "Todo app"
Enhanced: "A comprehensive todo management frontend application with local storage persistence, drag-and-drop reordering, filtering and categorization, responsive design for mobile and desktop. Users can create, edit, delete, and organize todos with categories, priorities, and due dates. All data stored in browser local storage with export/import functionality."

Make it PRODUCTION-READY FRONTEND, not just a demo. Focus on rich UI/UX, not backend complexity."""
    
    def _call_enhancement_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM to enhance the prompt."""
        try:
            if not self.openai_client:
                print("⚠️ No OpenAI client available for enhancement")
                return None
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert product manager and technical architect specializing in NextJS applications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Error calling enhancement LLM: {e}")
            return None
    
    def _parse_mvp_specification(self, response: str, original_prompt: str) -> MVPSpecification:
        """Parse the LLM response into an MVPSpecification object."""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                return MVPSpecification(
                    original_prompt=original_prompt,
                    enhanced_prompt=parsed.get('enhanced_prompt', original_prompt),
                    core_features=parsed.get('core_features', []),
                    technical_requirements=parsed.get('technical_requirements', []),
                    ui_components=parsed.get('ui_components', []),
                    data_structure=parsed.get('data_structure', {}),
                    routing_structure=parsed.get('routing_structure', ['/']),
                    styling_approach=parsed.get('styling_approach', 'tailwind'),
                    complexity_level=parsed.get('complexity_level', 'moderate'),
                    estimated_components=parsed.get('estimated_components', 5),
                    suggested_tech_stack=parsed.get('suggested_tech_stack', ['nextjs', 'react', 'typescript'])
                )
            else:
                return self._create_fallback_mvp(original_prompt)
                
        except json.JSONDecodeError:
            print("⚠️ Could not parse JSON response, creating fallback MVP")
            return self._create_fallback_mvp(original_prompt)
        except Exception as e:
            print(f"⚠️ Error parsing MVP specification: {e}")
            return self._create_fallback_mvp(original_prompt)
    
    def _create_fallback_mvp(self, original_prompt: str) -> MVPSpecification:
        """Create a basic MVP specification as fallback."""
        return MVPSpecification(
            original_prompt=original_prompt,
            enhanced_prompt=f"A comprehensive frontend-only NextJS application implementing: {original_prompt}. Include responsive design, interactive UI components, local state management, localStorage persistence, error handling, loading states, and production-ready code structure. Focus on rich user experience without backend dependencies.",
            core_features=[f"Interactive frontend for: {original_prompt}", "Responsive design", "Local data persistence"],
            technical_requirements=["responsive design", "error handling", "loading states", "localStorage integration", "component state management"],
            ui_components=["Layout", "Navigation", "MainContent", "Footer", "LoadingSpinner", "ErrorBoundary"],
            data_structure={"main": {"id": "string", "data": "any", "timestamp": "Date"}},
            routing_structure=["/", "/about"],
            styling_approach="tailwind",
            complexity_level="moderate",
            estimated_components=5,
            suggested_tech_stack=["nextjs", "react", "typescript", "tailwindcss"]
        )
    
    def format_mvp_for_coordinator(self, mvp_spec: MVPSpecification) -> str:
        """Format the MVP specification for the LLM coordinator."""
        return f"""MVP SPECIFICATION FOR NEXTJS APPLICATION

ENHANCED REQUIREMENT:
{mvp_spec.enhanced_prompt}

CORE FEATURES TO IMPLEMENT:
{chr(10).join(f"- {feature}" for feature in mvp_spec.core_features)}

TECHNICAL REQUIREMENTS:
{chr(10).join(f"- {req}" for req in mvp_spec.technical_requirements)}

UI COMPONENTS NEEDED:
{chr(10).join(f"- {component}" for component in mvp_spec.ui_components)}

DATA STRUCTURE:
{json.dumps(mvp_spec.data_structure, indent=2)}

ROUTING STRUCTURE:
{chr(10).join(f"- {route}" for route in mvp_spec.routing_structure)}

STYLING APPROACH: {mvp_spec.styling_approach}
COMPLEXITY LEVEL: {mvp_spec.complexity_level}
ESTIMATED COMPONENTS: {mvp_spec.estimated_components}

TECH STACK:
{chr(10).join(f"- {tech}" for tech in mvp_spec.suggested_tech_stack)}

This is a comprehensive MVP that should result in a complete, shareable, production-ready NextJS application.""" 