#!/usr/bin/env python3
"""
Request Enhancer

Enhances user edit requests to make them more technical, specific, and relevant 
to the actual codebase structure and components.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class RequestEnhancer:
    """
    Enhances user edit requests by analyzing the codebase and making requests
    more technical and specific.
    """
    
    def __init__(self):
        """Initialize the request enhancer."""
        self.openai_client = None
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
            except ImportError:
                print("⚠️ OpenAI package not installed")
        
        print("🎯 Request Enhancer initialized")
    
    def enhance_app_idea(self, app_idea: str) -> str:
        """
        Enhance an app idea for new app creation to make it more technical and specific.
        
        Args:
            app_idea: Original app idea (e.g., "todo app", "weather dashboard")
            
        Returns:
            Enhanced technical app specification with implementation details
        """
        print(f"🎯 Enhancing app idea: '{app_idea}'")
        
        # Use LLM to enhance the app idea if available
        if self.openai_client:
            enhanced_idea = self._llm_enhance_app_idea(app_idea)
        else:
            enhanced_idea = self._fallback_enhance_app_idea(app_idea)
        
        print(f"✅ Enhanced app idea: '{enhanced_idea[:100]}...'")
        return enhanced_idea
    
    def enhance_edit_request(self, 
                           user_request: str, 
                           app_path: str,
                           context_summary: str = "") -> str:
        """
        Enhance a user's edit request to make it more technical and specific.
        
        Args:
            user_request: Original user request (e.g., "add a navbar")
            app_path: Path to the app directory
            context_summary: Summary of relevant files and components
            
        Returns:
            Enhanced technical request with specific implementation details
        """
        print(f"🎯 Enhancing edit request: '{user_request}'")
        
        # Get codebase structure
        codebase_analysis = self._analyze_codebase_structure(app_path)
        
        # Use LLM to enhance the request
        if self.openai_client:
            enhanced_request = self._llm_enhance_request(
                user_request, 
                codebase_analysis,
                context_summary
            )
        else:
            enhanced_request = self._fallback_enhance_request(
                user_request, 
                codebase_analysis
            )
        
        print(f"✅ Enhanced request: '{enhanced_request[:100]}...'")
        return enhanced_request
    
    def _analyze_codebase_structure(self, app_path: str) -> Dict[str, Any]:
        """Analyze the codebase structure to understand components and architecture."""
        app_dir = Path(app_path)
        structure = {
            'components': [],
            'pages': [],
            'types': [],
            'utilities': [],
            'existing_imports': [],
            'framework_info': {
                'is_nextjs': False,
                'app_router': False,
                'uses_typescript': False,
                'uses_tailwind': False,
                'uses_shadcn': False
            }
        }
        
        if not app_dir.exists():
            return structure
        
        # Check framework indicators
        package_json = app_dir / 'package.json'
        if package_json.exists():
            try:
                with open(package_json) as f:
                    pkg_data = json.load(f)
                    deps = {**pkg_data.get('dependencies', {}), **pkg_data.get('devDependencies', {})}
                    
                    structure['framework_info']['is_nextjs'] = 'next' in deps
                    structure['framework_info']['uses_typescript'] = 'typescript' in deps
                    structure['framework_info']['uses_tailwind'] = 'tailwindcss' in deps
                    structure['framework_info']['uses_shadcn'] = '@radix-ui/react-slot' in deps
            except:
                pass
        
        # Check for App Router
        if (app_dir / 'app').exists():
            structure['framework_info']['app_router'] = True
        
        # Scan for components
        components_dir = app_dir / 'components'
        if components_dir.exists():
            for file_path in components_dir.rglob('*.tsx'):
                rel_path = str(file_path.relative_to(app_dir))
                structure['components'].append(rel_path)
        
        # Scan for pages/routes
        app_pages_dir = app_dir / 'app'
        if app_pages_dir.exists():
            for file_path in app_pages_dir.rglob('page.tsx'):
                rel_path = str(file_path.relative_to(app_dir))
                structure['pages'].append(rel_path)
        
        # Scan for types
        types_dir = app_dir / 'types'
        if types_dir.exists():
            for file_path in types_dir.rglob('*.ts'):
                rel_path = str(file_path.relative_to(app_dir))
                structure['types'].append(rel_path)
        
        # Scan for utilities
        for util_dir_name in ['lib', 'utils', 'helpers']:
            util_dir = app_dir / util_dir_name
            if util_dir.exists():
                for file_path in util_dir.rglob('*.ts'):
                    rel_path = str(file_path.relative_to(app_dir))
                    structure['utilities'].append(rel_path)
        
        return structure
    
    def _llm_enhance_request(self, 
                           user_request: str, 
                           codebase_analysis: Dict[str, Any],
                           context_summary: str) -> str:
        """Use LLM to enhance the request with technical specificity."""
        
        enhancement_prompt = f"""You are a senior software engineer analyzing a user's edit request to make it more technical and specific to their actual codebase.

USER REQUEST: "{user_request}"

CODEBASE ANALYSIS:
{json.dumps(codebase_analysis, indent=2)}

RELEVANT CONTEXT:
{context_summary}

INSTRUCTIONS:
Think step by step about what the user actually wants to accomplish:

1. **ANALYZE THE REQUEST**: What is the user trying to achieve at a high level?
2. **IDENTIFY SPECIFIC FILES**: Which exact files need to be modified based on the codebase structure?
3. **DETERMINE IMPLEMENTATION DETAILS**: What specific code changes are needed?
4. **CONSIDER DEPENDENCIES**: What imports, types, or utilities are needed?
5. **FRAMEWORK SPECIFICS**: How should this be implemented given the framework (NextJS, TypeScript, etc.)?

Generate an enhanced, technical version of the request that includes:
- Specific file paths that need modification
- Exact import statements that need to be added
- Specific component/function names to use
- Implementation approach (e.g., "add to JSX return statement", "create new component")
- Framework-specific details (e.g., "use 'use client' directive", "import from next/navigation")

EXAMPLE:
Original: "add a navbar"
Enhanced: "Add the existing Navigation component from '@/components/Navigation' to the app/layout.tsx file by importing it at the top and including <Navigation /> in the JSX return statement above the {{children}} element. Ensure the import statement uses the correct relative path and the component is rendered within the <body> tag."

Be specific about file paths, import statements, component names, and implementation details.

ENHANCED REQUEST:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a senior software engineer specializing in code analysis and technical specification writing."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            enhanced_request = response.choices[0].message.content.strip()
            
            # Clean up the response
            if enhanced_request.startswith("ENHANCED REQUEST:"):
                enhanced_request = enhanced_request.replace("ENHANCED REQUEST:", "").strip()
            
            return enhanced_request
            
        except Exception as e:
            print(f"❌ LLM enhancement failed: {e}")
            return self._fallback_enhance_request(user_request, codebase_analysis)
    
    def _fallback_enhance_request(self, 
                                user_request: str, 
                                codebase_analysis: Dict[str, Any]) -> str:
        """Fallback enhancement using rule-based approach."""
        
        enhanced_parts = []
        request_lower = user_request.lower()
        
        # Add framework context
        if codebase_analysis['framework_info']['is_nextjs']:
            if codebase_analysis['framework_info']['app_router']:
                enhanced_parts.append("In this NextJS 13+ App Router application")
            else:
                enhanced_parts.append("In this NextJS application")
        
        # Enhance common requests
        if any(word in request_lower for word in ['navbar', 'navigation', 'nav']):
            if 'components/Navigation.tsx' in codebase_analysis['components']:
                enhanced_parts.append("import and use the existing Navigation component from '@/components/Navigation'")
                enhanced_parts.append("Add it to app/layout.tsx in the JSX return statement")
            else:
                enhanced_parts.append("create a new Navigation component in components/Navigation.tsx")
                enhanced_parts.append("Import and use it in app/layout.tsx")
        
        elif any(word in request_lower for word in ['button', 'form', 'input']):
            if codebase_analysis['framework_info']['uses_shadcn']:
                enhanced_parts.append("using shadcn/ui components from '@/components/ui/'")
            enhanced_parts.append("with proper TypeScript types and event handlers")
        
        elif any(word in request_lower for word in ['page', 'route']):
            if codebase_analysis['framework_info']['app_router']:
                enhanced_parts.append("create a new page.tsx file in the app/ directory")
                enhanced_parts.append("using NextJS 13+ App Router conventions")
        
        # Add framework-specific details
        if codebase_analysis['framework_info']['uses_typescript']:
            enhanced_parts.append("with proper TypeScript types")
        
        if codebase_analysis['framework_info']['uses_tailwind']:
            enhanced_parts.append("styled with Tailwind CSS classes")
        
        # Construct enhanced request
        if enhanced_parts:
            enhanced_request = f"{user_request.capitalize()} - specifically: {', '.join(enhanced_parts)}."
        else:
            enhanced_request = f"{user_request} - implement this feature following the existing codebase patterns and architecture."
        
        return enhanced_request 

    def _llm_enhance_app_idea(self, app_idea: str) -> str:
        """Use LLM to enhance an app idea with detailed technical specifications."""
        try:
            prompt = f"""Enhance this app idea to be more technical and specific for a NextJS developer:

ORIGINAL IDEA: {app_idea}

Create a detailed technical specification that includes:
1. Core functionality and features
2. Suggested component structure
3. State management approach
4. UI/UX considerations
5. Data flow and architecture
6. Key technologies to use (shadcn/ui, Tailwind CSS, TypeScript)

Be specific about what components should be created and how they should interact.
Keep it practical and implementable.

ENHANCED SPECIFICATION:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert NextJS architect who creates detailed technical specifications from app ideas."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            enhanced_idea = response.choices[0].message.content.strip()
            return enhanced_idea
            
        except Exception as e:
            print(f"⚠️ LLM enhancement failed: {str(e)}")
            return self._fallback_enhance_app_idea(app_idea)

    def _fallback_enhance_app_idea(self, app_idea: str) -> str:
        """Fallback enhancement when LLM is not available."""
        idea_lower = app_idea.lower()
        
        # Common app patterns and their enhancements
        if any(word in idea_lower for word in ['todo', 'task', 'list']):
            return f"""NextJS Todo Application - {app_idea}

Technical Specification:
- Create a TodoList component with full CRUD operations
- Use TypeScript interfaces for Todo type definitions
- Implement local storage persistence with useLocalStorage hook
- Add filtering (all, active, completed) and sorting capabilities
- Use shadcn/ui components (Button, Input, Checkbox, Card)
- Implement responsive design with Tailwind CSS
- Add proper form validation and user feedback"""

        elif any(word in idea_lower for word in ['dashboard', 'analytics', 'chart']):
            return f"""NextJS Dashboard Application - {app_idea}

Technical Specification:
- Create modular dashboard components with data visualization
- Implement responsive grid layout for dashboard widgets
- Use TypeScript for type safety and data modeling
- Add state management for dashboard configuration
- Use shadcn/ui components for consistent UI
- Implement dark/light theme support
- Add data fetching with proper loading states"""

        elif any(word in idea_lower for word in ['weather', 'forecast']):
            return f"""NextJS Weather Application - {app_idea}

Technical Specification:
- Create WeatherCard and WeatherDetails components
- Implement geolocation and weather API integration
- Use TypeScript interfaces for weather data
- Add search functionality for different locations
- Use shadcn/ui components for modern interface
- Implement responsive design for mobile/desktop
- Add error handling and loading states"""

        else:
            return f"""NextJS Application - {app_idea}

Technical Specification:
- Design a component-based architecture using React functional components
- Implement TypeScript for type safety throughout the application
- Use shadcn/ui component library for consistent, accessible UI
- Apply Tailwind CSS for responsive styling and modern design
- Implement proper state management (useState/useReducer)
- Add local storage for data persistence where appropriate
- Follow NextJS best practices with proper file structure
- Ensure responsive design for all screen sizes"""