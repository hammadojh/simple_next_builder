#!/usr/bin/env python3
"""
Enhanced NextJS App Builder
A Python script that interfaces with multiple LLM providers to generate NextJS applications
with robust fallback handling and response validation.
"""

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import requests
import argparse
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    # dotenv not installed, continue without it
    pass

# Import the new indexing system
from .indexing import CodebaseIndexer
from .version_manager import VersionManager
from .llm_coordinator import LLMCoordinator

# Add to imports section
from .progress_loader import (
    show_progress, llm_progress, analysis_progress, 
    build_progress, file_progress, update_current_task,
    LoaderStyle
)


class MultiLLMAppBuilder:
    def __init__(self, openai_api_key=None, openrouter_api_key=None, anthropic_api_key=None):
        """Initialize the app builder with multiple LLM providers."""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        
        # Initialize OpenAI client if key is available
        self.openai_client = None
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            except ImportError:
                print("⚠️ OpenAI package not installed")
        
        # Initialize Anthropic client if key is available
        self.anthropic_client = None
        if self.anthropic_api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            except ImportError:
                print("⚠️ Anthropic package not installed")
        
        # Initialize the codebase indexer (will be set when needed)
        self._indexer: Optional[CodebaseIndexer] = None
        self._indexer_app_path: Optional[str] = None
        
        # Initialize version manager (will be set when needed)
        self._version_manager: Optional[VersionManager] = None
        self._version_manager_app_path: Optional[str] = None
        
        # Initialize LLM coordinator for intelligent planning and orchestration
        self.coordinator: Optional[LLMCoordinator] = None
        self.use_coordinator = True  # Flag to enable/disable coordinator mode
        
        print(f"🧠 LLM Coordinator mode: {'ENABLED' if self.use_coordinator else 'DISABLED'}")
        if self.use_coordinator:
            print("   📋 Will use intelligent planning and task coordination")
        else:
            print("   🔧 Will use traditional direct LLM approach")
        
        # Define LLM provider configurations
        self.llm_providers = [
            {
                "name": "Claude 4 Sonnet",
                "type": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4000,
                "temperature": 0.7
            },
            {
                "name": "Claude 3.7 Sonnet",
                "type": "openrouter",
                "model": "anthropic/claude-3.7-sonnet",
                "max_tokens": 4000,
                "temperature": 0.7
            },
            {
                "name": "GPT-4o",
                "type": "openai",
                "model": "gpt-4o",
                "max_tokens": 4000,
                "temperature": 0.7
            },
            {
                "name": "Claude-3.5 Sonnet",
                "type": "openrouter",
                "model": "anthropic/claude-3.5-sonnet",
                "max_tokens": 4000,
                "temperature": 0.7
            },
            {
                "name": "Gemini Pro",
                "type": "openrouter",
                "model": "google/gemini-pro-1.5",
                "max_tokens": 4000,
                "temperature": 0.7
            },
            {
                "name": "DeepSeek Coder",
                "type": "openrouter",
                "model": "deepseek/deepseek-coder",
                "max_tokens": 4000,
                "temperature": 0.7
            },
            {
                "name": "Qwen Coder",
                "type": "openrouter",
                "model": "qwen/qwen-2.5-coder-32b-instruct",
                "max_tokens": 4000,
                "temperature": 0.7
            },
            {
                "name": "Grok Beta",
                "type": "openrouter",
                "model": "x-ai/grok-beta",
                "max_tokens": 4000,
                "temperature": 0.7
            }
        ]
    
    def set_coordinator_mode(self, enabled: bool):
        """Enable or disable the LLM coordinator."""
        self.use_coordinator = enabled
        print(f"🧠 LLM Coordinator mode: {'ENABLED' if enabled else 'DISABLED'}")
        if enabled:
            print("   📋 Will use intelligent planning and task coordination")
        else:
            print("   🔧 Will use traditional direct LLM approach")
    
    def generate_edit_response(self, context_info: str, instructions: str) -> str:
        """Generate edit response for the coordinator system."""
        try:
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": f"CONTEXT:\n{context_info}\n\nINSTRUCTIONS:\n{instructions}\n\nGenerate the necessary edits to implement these changes."}
            ]
            
            return self.generate_with_fallback(messages, context="edit")
            
        except Exception as e:
            print(f"❌ Error generating edit response: {str(e)}")
            return None
    
    def get_system_prompt(self):
        """Get the system prompt that instructs the LLM to use the correct syntax."""
        return """You are an expert NextJS frontend developer. When given an app idea, you will create a complete, feature-rich FRONTEND-ONLY NextJS application using the specified syntax format.

🚨 CRITICAL JSX SYNTAX RULES - FAILURE TO FOLLOW WILL BREAK THE APP! 🚨

1. **RETURN STATEMENT SYNTAX**: ALWAYS end JSX return statements with closing parenthesis `)`, NEVER with curly brace `}`
   
   ✅ CORRECT JSX RETURN:
   ```javascript
   return (
     <div className="container">
       <h1>Title</h1>
       <p>Content</p>
     </div>
   )  // ← ENDS WITH CLOSING PARENTHESIS
   ```
   
   ❌ WRONG JSX RETURN:
   ```javascript
   return (
     <div className="container">
       <h1>Title</h1>
     </div>
   }  // ← WRONG! NEVER USE CURLY BRACE TO END RETURN
   ```

2. **FRONTEND-ONLY FOCUS**: 
   - NO authentication systems (no NextAuth, no login/logout)
   - NO database connections (no Prisma, no MongoDB, no SQL)
   - NO backend API routes (no pages/api/ unless absolutely necessary)
   - NO server-side functionality beyond static generation
   - Use React state, localStorage, or sessionStorage for data management
   - Focus on rich UI/UX and interactive frontend features

3. **DATA MANAGEMENT PATTERNS**:
   ✅ CORRECT DATA PATTERNS:
   ```javascript
   // Use React state for temporary data
   const [items, setItems] = useState([])
   
   // Use localStorage for persistence
   useEffect(() => {
     const saved = localStorage.getItem('items')
     if (saved) setItems(JSON.parse(saved))
   }, [])
   
   // Mock data for demonstrations
   const mockData = [
     { id: 1, title: "Sample Item", completed: false }
   ]
   ```

4. **COMPONENT STRUCTURE**: Use proper component architecture with:
   - Reusable components in /components directory
   - Pages in /app directory (App Router)
   - TypeScript interfaces in /types directory
   - Utility functions in /lib directory (if needed)

5. **STYLING**: Use shadcn/ui components with Tailwind CSS:
   ✅ CORRECT: Import and use shadcn components: `import { Button } from "@/components/ui/button"`
   ✅ CORRECT: Use shadcn semantic variants: `<Button variant="outline" size="lg">Click me</Button>`
   ✅ CORRECT: Combine with Tailwind: `<Button className="w-full mt-4">Submit</Button>`
   ❌ WRONG: Basic HTML elements for interactive components (use shadcn Button, Input, Card, etc.)

6. **RESPONSIVE DESIGN**: Always include responsive classes:
   `className="w-full md:w-1/2 lg:w-1/3"`

7. **INTERACTIVE FEATURES**: Focus on rich frontend interactivity:
   - Form handling with validation
   - Drag and drop (if relevant)
   - Animations and transitions
   - Modal dialogs and overlays
   - Search and filtering
   - Sorting and organizing
   - Local data persistence

8. **FILE CREATION SYNTAX**: Use the exact format:
   ```
   <new filename="path/to/file.tsx">
   // Content here
   </new>
   ```

9. **ERROR HANDLING**: Include proper error boundaries and loading states:
   ```javascript
   const [loading, setLoading] = useState(false)
   const [error, setError] = useState(null)
   ```

10. **ACCESSIBILITY**: Include proper ARIA labels and semantic HTML

11. **SHADCN/UI COMPONENTS**: ALWAYS use shadcn/ui components for modern, professional UI:
    📦 **ESSENTIAL COMPONENTS TO USE**:
    - `Button` (from "@/components/ui/button") - for all buttons
    - `Input` (from "@/components/ui/input") - for form inputs  
    - `Card, CardHeader, CardTitle, CardContent` (from "@/components/ui/card") - for containers
    - Use Lucide React icons: `import { Plus, Trash2, Edit } from "lucide-react"`
    
    ✅ **CORRECT SHADCN USAGE**:
    ```javascript
    import { Button } from "@/components/ui/button"
    import { Input } from "@/components/ui/input" 
    import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
    import { Plus, Search } from "lucide-react"
    
    <Card>
      <CardHeader>
        <CardTitle>Modern UI</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2">
          <Input placeholder="Search..." />
          <Button>
            <Search className="h-4 w-4 mr-2" />
            Search
          </Button>
        </div>
      </CardContent>
    </Card>
    ```
    
    ❌ **AVOID**: Plain HTML buttons, inputs, divs for interactive elements

🎯 **GOAL**: Create a production-ready, visually appealing, fully functional frontend application that users can immediately interact with and share. Focus on user experience and rich interactions with modern shadcn/ui components, not backend complexity.

Remember: FRONTEND ONLY - no auth, no database, no backend APIs. Rich UI/UX with shadcn/ui components and local state management."""

    def get_user_prompt(self, app_idea):
        """Create the user prompt with the app idea."""
        return """Build a NextJS application for the following idea:

""" + app_idea + """

IMPORTANT: All configuration files are already set up and working perfectly:
- ✅ NextJS 14.0.0 with app router
- ✅ TypeScript configured
- ✅ Tailwind CSS v3.4.0 configured and compiling
- ✅ PostCSS and autoprefixer configured
- ✅ Basic layout and navigation components provided

YOUR TASK: Focus ONLY on creating the application features and logic!

CRITICAL FOR INITIAL APP CREATION:
- **ALWAYS use <new filename="app/page.tsx"> to create the main application page**
- This completely replaces the template page with your actual app
- DO NOT use <edit> tags for the main page during initial creation

COMPLEXITY GUIDELINES:
- If the request mentions "simple" or is for basic tools (todo, calculator, counter, etc.): Create a single-page application
- If the request is for complex systems (e-commerce, social media, multi-feature apps): Create multi-page applications

REQUIRED ACTIONS:
1. **ALWAYS create app/page.tsx using <new> tag with the main application functionality**
2. Create supporting components as needed using <new> tag
3. Add additional pages only if the request clearly needs multiple views

DO NOT CREATE OR MODIFY:
- package.json, tailwind.config.js, postcss.config.js, next.config.mjs, tsconfig.json
- app/globals.css, app/layout.tsx (unless adding specific metadata)

CREATE AS NEEDED:
- app/page.tsx (REQUIRED - use <new> tag to replace template with actual app)
- Additional pages (app/*/page.tsx) only for complex applications
- Custom components (components/*.tsx)
- TypeScript interfaces (types/*.ts)
- Utility functions (utils/*.ts)
- Mock data files (data/*.ts)

FOR SIMPLE APPLICATIONS:
Focus on a single-page experience with:
- Clean, focused UI
- Essential functionality only
- Minimal navigation (if any)
- Direct interaction on the main page

FOR COMPLEX APPLICATIONS:
Create a multi-page experience with:
- Navigation between different sections
- Multiple feature areas
- User authentication flows
- Data management across pages

SPECIFIC FEATURES BY APP TYPE:

If it's a SIMPLE TODO/TASK app:
- Single page with task input, list, and management
- Local state management
- Add, remove, complete functionality
- No complex routing needed

If it's a SIMPLE CALCULATOR/TOOL:
- Single page with the tool interface
- State management for calculations/data
- Clear, intuitive controls
- Results display

If it's a DELIVERY/FOOD app:
- Restaurant/menu selection page (app/menu/page.tsx)
- Individual food item details (app/menu/[id]/page.tsx)
- Shopping cart functionality (app/cart/page.tsx)
- Checkout process (app/checkout/page.tsx)
- User profile/account page (app/profile/page.tsx)

If it's an E-COMMERCE app:
- Product catalog with categories (app/products/page.tsx)
- Individual product pages (app/products/[id]/page.tsx)  
- Shopping cart (app/cart/page.tsx)
- User account (app/account/page.tsx)
- Checkout process (app/checkout/page.tsx)

If it's a SOCIAL MEDIA app:
- User feed/timeline (app/feed/page.tsx)
- User profiles (app/profile/[id]/page.tsx)
- Create post page (app/create/page.tsx)
- Post details with comments (app/posts/[id]/page.tsx)

ALWAYS INCLUDE:
- Proper TypeScript interfaces
- Realistic mock data when needed
- Error handling and loading states
- Responsive design with Tailwind classes
- **shadcn/ui components for modern, professional UI**

🎨 **REQUIRED UI COMPONENTS**: Use shadcn/ui for ALL interactive elements:
- **Buttons**: `import { Button } from "@/components/ui/button"` with variants (default, outline, secondary, etc.)
- **Inputs**: `import { Input } from "@/components/ui/input"` for all form fields
- **Cards**: `import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"` for containers
- **Icons**: `import { IconName } from "lucide-react"` for all icons (Plus, Trash2, Edit, Search, etc.)

✅ **MODERN UI EXAMPLE**:
```
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input" 
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Plus, Trash2 } from "lucide-react"

<Card className="w-full max-w-md">
  <CardHeader>
    <CardTitle>Todo App</CardTitle>
  </CardHeader>
  <CardContent className="space-y-4">
    <div className="flex gap-2">
      <Input placeholder="Add new todo..." />
      <Button size="icon">
        <Plus className="h-4 w-4" />
      </Button>
    </div>
  </CardContent>
</Card>
```

**REMEMBER: Use <new filename="app/page.tsx"> for the main page, not <edit> tags!**

Return the complete application using <new> tags for initial creation.
Focus on delivering exactly what was requested - simple for simple requests, complex for complex requests!"""

    def validate_response(self, content: str, context: str = "edit") -> Tuple[bool, str]:
        """
        Enhanced validation of AI responses with structural code analysis.
        
        Returns:
            Tuple of (is_valid, detailed_feedback)
        """
        import re
        issues = []
        
        # Check for required syntax tags based on context
        if context == "create":
            # For app creation, expect <new> tags
            new_tags = re.findall(r'<new filename=\"([^\"]+)\">', content)
            if not new_tags:
                issues.append("No <new> tags found in response for app creation")
        elif context == "single_file":
            # For single file generation, just check that it contains code-like content
            if not any(keyword in content for keyword in ['import', 'export', 'const', 'function', 'interface', 'type', "'use client'", '"use client"']):
                issues.append("Response doesn't appear to contain valid code content")
        elif context == "file_plan":
            # For file plan generation, expect JSON with files array
            try:
                import json
                # Try to parse as JSON
                if content.strip().startswith('```json'):
                    # Extract JSON from markdown code block
                    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = content.strip()
                else:
                    json_str = content.strip()
                
                data = json.loads(json_str)
                if 'files' not in data:
                    issues.append("No 'files' array found in file plan JSON")
                elif not isinstance(data['files'], list) or len(data['files']) == 0:
                    issues.append("'files' array is empty or invalid in file plan")
            except json.JSONDecodeError:
                issues.append("File plan response is not valid JSON format")
        elif context == "edit_fallback":
            # Legacy fallback now also uses unified diff format
            if '*** Begin Patch' not in content:
                issues.append("No '*** Begin Patch' sentinel found - unified diff format required")
            if '*** End Patch' not in content:
                issues.append("No '*** End Patch' sentinel found - unified diff format required")
        elif context == "intent":
            # For intent-based editing, expect JSON with intents array
            try:
                # Extract JSON from response 
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = content.strip()
                
                data = json.loads(json_str)
                if 'intents' not in data:
                    issues.append("No 'intents' array found in JSON response")
                elif not isinstance(data['intents'], list) or len(data['intents']) == 0:
                    issues.append("'intents' array is empty or invalid")
            except json.JSONDecodeError:
                issues.append("Response is not valid JSON format")
            except Exception as e:
                issues.append(f"Error validating intent JSON: {str(e)}")
        else:
            # For regular editing, expect unified diff format
            if '*** Begin Patch' not in content:
                issues.append("No '*** Begin Patch' sentinel found - unified diff format required for edits")
            if '*** End Patch' not in content:
                issues.append("No '*** End Patch' sentinel found - unified diff format required for edits")
                
            # CRITICAL: Validate code structure in diff
            if context == "edit":
                structural_issues = self._validate_diff_code_structure(content)
                issues.extend(structural_issues)
        
        # Show positive feedback for good practices
        good_practices = []
        
        if context == "edit" and '*** Begin Patch' in content and '*** Update File:' in content:
            good_practices.append("Uses proper unified diff format")
        elif context == "create" and '<new filename=' in content:
            good_practices.append("Uses appropriate file creation tags")
        elif context == "single_file" and any(keyword in content for keyword in ['import', 'export', 'const', 'function']):
            good_practices.append("Contains valid code structure")
        elif context == "file_plan" and '"files"' in content:
            good_practices.append("Uses structured file plan format")
        elif context == "edit_fallback" and '*** Begin Patch' in content:
            good_practices.append("Uses proper unified diff format (fallback)")
        elif context == "intent" and '"intents"' in content:
            good_practices.append("Uses structured intent-based format")
            
        if 'className=' in content and not re.search(r'\bcustom-\w+', content):
            good_practices.append("Uses standard Tailwind classes")
        
        if good_practices:
            print("📋 Good practices detected:")
            for practice in good_practices:
                print(f"   ✓ {practice}")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            print("❌ Response validation failed:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print("✅ Response validation passed!")
        
        # Return detailed feedback
        feedback = "\n".join(issues) if issues else "Response validation passed"
        return is_valid, feedback
        
    def _validate_diff_code_structure(self, diff_content: str) -> List[str]:
        """
        Validate code structure in diff to catch common errors.
        
        Returns:
            List of structural issues found
        """
        issues = []
        
        # Check for variable reference problems
        variable_refs = re.findall(r'\+.*?(\w+)\.map\(', diff_content)
        duplicate_interfaces = re.findall(r'\+\\s*interface\\s+(\w+)', diff_content)
        duplicate_functions = re.findall(r'\+\\s*(?:function\\s+(\w+)|export\\s+default\\s+function\\s+(\w+))', diff_content)
        
        # Check for common patterns that indicate problems
        if any('mockPosts.map' in line for line in diff_content.split('\n') if line.startswith('+')):
            issues.append("CRITICAL: Uses 'mockPosts.map' but should use state variable (likely 'posts')")
            
        if any('+interface ' in line and 'interface ' in line for line in diff_content.split('\n')):
            # Check if interface is being added when it might already exist
            issues.append("WARNING: Adding interface - ensure it doesn't duplicate existing ones")
            
        # Check for malformed JSX in additions
        added_lines = [line[1:] for line in diff_content.split('\n') if line.startswith('+')]
        jsx_content = '\n'.join(added_lines)
        
        if '<' in jsx_content and '>' in jsx_content:
            # More intelligent JSX validation
            # Count actual JSX tags (not operators like < or >)
            jsx_tags = re.findall(r'<\s*[a-zA-Z][^>]*>', jsx_content)
            self_closing_tags = re.findall(r'<\s*[a-zA-Z][^>]*\/>', jsx_content)
            closing_tags = re.findall(r'<\s*\/[a-zA-Z][^>]*>', jsx_content)
            
            # Self-closing tags are balanced, so don't count them
            open_count = len(jsx_tags) - len(self_closing_tags)
            close_count = len(closing_tags)
            
            # Only flag if there's a significant imbalance (more than 5 difference)
            # This allows for partial diffs and context lines
            if abs(open_count - close_count) > 5:
                issues.append("WARNING: Potential JSX structure issues in diff")
        
        return issues



    def call_openai(self, messages: List[Dict], config: Dict) -> Optional[str]:
        """Call OpenAI API with real-time streaming."""
        if not self.openai_client:
            return None
        
        try:
            print(f"\n🤖 {config['name']} Response:")
            
            # Use streaming API
            response_stream = self.openai_client.chat.completions.create(
                model=config["model"],
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                stream=True  # Enable streaming
            )
            
            full_response = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Stream each token as it arrives
                    print(content, end="", flush=True)
            
            print("\n")
            return full_response
            
        except Exception as e:
            print(f"❌ OpenAI streaming error: {str(e)}")
            return None

    def call_anthropic(self, messages: List[Dict], config: Dict) -> Optional[str]:
        """Call Anthropic API with real-time streaming."""
        if not self.anthropic_client:
            return None
        
        try:
            # Convert messages to Anthropic format
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            print(f"\n🤖 {config['name']} Response:")
            
            # Use streaming API
            full_response = ""
            with self.anthropic_client.messages.stream(
                model=config["model"],
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                system=system_message,
                messages=user_messages
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    # Stream each token as it arrives
                    print(text, end="", flush=True)
            
            print("\n")
            return full_response
            
        except Exception as e:
            print(f"❌ Anthropic streaming error: {str(e)}")
            return None

    def call_openrouter(self, messages: List[Dict], config: Dict) -> Optional[str]:
        """Call OpenRouter API."""
        if not self.openrouter_api_key:
            return None
        
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/nextjs-master-builder",
                "X-Title": "NextJS Master Builder"
            }
            
            payload = {
                "model": config["model"],
                "messages": messages,
                "max_tokens": config["max_tokens"],
                "temperature": config["temperature"]
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"❌ OpenRouter error: {str(e)}")
            return None

    def generate_with_fallback(self, messages: List[Dict], context: str = "create") -> Optional[str]:
        """Generate response with fallback across multiple LLM providers."""
        # Stop all progress loaders before streaming
        from .progress_loader import progress_manager
        progress_manager.cleanup_all()
        
        print("🚀 Generating NextJS application with multi-LLM fallback...")
        
        for i, config in enumerate(self.llm_providers, 1):
            try:
                print(f"🤖 Trying {config['name']} ({i}/{len(self.llm_providers)})...")
                
                # Call the appropriate provider
                if config["type"] == "openai" and self.openai_client:
                    response = self.call_openai(messages, config)
                elif config["type"] == "anthropic" and self.anthropic_client:
                    response = self.call_anthropic(messages, config)
                elif config["type"] == "openrouter" and self.openrouter_api_key:
                    response = self.call_openrouter(messages, config)
                else:
                    continue
                
                if response:
                    # 🔍 DEBUG: Save and print the raw AI response
                    import os
                    import time
                    
                    # Save to debug file
                    debug_dir = "debug_responses"
                    os.makedirs(debug_dir, exist_ok=True)
                    timestamp = int(time.time())
                    debug_file = f"{debug_dir}/ai_response_{config['name'].replace(' ', '_')}_{timestamp}.txt"
                    
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(f"=== RAW AI RESPONSE FROM {config['name']} ===\n")
                        f.write(f"Provider: {config['name']}\n")
                        f.write(f"Model: {config['model']}\n")
                        f.write(f"Context: {context}\n")
                        f.write("=" * 50 + "\n")
                        f.write(response)
                        f.write("\n" + "=" * 50 + "\n")
                        f.write("=== END RESPONSE ===\n")
                    
                    print(f"🔍 DEBUG: Raw AI response saved to {debug_file}")
                    print("🔍 DEBUG: First 800 characters of AI response:")
                    print("-" * 60)
                    print(response[:800])
                    print("-" * 60)
                    if len(response) > 800:
                        print(f"... (truncated, full response in {debug_file})")
                    
                    # Check for JSX issues in the raw response
                    if 'return (' in response:
                        jsx_samples = []
                        lines = response.split('\n')
                        for i, line in enumerate(lines):
                            if 'return (' in line:
                                # Get this line and next few lines to see the pattern
                                start_idx = max(0, i-2)  # 2 lines before
                                end_idx = min(len(lines), i+8)  # 6 lines after
                                sample = '\n'.join(lines[start_idx:end_idx])
                                jsx_samples.append(f"Found at line {i+1}:\n{sample}")
                        
                        if jsx_samples:
                            print(f"🔍 FOUND {len(jsx_samples)} JSX RETURN STATEMENTS IN RAW RESPONSE:")
                            for j, sample in enumerate(jsx_samples[:2]):  # Show first 2 samples
                                print(f"Sample {j+1}:")
                                print(sample)
                                print("-" * 40)
                    
                    # Validate the response
                    is_valid, validation_feedback = self.validate_response(response, context)
                    
                    if is_valid:
                        print(f"✅ {config['name']} provided valid response!")
                        return response
                    else:
                        print(f"❌ {config['name']} response validation failed:")
                        print(f"   - {validation_feedback}")
                        print("⏭️ Trying next provider...")
                else:
                    print(f"⚠️ {config['name']} returned empty response")
                    
            except Exception as e:
                print(f"❌ Error with {config['name']}: {str(e)}")
                continue
        
        print("❌ All LLM providers failed to generate valid response")
        return None

    def clean_generated_content(self, content):
        """Clean the generated content by removing markdown code blocks."""
        if not content:
            return content
        
        # Remove markdown code blocks like ```tsx, ```javascript, ```typescript, etc.
        pattern = r'^```[a-zA-Z]*\n(.*?)\n```$'
        cleaned = re.sub(pattern, r'\1', content, flags=re.MULTILINE | re.DOTALL)
        
        # Also handle inline code blocks within our tags
        cleaned = re.sub(r'```[a-zA-Z]*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```', '', cleaned)
        
        # Clean up any extra whitespace that might be left
        cleaned = re.sub(r'\n\n\n+', '\n\n', cleaned)
        
        return cleaned.strip()

    def generate_app(self, app_idea):
        """Generate the NextJS app using either the coordinator or traditional approach."""
        try:
            # Check if coordinator should be used
            if self.use_coordinator:
                return self._generate_app_with_coordinator(app_idea)
            else:
                return self._generate_app_traditional(app_idea)
            
        except Exception as e:
            print(f"❌ Error generating app: {str(e)}")
            return None
    
    def _generate_app_with_coordinator(self, app_idea):
        """Generate app using the intelligent LLM coordinator."""
        print("🧠 Generating NextJS application with LLM Coordinator...")
        
        # CRITICAL: We need to set app context before using coordinator for new app creation
        # For new apps, we'll use a temporary approach and later set the actual app name
        # This will be properly set by the master builder after template creation
        
        # Initialize coordinator if not already done
        if not self.coordinator:
            self.coordinator = LLMCoordinator(app_builder=self)
        
        # For now, fall back to traditional approach since coordinator needs app context
        # The coordinator works better for editing existing apps
        print("🔄 Coordinator needs app context - using traditional approach for app creation...")
        print("💡 Coordinator will be used for editing this app later!")
        return self._generate_app_traditional(app_idea)
    
    def _generate_app_traditional(self, app_idea):
        """Generate app using the traditional direct LLM approach."""
        print("🚀 Generating NextJS application with traditional multi-LLM fallback...")
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.get_user_prompt(app_idea)}
        ]
        
        return self.generate_with_fallback(messages, context="create")

    def get_next_version(self, base_filename="input"):
        """Get the next version number for the output file."""
        version = 1
        while os.path.exists(f"{base_filename}_v{version}.txt"):
            version += 1
        return version

    def save_to_file(self, content, app_idea):
        """Save the generated content to a versioned file."""
        version = self.get_next_version()
        filename = f"input_v{version}.txt"
        
        # Add metadata header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""<!-- 
Generated NextJS Application
App Idea: {app_idea}
Generated: {timestamp}
Version: {version}
Multi-LLM Builder with validation
-->

"""
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(header + content)
            
            print(f"✅ Application saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving file: {str(e)}")
            return None

    def build_app(self, app_idea):
        """Main method to build the app from idea to file."""
        print(f"🚀 Building NextJS app for: {app_idea}")
        print("-" * 50)
        
        # Generate the app
        generated_content = self.generate_app(app_idea)
        
        if not generated_content:
            print("❌ Failed to generate application with all providers")
            return None
        
        print("✅ Application generated successfully!")
        
        # Save to file
        filename = self.save_to_file(generated_content, app_idea)
        
        if filename:
            print(f"📁 File saved as: {filename}")
            print(f"📊 Content length: {len(generated_content)} characters")
            return filename
        
        return None

    def build_and_run(self, auto_install_deps: bool = True) -> bool:
        """
        Build the app to verify changes work, with enhanced TypeScript error detection.
        
        Args:
            auto_install_deps: Whether to automatically install dependencies
            
        Returns:
            True if build succeeds, False otherwise
        """
        import subprocess
        import os
        import re
        
        app_path = self.get_app_path()
        
        with build_progress("NextJS application"):
            try:
                # Change to app directory
                original_dir = os.getcwd()
                os.chdir(app_path)
                
                # Install dependencies if requested
                if auto_install_deps:
                    update_current_task("installing dependencies")
                    print("📦 Installing dependencies...")
                    result = subprocess.run(['npm', 'install'], 
                                          capture_output=True, text=True, timeout=120)
                    if result.returncode != 0:
                        print(f"⚠️ npm install warnings: {result.stderr}")
                
                # Try to build the app with enhanced error capture
                update_current_task("building NextJS app")
                print("🔨 Building NextJS app...")
                result = subprocess.run(['npm', 'run', 'build'], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("✅ Build successful!")
                    return True
                else:
                    print("❌ Build failed:")
                    
                    # Enhanced error parsing for TypeScript issues
                    error_output = result.stdout + result.stderr
                    self._parse_and_display_typescript_errors(error_output)
                    return False
                    
            except subprocess.TimeoutExpired:
                print("⏰ Build timed out")
                return False
            except FileNotFoundError:
                print("❌ npm not found - please install Node.js")
                return False
            except Exception as e:
                print(f"❌ Build error: {str(e)}")
                return False
            finally:
                try:
                    os.chdir(original_dir)
                except:
                    pass
    
    def _parse_and_display_typescript_errors(self, error_output: str):
        """Parse and display TypeScript errors in a structured way."""
        import re
        
        print("📋 TypeScript Error Analysis:")
        print("=" * 50)
        
        # Parse TypeScript errors
        ts_error_pattern = r'\./(.*?):(\d+):(\d+)\s*\n\s*Type error:\s*(.*?)(?=\n\n|\nCompiled|\n\s*$)'
        ts_errors = re.findall(ts_error_pattern, error_output, re.DOTALL)
        
        if ts_errors:
            print(f"🚨 Found {len(ts_errors)} TypeScript error(s):")
            for i, (file_path, line, col, error_msg) in enumerate(ts_errors, 1):
                print(f"   {i}. {file_path}:{line}:{col}")
                print(f"      📝 {error_msg.strip()}")
                
                # Provide specific guidance for common errors
                if "does not exist on type 'never'" in error_msg:
                    print(f"      💡 Tip: Check if interfaces/types are properly defined and imported")
                elif "interface" in error_msg.lower():
                    print(f"      💡 Tip: Ensure interfaces are defined outside function components")
                elif "useState" in error_msg:
                    print(f"      💡 Tip: Check useState type annotations")
                print()
        
        # Parse syntax errors
        syntax_error_pattern = r'Expected.*(?:;|,|\}|\)|>)'
        syntax_errors = re.findall(syntax_error_pattern, error_output)
        
        if syntax_errors:
            print(f"🔍 Found {len(syntax_errors)} syntax error(s):")
            for i, error in enumerate(syntax_errors, 1):
                print(f"   {i}. {error}")
                
        # Parse specific structural issues
        if "interface" in error_output and "function" in error_output:
            print("🏗️  Structural Issue Detected:")
            print("   💡 Interface may be defined inside a function component")
            print("   💡 Move interface definitions to the top level of the file")
            
        print("=" * 50)

    def get_edit_prompt(self, app_idea, app_structure):
        """Create a prompt for editing an existing app using unified diff format."""
        return f"""You are editing an existing NextJS application. Here is the current app structure and content:

{app_structure}

User wants to make the following changes:
{app_idea}

INSTRUCTIONS:
Think step by step through this edit:

1. **UNDERSTAND THE REQUEST**: What specific changes need to be made?
2. **IDENTIFY TARGET FILES**: Which files need to be modified based on the request?
3. **PLAN THE CHANGES**: What imports, components, or logic need to be added/modified?
4. **CONSIDER DEPENDENCIES**: What other files might be affected?
5. **IMPLEMENT PRECISELY**: Generate the exact diff needed with proper context.

🚨 CRITICAL DIFF GENERATION RULES 🚨

You MUST respond with a UNIFIED DIFF in this EXACT format:

*** Begin Patch
*** Update File: path/to/file.tsx
@@ -old_start,old_count +new_start,new_count @@ optional_context
- old line to remove
+ new line to add
 unchanged context line
*** End Patch

🔗 CRITICAL: NEXTJS 13+ LINK COMPONENT USAGE:
⚠️  NEVER nest <a> tags inside <Link> components - this causes runtime errors!

✅ CORRECT: <Link href="/path" className="styles">Text</Link>
❌ WRONG: <Link href="/path"><a className="styles">Text</a></Link>

KEY REQUIREMENTS:
1. Use ONLY the unified diff format shown above
2. Include 2-3 lines of context before and after changes for accurate matching
3. Use proper +/- prefixes for added/removed lines
4. Use space prefix for unchanged context lines
5. Multiple files = multiple "*** Update File:" sections in ONE patch
6. NO line-based <edit> tags - ONLY unified diffs
7. Context lines must EXACTLY match existing file content

🎯 CRITICAL CODE ANALYSIS REQUIREMENTS:
- **USE EXISTING VARIABLES**: Reference only variables/functions shown in "Variables:" and "State Variables:" sections
- **NO DUPLICATE DECLARATIONS**: Don't redeclare interfaces or components that already exist
- **MAINTAIN STRUCTURE**: Keep existing component structure and just add/modify what's needed
- **VARIABLE CONSISTENCY**: If you see `const [posts, setPosts] = useState`, use `posts.map()` NOT `mockPosts.map()`

EXAMPLE:
*** Begin Patch
*** Update File: app/page.tsx
@@ -45,7 +45,8 @@ export default function HomePage() {{
   const [isLoading, setIsLoading] = useState(false);
   
   return (
-    <div className="container mx-auto p-8">
+    <div className="container mx-auto p-8 bg-gray-50">
+      <header className="mb-8">
      <h1 className="text-3xl font-bold mb-6">Welcome</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
*** End Patch

This diff-based approach ensures:
✅ Context-aware matching (no line number issues)
✅ Fuzzy matching handles small file changes
✅ Transactional safety with automatic rollback
✅ No overlapping edit corruption
✅ Standard format used by Git and professional tools

Generate a unified diff that implements the requested changes with proper context matching and variable consistency."""

    def _ensure_indexer(self, app_directory: str):
        """Ensure indexer is initialized for the given app directory."""
        if self._indexer is None or self._indexer_app_path != app_directory:
            print(f"🏗️ Initializing semantic indexer for: {app_directory}")
            self._indexer = CodebaseIndexer(app_directory)
            self._indexer_app_path = app_directory
            
            # Check if we need to build initial index
            stats = self._indexer.get_stats()
            if stats['storage']['total_chunks'] == 0:
                print("🔄 Building initial semantic index...")
                self._indexer.index_project()
            else:
                # Do a smart update to catch any changes
                if hasattr(self._indexer, 'smart_update'):
                    self._indexer.smart_update()
                print(f"📚 Using existing index with {stats['storage']['total_chunks']} chunks")
    
    def _ensure_version_manager(self, app_directory: str):
        """Ensure version manager is initialized for the given app directory."""
        if self._version_manager is None or self._version_manager_app_path != app_directory:
            print(f"🗂️ Initializing version manager for: {app_directory}")
            self._version_manager = VersionManager(app_directory)
            self._version_manager_app_path = app_directory

    def analyze_app_structure_enhanced(self, app_directory: str) -> str:
        """
        ENHANCED SEMANTIC CONTEXT ANALYSIS - Replaces manual file analysis with AI-powered indexing.
        
        This method now uses vector indexing and semantic search instead of manual file reading,
        providing more relevant and focused context for LLM prompts.
        
        Args:
            app_directory: Path to the app directory
            
        Returns:
            Semantically relevant context based on the current editing request
        """
        from pathlib import Path
        
        app_path = Path(app_directory)
        
        if not app_path.exists():
            raise FileNotFoundError(f"App directory does not exist: {app_directory}")
        
        # Initialize the semantic indexer
        self._ensure_indexer(app_directory)
        
        # For now, return a general context since we don't have the specific request
        # This will be enhanced when we have the actual user request
        context_info = []
        context_info.append(f"🏗️ SEMANTIC CODEBASE ANALYSIS")
        context_info.append("=" * 60)
        context_info.append(f"📁 Project: {app_directory}")
        
        # Get index statistics
        stats = self._indexer.get_stats()
        context_info.append(f"📊 Indexed: {stats['storage']['total_chunks']} code chunks")
        context_info.append(f"📈 File types: {dict(stats['storage']['file_types'])}")
        context_info.append(f"🧩 Chunk types: {dict(stats['storage']['chunk_types'])}")
        context_info.append("=" * 60)
        
        # Get a general overview of the app structure
        overview_chunks = self._indexer.search_code("NextJS app structure layout components", max_results=15)
        
        if overview_chunks:
            context_info.append("🔍 KEY APP COMPONENTS:")
            context_info.append("-" * 40)
            
            for i, chunk in enumerate(overview_chunks[:10], 1):
                context_info.append(f"{i}. {chunk['file_path']} ({chunk['chunk_type']})")
                if chunk['metadata']['component_name']:
                    context_info.append(f"   Component: {chunk['metadata']['component_name']}")
                elif chunk['metadata']['function_name']:
                    context_info.append(f"   Function: {chunk['metadata']['function_name']}")
        
        context_info.append("\n" + "=" * 60)
        context_info.append("🚀 SEMANTIC INDEXING ACTIVE - Context will be provided based on specific requests")
        context_info.append("💡 This replaces manual file analysis with AI-powered relevant context retrieval")
        context_info.append("=" * 60)
        
        return '\n'.join(context_info)
    
    def get_semantic_context_for_request(self, user_request: str, app_directory: str, 
                                       current_file: str = None, recent_files: List[str] = None) -> str:
        """
        Get context using intelligent context selection to prevent token overflow.
        
        🧠 ENHANCED: Uses AI-powered context selection to choose only relevant files.
        Prevents the 227K token overflow that was causing LLM failures.
        """
        try:
            # Import the intelligent context selector and request enhancer
            from .context_selector import IntelligentContextSelector
            from .request_enhancer import RequestEnhancer
            
            # Initialize request enhancer
            request_enhancer = RequestEnhancer()
            
            # Enhance the user request to be more technical and specific
            enhanced_request = request_enhancer.enhance_edit_request(
                user_request=user_request,
                app_path=app_directory
            )
            
            print(f"🎯 Original request: {user_request}")
            print(f"✨ Enhanced request: {enhanced_request}")
            
            # Initialize context selector with increased token limit
            context_selector = IntelligentContextSelector(max_tokens=150000)  # More generous limit
            
            # Select optimal context for this request using enhanced request
            context_selection = context_selector.select_context(
                request=enhanced_request,
                app_path=app_directory,
                operation_type="edit"
            )
            
            # Format for LLM consumption
            formatted_context = context_selector.format_context_for_llm(context_selection)
            
            print(f"✅ Intelligent context selection completed")
            print(f"📊 Selected {len(context_selection.selected_files)} files ({context_selection.total_tokens:,} tokens)")
            
            return formatted_context
            
        except Exception as e:
            print(f"❌ Intelligent context selection failed: {e}")
            print("🔄 Falling back to basic context selection...")
            
            # Fallback to minimal context if intelligent selection fails
            return self._get_basic_context_fallback(enhanced_request if 'enhanced_request' in locals() else user_request, app_directory)
    
    def _get_basic_context_fallback(self, user_request: str, app_directory: str) -> str:
        """
        Basic context fallback when intelligent context selection fails.
        
        Provides minimal but essential context to prevent complete failure.
        """
        from pathlib import Path
        
        context_parts = [
            "📁 BASIC CONTEXT (Fallback Mode)",
            "=" * 50,
            f"User Request: {user_request}",
            f"App Directory: {app_directory}",
            "=" * 50
        ]
        
        app_path = Path(app_directory)
        essential_files = ["package.json", "app/page.tsx", "app/layout.tsx"]
        
        for file_path in essential_files:
            full_path = app_path / file_path
            if full_path.exists():
                try:
                    content = full_path.read_text()
                    # Only include first 20 lines to keep context minimal
                    lines = content.split('\n')[:20]
                    context_parts.extend([
                        f"\n📄 FILE: {file_path} (first 20 lines)",
                        "```",
                        '\n'.join(lines),
                        "```"
                    ])
                except Exception as e:
                    context_parts.append(f"❌ Error reading {file_path}: {str(e)}")
        
        final_context = "\n".join(context_parts)
        estimated_tokens = len(final_context) // 4
        context_parts.append(f"\n📊 Fallback context size: {len(final_context):,} characters (~{estimated_tokens:,} tokens)")
        
        print(f"🔄 Using basic fallback context ({estimated_tokens:,} tokens)")
        return "\n".join(context_parts)
    
    def _get_relevant_excerpt(self, content: str, user_request: str, max_lines: int = 20) -> str:
        """
        Extract relevant excerpt from file content based on user request.
        
        Args:
            content: Full file content
            user_request: User's request to find relevant parts
            max_lines: Maximum lines to include
            
        Returns:
            Relevant excerpt of the file
        """
        lines = content.split('\n')
        
        # If file is small enough, return first portion
        if len(lines) <= max_lines:
            return content
        
        # Look for keywords from user request in the code
        request_keywords = user_request.lower().split()
        relevant_lines = []
        
        # Find lines that contain request keywords or are structurally important
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Always include imports, exports, and function declarations
            if any(keyword in line for keyword in ['import ', 'export ', 'function ', 'const ', 'interface ', 'type ']):
                relevant_lines.append((i, line))
            # Include lines that match user request keywords
            elif any(keyword in line_lower for keyword in request_keywords if len(keyword) > 2):
                relevant_lines.append((i, line))
        
        # If we found relevant lines, return them with context
        if relevant_lines:
            # Sort by line number and take first max_lines
            relevant_lines.sort()
            selected_lines = relevant_lines[:max_lines]
            
            # Add some context around important lines
            result_lines = []
            for line_num, line_content in selected_lines:
                # Add a bit of context (line numbers for reference)
                result_lines.append(f"{line_num + 1:3}: {line_content}")
            
            return '\n'.join(result_lines)
        
        # Fallback: return first max_lines of the file
        excerpt_lines = []
        for i in range(min(max_lines, len(lines))):
            excerpt_lines.append(f"{i + 1:3}: {lines[i]}")
        
        if len(lines) > max_lines:
            excerpt_lines.append(f"... ({len(lines) - max_lines} more lines)")
        
        return '\n'.join(excerpt_lines)

    def analyze_file_with_line_count(self, file_path: str, relative_path: str) -> str:
        """
        Analyze a file and return its content with line numbers and total count.
        
        Args:
            file_path: Full path to the file
            relative_path: Relative path for display
            
        Returns:
            Formatted string with line numbers and total count
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Create formatted content with line numbers
            content_lines = []
            content_lines.append(f"\n📄 FILE: {relative_path} ({len(lines)} lines)")
            content_lines.append("-" * 50)
            
            for i, line in enumerate(lines, 1):
                # Remove trailing newline for display but keep track of actual content
                display_line = line.rstrip('\n')
                content_lines.append(f"{i:3d}: {display_line}")
            
            content_lines.append(f"\n[END OF FILE - Total: {len(lines)} lines]")
            
            return '\n'.join(content_lines)
            
        except Exception as e:
            return f"\n📄 FILE: {relative_path} (ERROR reading file: {e})"
    
    def get_app_path(self):
        """Get the full path to the current app directory."""
        if not hasattr(self, 'app_name') or not self.app_name:
            raise ValueError("App name not set")
        if not hasattr(self, 'apps_dir') or not self.apps_dir:
            raise ValueError("Apps directory not set")
        return str(self.apps_dir / self.app_name)
    
    def edit_app(self, app_idea, use_intent_based: bool = False):
        """
        Edit an existing NextJS app with automatic rollback on failure.
        
        Args:
            app_idea: Description of changes to make
            use_intent_based: If True, use intent-based editing (more robust)
                             If False, use traditional diff-based editing
        
        This method:
        1. Creates automatic snapshot before editing
        2. Tries multiple editing strategies with rollback on failure
        3. Verifies build success after each attempt
        4. Automatically recovers from failures
        5. Guarantees the app remains in a working state
        """
        # Check if coordinator should be used for editing
        if self.use_coordinator:
            return self._edit_app_with_coordinator(app_idea)
        else:
            return self._edit_app_with_rollback(app_idea, use_intent_based)
    
    def _edit_app_with_coordinator(self, app_idea):
        """Edit app using the intelligent LLM coordinator."""
        print("🧠 Editing NextJS application with LLM Coordinator...")
        
        # Initialize coordinator if not already done
        if not self.coordinator:
            self.coordinator = LLMCoordinator(app_builder=self)
        
        # Use coordinator to create and execute editing plan
        app_directory = self.get_app_path()
        success, error_msg = self.coordinator.coordinate_app_editing(app_idea, app_directory)
        
        if success:
            print("✅ Coordinator successfully edited the application!")
            return True
        else:
            print(f"❌ Coordinator failed: {error_msg}")
            print("🔄 Falling back to traditional rollback approach...")
            return self._edit_app_with_rollback(app_idea, prefer_intent_based=True)
    
    def _edit_app_with_rollback(self, app_idea: str, prefer_intent_based: bool = False) -> bool:
        """
        Robust edit method with automatic rollback and multiple strategies.
        
        This method GUARANTEES that the app remains in a working state by:
        1. Creating a snapshot before any changes
        2. Trying multiple editing strategies
        3. Rolling back on any failure
        4. Using progressively simpler approaches
        5. Never leaving the app in a broken state
        
        Args:
            app_idea: Description of changes to make
            prefer_intent_based: Whether to prefer intent-based editing first
            
        Returns:
            bool: True if edit was successful, False if all strategies failed
        """
        app_path = self.get_app_path()
        
        # Initialize version manager
        self._ensure_version_manager(app_path)
        
        print("🛡️ Starting robust edit with automatic rollback...")
        print(f"🎯 Edit request: {app_idea}")
        print(f"📁 App: {self.app_name}")
        
        # Step 1: Create snapshot before any changes
        try:
            snapshot_id = self._version_manager.create_snapshot(f"Before edit: {app_idea[:50]}...")
            print(f"📸 Created safety snapshot: {snapshot_id}")
        except Exception as e:
            print(f"❌ Failed to create snapshot: {e}")
            print("⚠️ Proceeding without snapshot - RISKY!")
            snapshot_id = None
        
        # Define editing strategies in order of preference
        strategies = []
        
        if prefer_intent_based:
            strategies = [
                ("intent_based", "Intent-based editing with structured JSON"),
                ("diff_based", "Unified diff with context matching"),
                ("line_based_fallback", "Legacy line-based editing"),
                ("manual_fix", "Manual syntax error fixing"),
            ]
        else:
            strategies = [
                ("diff_based", "Unified diff with context matching"),
                ("intent_based", "Intent-based editing with structured JSON"),
                ("line_based_fallback", "Legacy line-based editing"),
                ("manual_fix", "Manual syntax error fixing"),
            ]
        
        # Try each strategy until one succeeds
        for i, (strategy_name, strategy_desc) in enumerate(strategies, 1):
            print(f"\n🔧 Strategy {i}/{len(strategies)}: {strategy_desc}")
            
            try:
                # Attempt the edit strategy
                success = self._try_edit_strategy(strategy_name, app_idea)
                
                if success:
                    # Check if app still builds
                    print("🔨 Verifying build after edit...")
                    build_success = self.build_and_run(auto_install_deps=False)
                    
                    if build_success:
                        # Log successful attempt
                        if self._version_manager:
                            self._version_manager.log_edit_attempt(
                                strategy=strategy_name,
                                description=app_idea,
                                success=True,
                                build_success=True
                            )
                        
                        print(f"🎉 Edit successful with {strategy_name}!")
                        print("✅ App builds correctly after changes")
                        return True
                    else:
                        print(f"⚠️ Edit applied but build failed with {strategy_name}")
                        
                        # Log failed build attempt
                        if self._version_manager:
                            self._version_manager.log_edit_attempt(
                                strategy=strategy_name,
                                description=app_idea,
                                success=True,
                                build_success=False,
                                error_message="Build failed after edit"
                            )
                        
                        # Rollback to snapshot
                        if snapshot_id and self._version_manager:
                            print("🔄 Rolling back due to build failure...")
                            rollback_success = self._version_manager.rollback_to_snapshot(snapshot_id)
                            if rollback_success:
                                print("✅ Rollback successful, trying next strategy...")
                            else:
                                print("❌ Rollback failed! App may be in inconsistent state!")
                        
                        continue  # Try next strategy
                else:
                    print(f"❌ {strategy_name} failed to apply changes")
                    
                    # Log failed attempt
                    if self._version_manager:
                        self._version_manager.log_edit_attempt(
                            strategy=strategy_name,
                            description=app_idea,
                            success=False,
                            error_message=f"{strategy_name} failed to apply"
                        )
                    
                    # Rollback to snapshot if changes were partially applied
                    if snapshot_id and self._version_manager:
                        print("🔄 Rolling back due to edit failure...")
                        rollback_success = self._version_manager.rollback_to_snapshot(snapshot_id)
                        if rollback_success:
                            print("✅ Rollback successful, trying next strategy...")
                        else:
                            print("❌ Rollback failed! App may be in inconsistent state!")
                    
                    continue  # Try next strategy
                    
            except Exception as e:
                print(f"❌ Strategy {strategy_name} encountered error: {str(e)}")
                
                # Log exception
                if self._version_manager:
                    self._version_manager.log_edit_attempt(
                        strategy=strategy_name,
                        description=app_idea,
                        success=False,
                        error_message=f"Exception: {str(e)}"
                    )
                
                # Rollback on exception
                if snapshot_id and self._version_manager:
                    print("🔄 Rolling back due to exception...")
                    rollback_success = self._version_manager.rollback_to_snapshot(snapshot_id)
                    if rollback_success:
                        print("✅ Rollback successful, trying next strategy...")
                    else:
                        print("❌ Rollback failed! App may be in inconsistent state!")
                
                continue  # Try next strategy
        
        # All strategies failed
        print("💥 All editing strategies failed!")
        
        # Final rollback to ensure app is in working state
        if snapshot_id and self._version_manager:
            print("🔄 Final rollback to ensure app stability...")
            final_rollback = self._version_manager.rollback_to_snapshot(snapshot_id)
            if final_rollback:
                print("✅ App restored to working state")
            else:
                print("❌ Final rollback failed - app may be unstable!")
                # Try rollback to last known working state as ultimate fallback
                emergency_rollback = self._version_manager.rollback_to_last_working()
                if emergency_rollback:
                    print("🆘 Emergency rollback to last working state successful!")
                else:
                    print("💀 CRITICAL: All rollback attempts failed!")
        
        print("📊 Edit operation completed with all strategies exhausted")
        return False
    
    def _try_edit_strategy(self, strategy_name: str, app_idea: str) -> bool:
        """Try a specific editing strategy."""
        if strategy_name == "intent_based":
            return self._edit_app_with_intents(app_idea)
        elif strategy_name == "diff_based":
            return self._edit_app_with_diffs(app_idea)
        elif strategy_name == "line_based_fallback":
            return self._edit_app_with_line_based_fallback(app_idea)
        elif strategy_name == "manual_fix":
            return self._edit_app_with_manual_fixes(app_idea)
        else:
            print(f"❌ Unknown strategy: {strategy_name}")
            return False
    
    def _edit_app_with_line_based_fallback(self, app_idea: str) -> bool:
        """Fallback to line-based editing for simple changes."""
        print("🔄 Using line-based fallback editing...")
        
        # Get semantic context
        semantic_context = self.get_semantic_context_for_request(
            user_request=app_idea, 
            app_directory=self.get_app_path()
        )
        
        if not semantic_context:
            print("❌ Failed to get semantic context")
            return False
        
        # Generate unified diff edit
        prompt = f"""You are editing a NextJS app using unified diff format.

CURRENT APP CONTEXT:
{semantic_context}

USER REQUEST:
{app_idea}

🚨 CRITICAL DIFF GENERATION RULES 🚨

You MUST respond with a UNIFIED DIFF in this EXACT format:

*** Begin Patch
*** Update File: path/to/file.tsx
@@ -old_start,old_count +new_start,new_count @@ optional_context
- old line to remove
+ new line to add
 unchanged context line
*** End Patch

KEY REQUIREMENTS:
1. Use ONLY the unified diff format shown above
2. Include 2-3 lines of context before and after changes for accurate matching
3. Use proper +/- prefixes for added/removed lines
4. Use space prefix for unchanged context lines
5. Multiple files = multiple "*** Update File:" sections in ONE patch
6. NO line-based <edit> tags - ONLY unified diffs
7. Context lines must EXACTLY match existing file content

Make minimal, surgical changes with proper context matching.
"""
        
        is_valid, response = self.make_openai_request(prompt, context="edit")
        
        if not is_valid:
            return False
        
        # Apply using CodeBuilder
        from .code_builder import CodeBuilder
        import tempfile
        import os
        
        timestamp = int(time.time())
        edit_file = os.path.join("inputs", self.app_name, f"line_fallback_{timestamp}.txt")
        
        os.makedirs(os.path.dirname(edit_file), exist_ok=True)
        
        with open(edit_file, 'w') as f:
            f.write(response)
        
        try:
            code_builder = CodeBuilder(edit_file, self.get_app_path())
            code_builder.build()
            return True
        except Exception as e:
            print(f"❌ Line-based fallback failed: {e}")
            return False
    
    def _edit_app_with_manual_fixes(self, app_idea: str) -> bool:
        """Apply manual fixes for common syntax errors."""
        print("🔧 Applying manual syntax fixes...")
        
        app_path = Path(self.get_app_path())
        
        # Check for common syntax errors in main files
        main_files = [
            app_path / "app" / "page.tsx",
            app_path / "app" / "layout.tsx"
        ]
        
        fixed_any = False
        
        for file_path in main_files:
            if not file_path.exists():
                continue
                
            try:
                content = file_path.read_text()
                original_content = content
                
                # Fix common syntax issues
                # 1. Double opening braces in useEffect
                content = re.sub(r'useEffect\(\(\) => \{ \{', 'useEffect(() => {', content)
                
                # 2. Missing closing parentheses in JSX returns
                content = re.sub(r'return \((.*?)\n\s*\}', r'return (\1\n  )', content, flags=re.DOTALL)
                
                # 3. Fix malformed imports
                content = re.sub(r'import \{ useState \} from \'react\'\nimport \{ useState', 'import { useState', content)
                
                # 4. Fix missing semicolons in key places
                content = re.sub(r'(\]\))\n\s*(const|let|var)', r'\1;\n\2', content)
                
                if content != original_content:
                    file_path.write_text(content)
                    print(f"🔧 Applied manual fixes to {file_path.name}")
                    fixed_any = True
                    
            except Exception as e:
                print(f"⚠️ Could not fix {file_path.name}: {e}")
                continue
        
        return fixed_any
    
    def _edit_app_with_intents(self, app_idea):
        """
        Edit app using structured intents with enhanced robust parsing.
        
        ENHANCED: Uses robust parser, live progress indicators, and intelligent context selection.
        """
        print("🎯 Using enhanced intent-based editing with robust parsing...")
        
        # Enhanced request processing
        print("\n🎯 Request Enhancement Phase")
        print("=" * 40)
        
        try:
            from .request_enhancer import RequestEnhancer
            enhancer = RequestEnhancer()
            enhanced_request = enhancer.enhance_edit_request(app_idea, self.get_app_path())
            print(f"✅ Request enhanced ({len(enhanced_request)} chars)")
        except Exception as e:
            print(f"⚠️ Request enhancement failed: {e}")
            print("🔄 Using original request")
            enhanced_request = app_idea
        
        # Intelligent context selection
        print("\n🧠 Context Selection Phase") 
        print("=" * 40)
        
        try:
            from .context_selector import IntelligentContextSelector
            context_selector = IntelligentContextSelector()
            app_path = self.get_app_path()
            context_files = context_selector.select_context(enhanced_request, app_path)
            
            print(f"📁 Selected {len(context_files.selected_files)} context files")
            print(f"📊 Total context: {context_files.total_tokens} tokens")
            
            # Extract just file paths for the enhanced method
            context_file_paths = [cf.path for cf in context_files.selected_files]
            
        except Exception as e:
            print(f"⚠️ Context selection failed: {e}")
            print("🔄 Using basic context")
            context_file_paths = []
        
        # Enhanced intent generation with live progress
        print(f"\n🚀 Intent Generation & Application Phase")
        print("=" * 40)
        
        try:
            # Use our enhanced method with robust parsing and live progress
            result = self.generate_intent_based_edits(enhanced_request, context_file_paths)
            
            if result == "SUCCESS":
                print("🎉 Intent-based editing completed successfully!")
                return True
            elif result == "PARTIAL_SUCCESS":
                print("⚠️ Intent-based editing partially successful")
                print("🔄 Some operations failed but core functionality may be working")
                return True  # Consider partial success as success for now
            else:
                print("❌ Enhanced intent-based editing failed")
                return False
                
        except Exception as e:
            print(f"💥 Exception during intent-based editing: {e}")
            return False
    
    def _edit_app_with_diffs(self, app_idea):
        """Edit app using traditional diff-based approach (with sanitization)."""
        from .diff_builder import AtomicDiffBuilder
        
        print(f"📝 Editing existing app with diff-based approach: {self.app_name}")
        print(f"🎯 Requested changes: {app_idea}")
        
        # Get semantically relevant context for this edit request
        semantic_context = self.get_semantic_context_for_request(
            user_request=app_idea, 
            app_directory=self.get_app_path()
        )
        if not semantic_context:
            print("❌ Failed to get semantic context for edit request")
            return False
        
        # Generate edit response using diff format
        print("🤖 Generating unified diff with semantic context...")
        is_valid, response = self.make_openai_request(
            self.get_edit_prompt(app_idea, semantic_context),
            context="edit"
        )
        
        if not is_valid:
            print("❌ Failed to generate valid diff")
            return False
        
        # Save the diff to a temporary file
        import tempfile
        import os
        
        timestamp = int(time.time())
        diff_file = os.path.join("inputs", self.app_name, f"diff_{timestamp}.patch")
        
        # Ensure inputs directory exists
        os.makedirs(os.path.dirname(diff_file), exist_ok=True)
        
        with open(diff_file, 'w') as f:
            f.write(response)
        
        print(f"💾 Saved diff to: {diff_file}")
        
        # Apply the diff
        print("🔧 Applying unified diff...")
        diff_builder = AtomicDiffBuilder(self.get_app_path(), diff_file)
        results = diff_builder.apply_patch_atomically()
        success = all(result.success for result in results)
        
        if not success:
            print("❌ Failed to apply diff - attempting recovery...")
            # Try to fix any issues and retry
            return self._handle_diff_failure(app_idea, semantic_context)
        
        print("✅ Diff applied successfully!")
        
        # Build and run to verify changes work
        print("🔨 Building app to verify changes...")
        build_success = self.build_and_run(auto_install_deps=True)
        
        if build_success:
            print("🎉 App edited successfully and builds correctly!")
            return True
        else:
            print("⚠️ App edited but has build errors - attempting automatic fixes...")
            return self.auto_fix_build_errors()
    
    def _handle_diff_failure(self, app_idea, semantic_context):
        """Handle diff application failure with fallback strategies."""
        print("🩹 Diff application failed - trying recovery strategies...")
        
        # Strategy 1: Generate a simpler diff with more context
        print("📋 Strategy 1: Generating diff with enhanced semantic context...")
        enhanced_prompt = self.get_edit_prompt(app_idea, semantic_context) + """

⚠️ ENHANCED CONTEXT REQUIREMENTS ⚠️
The previous diff failed to apply. Please generate a new diff with:
1. More context lines (4-5 lines before and after changes)
2. Exact matching of existing content
3. Smaller, more targeted changes
4. Extra care with whitespace and indentation matching

Focus on making minimal, surgical changes that are easy to apply."""

        is_valid, response = self.make_openai_request(enhanced_prompt, context="edit")
        
        if is_valid:
            # Try applying the enhanced diff
            import tempfile
            import os
            
            timestamp = int(time.time())
            diff_file = os.path.join("inputs", self.app_name, f"diff_recovery_{timestamp}.patch")
            
            with open(diff_file, 'w') as f:
                f.write(response)
                
            from .diff_builder import AtomicDiffBuilder
            diff_builder = AtomicDiffBuilder(self.get_app_path(), diff_file)
            results = diff_builder.apply_patch_atomically()
            success = all(result.success for result in results)
            
            if success:
                print("✅ Recovery diff applied successfully!")
                return True
        
        # Strategy 2: Fall back to line-based editing for critical fixes
        print("📋 Strategy 2: Falling back to legacy line-based editing...")
        return self._fallback_to_line_based_edit(app_idea, semantic_context)
    
    def _fallback_to_line_based_edit(self, app_idea, semantic_context):
        """Fallback to the old line-based editing system as last resort."""
        from .code_builder import CodeBuilder
        
        print("🔄 Using legacy line-based editing system...")
        
        # Generate unified diff response  
        legacy_prompt = """You are editing an existing NextJS application. Here is the semantic context:

""" + semantic_context + """

User wants to make the following changes:
""" + app_idea + """

🚨 CRITICAL DIFF GENERATION RULES 🚨

You MUST respond with a UNIFIED DIFF in this EXACT format:

*** Begin Patch
*** Update File: path/to/file.tsx
@@ -old_start,old_count +new_start,new_count @@ optional_context
- old line to remove
+ new line to add
 unchanged context line
*** End Patch

KEY REQUIREMENTS:
1. Use ONLY the unified diff format shown above
2. Include 2-3 lines of context before and after changes for accurate matching
3. Use proper +/- prefixes for added/removed lines
4. Use space prefix for unchanged context lines
5. Multiple files = multiple "*** Update File:" sections in ONE patch
6. NO line-based <edit> tags - ONLY unified diffs
7. Context lines must EXACTLY match existing file content

Make minimal, targeted changes to implement the requested features."""

        is_valid, response = self.make_openai_request(legacy_prompt, context="edit")  # Use unified diff format
        
        if not is_valid:
            print("❌ Failed to generate fallback response")
            return False
        
        # Save and apply using legacy system
        timestamp = int(time.time())
        edit_file = os.path.join("inputs", self.app_name, f"fallback_{timestamp}.txt")
        
        with open(edit_file, 'w') as f:
            f.write(response)
        
        code_builder = CodeBuilder(edit_file, self.get_app_path())
        return code_builder.build()

    def parse_build_errors(self) -> List[str]:
        """
        Parse build errors by running npm build and extracting error details.
        
        Returns:
            List of build error strings
        """
        import subprocess
        import os
        
        app_path = self.get_app_path()
        
        try:
            # Change to app directory
            original_dir = os.getcwd()
            os.chdir(app_path)
            
            # Try to build the app to get errors
            result = subprocess.run(['npm', 'run', 'build'], 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Build successful, no errors
                return []
            
            # Parse the build output to extract errors
            build_output = result.stdout + result.stderr
            errors = self._extract_build_errors_from_output(build_output)
            
            return errors
                
        except subprocess.TimeoutExpired:
            return ["Build timed out"]
        except FileNotFoundError:
            return ["npm not found - please install Node.js"]
        except Exception as e:
            return [f"Build error: {str(e)}"]
        finally:
            # Return to original directory
            try:
                os.chdir(original_dir)
            except:
                pass
    
    def _extract_build_errors_from_output(self, build_output: str) -> List[str]:
        """Extract specific error messages from NextJS build output."""
        errors = []
        lines = build_output.split('\n')
        
        current_error = []
        in_error_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect error sections
            if 'Failed to compile' in line or 'Error:' in line:
                if current_error:
                    errors.append('\n'.join(current_error))
                current_error = [line_stripped]
                in_error_section = True
                continue
            
            # TypeScript/ESLint error patterns
            if any(pattern in line for pattern in [
                'Type error:', 'SyntaxError:', 'ReferenceError:', 
                'the name', 'is defined multiple times', 'does not exist on type',
                'Property', 'Argument of type', 'Cannot find module'
            ]):
                if not in_error_section:
                    current_error = []
                    in_error_section = True
                current_error.append(line_stripped)
                continue
            
            # File path lines (like ./app/page.tsx)
            if line_stripped.startswith('./') and any(ext in line for ext in ['.tsx', '.ts', '.jsx', '.js']):
                if not in_error_section:
                    current_error = []
                    in_error_section = True
                current_error.append(line_stripped)
                continue
            
            # Error details and context
            if in_error_section and line_stripped:
                # Look for error indicators
                if any(indicator in line for indicator in [
                    '×', '✗', '╭─', '│', '╰──', '·', 'at line', 'previous definition',
                    'redefined here', 'Import trace'
                ]):
                    current_error.append(line_stripped)
                    continue
                
                # End of error section
                if any(end_pattern in line for end_pattern in [
                    '> Build failed', 'webpack errors', 'info  -', 'warn  -'
                ]):
                    if current_error:
                        errors.append('\n'.join(current_error))
                        current_error = []
                    in_error_section = False
                    break
        
        # Add any remaining error
        if current_error:
            errors.append('\n'.join(current_error))
        
        return errors

    def detect_infrastructure_errors(self, errors: List[str]) -> List[str]:
        """
        Detect infrastructure-related errors (missing dependencies, etc.)
        
        Args:
            errors: List of error strings
            
        Returns:
            List of infrastructure error strings
        """
        infrastructure_errors = []
        
        for error in errors:
            error_lower = error.lower()
            
            # Check for dependency issues
            if any(keyword in error_lower for keyword in [
                'module not found', 'cannot resolve', 'package not found',
                'dependency', 'npm error', 'yarn error', 'pnpm error',
                'missing dependency', 'failed to install'
            ]):
                infrastructure_errors.append(error)
            
            # Check for build tool issues
            elif any(keyword in error_lower for keyword in [
                'webpack', 'next/config', 'build failed',
                'compilation error', 'bundler error'
            ]):
                infrastructure_errors.append(error)
        
        return infrastructure_errors

    def fix_infrastructure_errors(self, infrastructure_errors: List[str]) -> bool:
        """
        Fix infrastructure-related errors using smart build error analysis.
        
        Args:
            infrastructure_errors: List of infrastructure error strings
            
        Returns:
            True if fixes were applied successfully
        """
        import subprocess
        import os
        
        app_path = self.get_app_path()
        
        try:
            # Change to app directory
            original_dir = os.getcwd()
            os.chdir(app_path)
            
            # Combine all infrastructure errors into one string for analysis
            error_output = "\n".join(infrastructure_errors)
            
            # Try smart build error analysis first
            print("🧠 Using smart build error analysis for infrastructure errors...")
            try:
                from .build_error_analyzer import SmartBuildErrorAnalyzer
                from .smart_task_executor import SmartTaskExecutor
                
                # Analyze the errors with full context
                analyzer = SmartBuildErrorAnalyzer(app_path)
                analysis = analyzer.analyze_build_error(error_output)
                
                print(f"   📋 Analysis: {analysis.error_summary}")
                print(f"   🎯 Root cause: {analysis.root_cause}")
                print(f"   📊 Confidence: {analysis.confidence:.1%}")
                
                if analysis.confidence > 0.4 and analysis.tasks:
                    # Execute the fix tasks
                    executor = SmartTaskExecutor(app_path)
                    success, task_results = executor.execute_analysis(analysis)
                    
                    if success:
                        print("✅ Smart infrastructure error fix applied successfully!")
                        return True
                    else:
                        print("⚠️ Smart infrastructure error fix partially failed")
                        print(executor.get_execution_summary())
                        # Continue with fallback
                else:
                    print(f"❌ Low confidence analysis ({analysis.confidence:.1%}), using fallback...")
                    
            except Exception as e:
                print(f"❌ Smart error analysis failed: {e}")
            
            # Fallback: Use DependencyManager for comprehensive dependency fixing
            print("🔧 Fallback: Analyzing and fixing missing dependencies...")
            try:
                from .dependency_manager import DependencyManager
                dependency_manager = DependencyManager(app_path)
                if dependency_manager.auto_manage_dependencies():
                    print("✅ Dependencies analyzed and installed successfully")
                    return True
                else:
                    print("⚠️ Dependency management completed with warnings")
                    return True  # Still consider it successful
            except Exception as e:
                print(f"❌ Error with dependency manager: {e}")
                # Final fallback to basic npm install
                print("🔧 Final fallback: Basic npm install...")
                result = subprocess.run(['npm', 'install'], 
                                      capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print("✅ Basic npm install completed successfully")
                    return True
                else:
                    print(f"❌ npm install failed: {result.stderr}")
                    return False
                
        except Exception as e:
            print(f"❌ Error fixing infrastructure: {e}")
            return False
        finally:
            os.chdir(original_dir)

    def auto_fix_build_errors(self):
        """Automatically fix build errors using diff-based approach with enhanced error handling."""
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"🔧 Auto-fix attempt {attempt + 1}/{max_attempts}")
            
            # Parse build errors with enhanced extraction
            build_errors = self.parse_build_errors()
            if not build_errors:
                print("✅ No build errors found!")
                return True
            
            print(f"🚨 Found {len(build_errors)} build errors:")
            for i, error in enumerate(build_errors, 1):
                print(f"   {i}. {error}")
            
            # Check for infrastructure errors first
            infrastructure_errors = self.detect_infrastructure_errors(build_errors)
            if infrastructure_errors:
                print("🔧 Detected infrastructure errors - applying automatic fixes...")
                if self.fix_infrastructure_errors(infrastructure_errors):
                    # Try building again after infrastructure fixes
                    if self.build_and_run(auto_install_deps=False):
                        print("✅ Infrastructure fixes resolved the build errors!")
                        return True
                    else:
                        print("📋 Infrastructure fixed, but other errors remain...")
                        continue
            
            # Get semantic context for error fixing
            error_context_query = f"build errors: {'; '.join(build_errors[:3])}"  # Use first few errors as query
            app_structure = self.get_semantic_context_for_request(
                user_request=error_context_query,
                app_directory=self.get_app_path()
            )
            if not app_structure:
                print("❌ Failed to get semantic context for error fixing")
                return False
            
            # Extract error files content for better context
            error_files_content = self.extract_error_files_content(build_errors, app_structure)
            
            # Build comprehensive error context
            error_context = """BUILD ERRORS THAT NEED TO BE FIXED:
""" + '\n'.join([f"ERROR {i+1}: {error}" for i, error in enumerate(build_errors)])
            
            if error_files_content:
                error_context += """

SPECIFIC FILES WITH ERRORS:
""" + error_files_content
            
            # Generate fix using diff format
            print("🤖 Generating fix diff...")
            is_valid, response = self.make_openai_request(
                self.get_build_fix_prompt(app_structure, error_context),
                context="edit"
            )
            
            if not is_valid:
                print(f"❌ Failed to generate valid fix diff (attempt {attempt + 1})")
                continue
            
            # Save and apply the fix diff
            timestamp = int(time.time())
            fix_file = os.path.join("inputs", self.app_name, f"autofix_{timestamp}.patch")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(fix_file), exist_ok=True)
            
            with open(fix_file, 'w') as f:
                f.write(response)
            
            print(f"💾 Saved fix diff to: {fix_file}")
            
            # Apply the fix diff
            from .diff_builder import AtomicDiffBuilder
            diff_builder = AtomicDiffBuilder(self.get_app_path(), fix_file)
            results = diff_builder.apply_patch_atomically()
            fix_success = all(result.success for result in results)
            
            if not fix_success:
                print(f"❌ Failed to apply fix diff (attempt {attempt + 1})")
                # Try fallback approach for this attempt
                if self._try_fallback_fix(build_errors, app_structure, error_context):
                    print("✅ Fallback fix applied successfully!")
                else:
                    print("❌ Fallback fix also failed")
                continue
            
            print(f"✅ Fix diff applied successfully (attempt {attempt + 1})")
            
            # Test the build again
            print("🔨 Testing build after fixes...")
            if self.build_and_run(auto_install_deps=False):
                print("🎉 Build errors fixed successfully!")
                return True
            else:
                print(f"⚠️ Some errors remain after attempt {attempt + 1}")
        
        print("💥 Failed to fix build errors after maximum attempts")
        return False
    
    def _try_fallback_fix(self, build_errors, semantic_context, error_context):
        """Try a fallback fix using the legacy line-based system."""
        print("🔄 Attempting fallback fix with legacy system...")
        
        from .code_builder import CodeBuilder
        
        # Generate legacy fix prompt
        legacy_fix_prompt = """You are fixing build errors in an existing NextJS application. Here is the semantic context:

""" + semantic_context + """

""" + error_context + """

🚨 CRITICAL DIFF GENERATION RULES 🚨

You MUST respond with a UNIFIED DIFF in this EXACT format:

*** Begin Patch
*** Update File: path/to/file.tsx
@@ -old_start,old_count +new_start,new_count @@ optional_context
- old line to remove
+ new line to add
 unchanged context line
*** End Patch

Make minimal, surgical fixes to resolve the exact errors shown above.

Focus on fixing:
- Syntax errors (missing commas, brackets, quotes)
- JSX return statement issues
- Import/export problems
- Type errors"""

        is_valid, response = self.make_openai_request(legacy_fix_prompt, context="edit")
        
        if not is_valid:
            return False
        
        # Apply using legacy system
        timestamp = int(time.time())
        fallback_file = os.path.join("inputs", self.app_name, f"fallback_fix_{timestamp}.txt")
        
        with open(fallback_file, 'w') as f:
            f.write(response)
        
        code_builder = CodeBuilder(fallback_file, self.get_app_path())
        return code_builder.build()

    def make_openai_request(self, prompt: str, context: str = "create") -> Tuple[bool, str]:
        """
        Make an AI request and return validation result and response.
        
        Args:
            prompt: The prompt to send to the AI
            context: Context for validation ("create" or "edit")
            
        Returns:
            Tuple of (is_valid, response)
        """
        try:
            print("🎯 Preparing AI request...")
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            response = self.generate_with_fallback(messages, context=context)
            if not response:
                return False, ""
            
            # 🔍 DEBUG: Save and print the raw AI response
            print("📝 Processing response...")
            import os
            import time
            
            # Save to debug file
            debug_dir = "debug_responses"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = int(time.time())
            debug_file = f"{debug_dir}/ai_response_{timestamp}.txt"
            
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=== RAW AI RESPONSE ===\n")
                f.write(response)
                f.write("\n=== END RESPONSE ===\n")
            
            print(f"🔍 DEBUG: Raw AI response saved to {debug_file}")
            print("🔍 DEBUG: First 500 characters of AI response:")
            print("-" * 50)
            print(response[:500])
            print("-" * 50)
            if len(response) > 500:
                print(f"... (truncated, full response in {debug_file})")
            
            # Check for JSX issues in the raw response
            if 'return (' in response and '}>' in response:
                jsx_samples = []
                lines = response.split('\n')
                for i, line in enumerate(lines):
                    if 'return (' in line:
                        # Get this line and next few lines to see the pattern
                        sample = '\n'.join(lines[i:i+10])
                        jsx_samples.append(f"Line {i+1}: {sample}")
                
                if jsx_samples:
                    print("🚨 POTENTIAL JSX ISSUES DETECTED IN RAW RESPONSE:")
                    for sample in jsx_samples[:3]:  # Show first 3 samples
                        print(sample)
                        print("-" * 30)
            
            # Validate the response
            print("✅ Validating response...")
            is_valid, validation_feedback = self.validate_response(response, context=context)
            
            if is_valid:
                print("✅ Generated valid response")
                return True, response
            else:
                print("❌ Response validation failed:")
                for feedback in validation_feedback[:3]:  # Limit feedback
                    print(f"   - {feedback}")
                return False, response
                
        except Exception as e:
            print(f"❌ Error in AI request: {str(e)}")
            return False, ""

    def extract_error_files_content(self, build_errors: List[str], app_structure: str) -> str:
        """
        Extract content from files mentioned in build errors for better context.
        
        Args:
            build_errors: List of build error messages
            app_structure: Current app structure context
            
        Returns:
            String containing relevant file contents with error context
        """
        try:
            error_files_content = []
            app_path = Path(self.get_app_path())
            
            # Extract file paths from error messages
            error_files = set()
            for error in build_errors:
                # Common patterns for file paths in NextJS errors
                import re
                file_patterns = [
                    r'\./(app/[^:\s]+)',
                    r'\./(src/[^:\s]+)',
                    r'\./(components/[^:\s]+)',
                    r'\./(lib/[^:\s]+)',
                    r'/([^/\s]+\.tsx?)',
                    r'/([^/\s]+\.jsx?)',
                ]
                
                for pattern in file_patterns:
                    matches = re.findall(pattern, error)
                    error_files.update(matches)
            
            # Read content from error files
            for file_path in error_files:
                full_path = app_path / file_path
                if full_path.exists() and full_path.is_file():
                    try:
                        content = full_path.read_text(encoding='utf-8')
                        error_files_content.append(f"""
=== {file_path} ===
{content}
=== END {file_path} ===
""")
                    except Exception as e:
                        error_files_content.append(f"Error reading {file_path}: {e}")
            
            return '\n'.join(error_files_content) if error_files_content else ""
            
        except Exception as e:
            print(f"❌ Error extracting error files content: {e}")
            return ""

    def get_build_fix_prompt(self, app_structure: str, error_context: str) -> str:
        """
        Generate a prompt for fixing build errors using unified diff format.
        
        Args:
            app_structure: Current app structure and context
            error_context: Build errors and file contents that need fixing
            
        Returns:
            Prompt string for AI to generate fixes
        """
        return f"""You are fixing build errors in a NextJS application. Please analyze the errors and generate a unified diff to fix them.

CURRENT APP STRUCTURE:
{app_structure}

{error_context}

INSTRUCTIONS:
Think step by step through this fix:

1. **ANALYZE THE ERRORS**: What exactly is causing each build error?
2. **IDENTIFY ROOT CAUSES**: Are these syntax, import, type, or logic errors?
3. **PLAN THE FIXES**: What specific changes will resolve each error?
4. **PRIORITIZE FIXES**: Which errors should be fixed first to avoid cascading issues?
5. **IMPLEMENT PRECISELY**: Generate the exact diff needed.

TECHNICAL REQUIREMENTS:
1. Generate a unified diff using *** Begin Patch / *** End Patch format
2. Fix ONLY the specific errors mentioned above
3. Make minimal, surgical changes
4. Ensure all JSX syntax is correct
5. Fix import/export issues
6. Resolve TypeScript type errors
7. Use exact whitespace and indentation matching

EXAMPLE FORMAT:
*** Begin Patch
*** Update File: app/page.tsx
@@ -10,7 +10,7 @@
   const [count, setCount] = useState(0)
 
   return (
-    <div className="container">
+    <div className="container mx-auto">
       <h1>{{count}}</h1>
     </div>
   )
*** End Patch

Focus on making the minimum changes needed to resolve the build errors."""

    def generate_build_fix_response(self, errors: List[str], app_structure: str) -> str:
        """
        Generate build fix instructions based on specific build errors.
        
        Args:
            errors: List of build error messages
            app_structure: Current app structure context
            
        Returns:
            Fix instructions as a string, or empty string if failed
        """
        try:
            error_context = self.extract_error_files_content(errors, app_structure)
            prompt = self.get_build_fix_prompt(app_structure, error_context)
            
            is_valid, response = self.make_openai_request(prompt, context="edit")
            
            if is_valid:
                return response
            else:
                print("❌ Failed to generate valid build fix response")
                return ""
                
        except Exception as e:
            print(f"❌ Error generating build fix response: {e}")
            return ""

    def generate_file_rewrite_response(self, file_paths: List[str], errors: List[str], semantic_context: str) -> str:
        """
        Generate complete file rewrite instructions.
        
        Args:
            file_paths: List of file paths to rewrite
            errors: List of current errors
            semantic_context: Additional context about the changes needed
            
        Returns:
            Rewrite instructions as a string, or empty string if failed
        """
        try:
            # Build context for file rewriting
            files_content = []
            app_path = Path(self.get_app_path())
            
            for file_path in file_paths:
                full_path = app_path / file_path
                if full_path.exists():
                    try:
                        content = full_path.read_text()
                        files_content.append(f"=== {file_path} ===\n{content}\n")
                    except Exception as e:
                        files_content.append(f"=== {file_path} ===\nError reading file: {e}\n")
                else:
                    files_content.append(f"=== {file_path} ===\nFile does not exist\n")
            
            # Create rewrite prompt
            prompt = f"""You need to completely rewrite the following files to fix build errors.

CURRENT ERRORS:
{chr(10).join(errors)}

SEMANTIC CONTEXT:
{semantic_context}

CURRENT FILES:
{chr(10).join(files_content)}

Please provide complete, corrected file contents using the <new> tag format:

<new filename="path/to/file.tsx">
Complete corrected file content here
</new>

Requirements:
1. Fix ALL the build errors mentioned above
2. Maintain existing functionality where possible
3. Use proper NextJS 13+ App Router patterns
4. Use correct TypeScript types
5. Follow the coding standards shown in the original files"""

            is_valid, response = self.make_openai_request(prompt, context="create")
            
            if is_valid:
                return response
            else:
                print("❌ Failed to generate valid file rewrite response")
                return ""
                
        except Exception as e:
            print(f"❌ Error generating file rewrite response: {e}")
            return ""

    def generate_intent_based_edits(self, request: str, context_files: List[str] = None) -> Optional[str]:
        """
        Generate structured editing intents using intent-based approach.
        
        ENHANCED: Uses robust parser with live progress indicators.
        """
        print("\n🎯 Generating structured editing intents...")
        print("=" * 50)
        
        # Show progress for context preparation
        import time
        import sys
        
        def show_progress_dots(message, duration=1.0):
            """Show simple progress message (no animation to avoid stdout conflicts)"""
            print(f"{message}...")
            time.sleep(duration * 0.1)  # Brief pause instead of animation
            print(f"{message} ✓")
        
        # Prepare context with progress
        show_progress_dots("📋 Preparing context files", 0.8)
        
        context_content = ""
        if context_files:
            print(f"📁 Including {len(context_files)} context files:")
            for file_path in context_files:
                print(f"   • {file_path}")
                try:
                    full_path = self.apps_dir / self.app_name / file_path
                    if full_path.exists():
                        content = full_path.read_text()
                        # Include only file structure and key imports, not full content
                        lines = content.split('\n')
                        
                        # Extract key information instead of full content
                        summary_lines = []
                        summary_lines.append(f"File: {file_path}")
                        
                        # Include imports (first 10 lines that start with import/export)
                        import_lines = [line for line in lines[:20] if line.strip().startswith(('import', 'export', 'const', 'interface', 'type'))]
                        if import_lines:
                            summary_lines.extend(import_lines[:5])
                        
                        # Include key structural info
                        if 'function' in content or 'const' in content:
                            summary_lines.append("Contains: React components/functions")
                        if 'interface' in content or 'type' in content:
                            summary_lines.append("Contains: TypeScript definitions")
                        
                        summary_lines.append(f"Size: {len(content)} characters")
                        summary_lines.append("---")
                        
                        context_content += f"\n### {file_path}\n" + '\n'.join(summary_lines) + "\n"
                except Exception as e:
                    print(f"   ⚠️ Could not read {file_path}: {e}")
        
        # Prepare the enhanced prompt
        show_progress_dots("📝 Preparing AI prompt", 0.5)
        
        # Import the intent-based prompt function
        from .intent_editor import get_intent_based_prompt
        
        enhanced_prompt = f"""
{get_intent_based_prompt()}

### CURRENT CONTEXT
{context_content}

### USER REQUEST
{request}

Please analyze the request and current context, then provide structured editing intents in the exact JSON format specified above.
Focus on creating the missing pages and functionality as requested.
"""
        
        print(f"📊 Prompt size: {len(enhanced_prompt)} characters")
        
        # Try multiple LLM providers with live progress
        for i, provider in enumerate(self.llm_providers, 1):
            provider_name = provider.get('name', 'Unknown')
            print(f"\n🤖 [{i}/{len(self.llm_providers)}] Trying {provider_name}...")
            
            # Show connection progress
            show_progress_dots(f"🔗 Connecting to {provider_name}", 0.8)
            
            try:
                # Show generation progress
                print("🧠 Generating response...")
                
                # Simple thinking indicator (no animation to avoid stdout conflicts)
                print("🧠 AI processing your request...")
                
                # Call LLM based on provider type
                messages = [
                    {"role": "system", "content": "You are an expert NextJS developer. Respond with valid JSON."},
                    {"role": "user", "content": enhanced_prompt}
                ]
                
                response = None
                if provider["type"] == "anthropic" and self.anthropic_client:
                    response = self.call_anthropic(messages, provider)
                elif provider["type"] == "openai" and self.openai_client:
                    response = self.call_openai(messages, provider)
                elif provider["type"] == "openrouter" and self.openrouter_api_key:
                    response = self.call_openrouter(messages, provider)
                
                if not response:
                    print(f"❌ No response from {provider_name}")
                    continue
                
                print(f"✅ Received response from {provider_name} ({len(response)} chars)")
                
                # Show parsing progress
                show_progress_dots("🔍 Parsing AI response", 0.8)
                
                # Use the robust parser
                from .intent_editor import parse_ai_intent_response_robust
                intents = parse_ai_intent_response_robust(response)
                
                if intents:
                    print(f"🎉 Successfully extracted {len(intents)} intents!")
                    
                    # Show what was extracted
                    print("📋 Extracted intents:")
                    for j, intent in enumerate(intents, 1):
                        action_icon = {"insert": "📄", "replace": "✏️", "modify": "🔧", "delete": "🗑️"}.get(intent.action, "📝")
                        print(f"   {j}. {action_icon} {intent.file_path} ({intent.action})")
                    
                    # Apply the intents
                    print(f"\n🚀 Applying intents to codebase...")
                    from .intent_editor import IntentBasedEditor
                    
                    editor = IntentBasedEditor(self.apps_dir / self.app_name)
                    results = editor.apply_intent_list(intents)
                    
                    # Check results
                    successful = sum(1 for success, _ in results if success)
                    
                    if successful == len(results):
                        print(f"\n🎉 All {len(results)} operations completed successfully!")
                        return "SUCCESS"
                    else:
                        print(f"\n⚠️ {successful}/{len(results)} operations succeeded")
                        
                        # Show failed operations
                        for i, (success, error) in enumerate(results):
                            if not success:
                                print(f"   ❌ Operation {i+1}: {error}")
                        
                        if successful > 0:
                            return "PARTIAL_SUCCESS"
                        else:
                            print(f"❌ No operations succeeded with {provider_name}")
                            continue
                else:
                    print(f"❌ Could not extract intents from {provider_name} response")
                    print("📝 Raw response preview:")
                    print(response[:200] + "..." if len(response) > 200 else response)
                    continue
                    
            except Exception as e:
                print(f"❌ Error with {provider_name}: {str(e)}")
                continue
        
        print("\n💥 All LLM providers failed to generate usable intents")
        return None

    def generate_single_file(self, file_path: str, file_description: str, context: str = "") -> Optional[str]:
        """
        Generate a single file using AI - more reliable than multi-file generation.
        
        Args:
            file_path: Path of the file to generate (e.g., "app/page.tsx")
            file_description: Description of what this file should contain
            context: Additional context about the app and related files
            
        Returns:
            Generated file content or None if failed
        """
        print(f"🎯 Generating single file: {file_path}")
        
        # Determine file type and create appropriate prompt
        file_type = self._get_file_type(file_path)
        
        single_file_prompt = f"""You are creating a single file for a NextJS application.

FILE TO CREATE: {file_path}
FILE DESCRIPTION: {file_description}

CONTEXT:
{context}

REQUIREMENTS:
1. Generate ONLY the content for {file_path}
2. Make it complete and functional
3. Follow NextJS 13+ App Router patterns
4. Use TypeScript with proper types
5. Use shadcn/ui components where appropriate
6. Include proper imports and exports

FILE TYPE GUIDELINES:
{self._get_file_type_guidelines(file_type)}

Return ONLY the file content - no explanations, no tags, just the raw code that should go in {file_path}.
"""

        try:
            messages = [
                {"role": "system", "content": self.get_single_file_system_prompt()},
                {"role": "user", "content": single_file_prompt}
            ]
            
            response = self.generate_with_fallback(messages, context="single_file")
            
            if response:
                # Clean the response - remove any markdown code blocks or explanations
                cleaned_content = self._clean_single_file_response(response)
                print(f"✅ Generated {file_path} ({len(cleaned_content)} characters)")
                return cleaned_content
            else:
                print(f"❌ Failed to generate {file_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error generating {file_path}: {str(e)}")
            return None

    def _get_file_type(self, file_path: str) -> str:
        """Determine the type of file based on its path."""
        if file_path.endswith('.tsx'):
            if 'page.tsx' in file_path:
                return 'page'
            elif 'layout.tsx' in file_path:
                return 'layout'
            elif file_path.startswith('components/'):
                return 'component'
            else:
                return 'tsx'
        elif file_path.endswith('.ts'):
            if file_path.startswith('types/'):
                return 'types'
            elif file_path.startswith('lib/') or file_path.startswith('utils/'):
                return 'utility'
            elif file_path.startswith('data/'):
                return 'data'
            else:
                return 'ts'
        else:
            return 'other'

    def _get_file_type_guidelines(self, file_type: str) -> str:
        """Get specific guidelines for different file types."""
        guidelines = {
            'page': """
- Use 'use client' directive for interactive pages
- Export default function component
- Implement proper loading and error states
- Use shadcn/ui components for UI elements
- Include proper TypeScript types for props""",
            
            'component': """
- Use 'use client' directive if component has state/interactivity
- Export default function component
- Define proper TypeScript interfaces for props
- Use shadcn/ui components for consistency
- Include proper accessibility attributes""",
            
            'types': """
- Export all interfaces and types
- Use proper TypeScript conventions
- Include JSDoc comments for complex types
- Consider making types generic where appropriate""",
            
            'utility': """
- Export individual functions
- Include proper TypeScript types
- Add JSDoc comments for public functions
- Handle edge cases and errors gracefully""",
            
            'data': """
- Export const data arrays/objects
- Use proper TypeScript types
- Include realistic, diverse mock data
- Consider data relationships and consistency"""
        }
        
        return guidelines.get(file_type, "Follow NextJS and TypeScript best practices.")

    def _clean_single_file_response(self, response: str) -> str:
        """Clean AI response to extract just the file content."""
        # Remove markdown code blocks
        if '```' in response:
            # Find the largest code block
            import re
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
            if code_blocks:
                # Return the largest code block (most likely the actual content)
                return max(code_blocks, key=len).strip()
        
        # If no code blocks, return the response as-is (cleaned)
        lines = response.split('\n')
        
        # Skip any explanatory text at the beginning
        start_index = 0
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['import', 'export', 'const', 'function', 'interface', 'type', "'use client'"]):
                start_index = i
                break
        
        return '\n'.join(lines[start_index:]).strip()

    def generate_app_by_files(self, app_idea: str, file_plan: List[Dict[str, str]]) -> bool:
        """
        Generate an app by creating files one at a time.
        
        Args:
            app_idea: The original app idea
            file_plan: List of dicts with 'path', 'description', and 'dependencies'
            
        Returns:
            True if all files were generated successfully
        """
        print(f"📁 Generating app with {len(file_plan)} files...")
        
        generated_files = {}
        failed_files = []
        
        # Sort files by dependencies (independent files first)
        sorted_files = self._sort_files_by_dependencies(file_plan)
        
        for i, file_info in enumerate(sorted_files, 1):
            file_path = file_info['path']
            description = file_info['description']
            
            print(f"\n[{i}/{len(sorted_files)}] Creating {file_path}...")
            
            # Build context from previously generated files
            context = self._build_context_for_file(file_info, generated_files, app_idea)
            
            # Generate the file
            content = self.generate_single_file(file_path, description, context)
            
            if content:
                generated_files[file_path] = content
                print(f"✅ Generated {file_path}")
            else:
                failed_files.append(file_path)
                print(f"❌ Failed to generate {file_path}")
        
        if failed_files:
            print(f"\n⚠️ Failed to generate {len(failed_files)} files:")
            for file_path in failed_files:
                print(f"   ❌ {file_path}")
            return False
        
        # Write all files to disk
        return self._write_generated_files(generated_files)

    def _sort_files_by_dependencies(self, file_plan: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Sort files so dependencies are created before dependents."""
        # Simple heuristic: types first, then data, then utils, then components, then pages
        priority_order = {
            'types/': 0,
            'data/': 1,
            'lib/': 2,
            'utils/': 2,
            'components/': 3,
            'app/': 4
        }
        
        def get_priority(file_info):
            path = file_info['path']
            for prefix, priority in priority_order.items():
                if path.startswith(prefix):
                    return priority
            return 5  # Default priority
        
        return sorted(file_plan, key=get_priority)

    def _build_context_for_file(self, file_info: Dict[str, str], generated_files: Dict[str, str], app_idea: str) -> str:
        """Build context for generating a specific file."""
        context = f"APP IDEA: {app_idea}\n\n"
        
        # Add relevant already-generated files as context
        dependencies = file_info.get('dependencies', [])
        
        for dep in dependencies:
            if dep in generated_files:
                context += f"=== {dep} ===\n{generated_files[dep]}\n\n"
        
        # Add any type definitions that might be relevant
        for file_path, content in generated_files.items():
            if file_path.startswith('types/') and file_path != file_info['path']:
                context += f"=== {file_path} ===\n{content}\n\n"
        
        return context

    def _write_generated_files(self, generated_files: Dict[str, str]) -> bool:
        """
        Write generated files to disk.
        
        Args:
            generated_files: Dict mapping file paths to their content
            
        Returns:
            True if all files were written successfully
        """
        print(f"\n📝 Writing {len(generated_files)} files to disk...")
        
        app_path = Path(self.get_app_path())
        failed_files = []
        
        for file_path, content in generated_files.items():
            try:
                target_file = app_path / file_path
                
                # Create parent directories if they don't exist
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Write the file
                target_file.write_text(content)
                
                print(f"✅ Written: {file_path}")
                
            except Exception as e:
                print(f"❌ Failed to write {file_path}: {str(e)}")
                failed_files.append(file_path)
        
        if failed_files:
            print(f"\n⚠️ Failed to write {len(failed_files)} files:")
            for file_path in failed_files:
                print(f"   ❌ {file_path}")
            return False
        
        print(f"\n✅ All {len(generated_files)} files written successfully!")
        return True

    def generate_file_plan(self, app_idea: str) -> List[Dict[str, str]]:
        """
        Generate a plan of files to create for an app idea.
        
        Args:
            app_idea: The app idea description
            
        Returns:
            List of file plans with path, description, and dependencies
        """
        print("📋 Generating file plan...")
        
        planning_prompt = f"""Create a file plan for a NextJS application based on this idea:

APP IDEA: {app_idea}

Analyze the app idea and create a detailed plan of files that need to be created. Consider:
1. What type of app this is (simple utility, complex multi-page, etc.)
2. What components and pages are needed
3. What data structures and types are required
4. What utilities and hooks might be needed

Return a JSON array of file plans in this exact format:

{{
  "files": [
    {{
      "path": "types/talent.ts",
      "description": "TypeScript interfaces for talent profiles and user interactions",
      "dependencies": [],
      "priority": 1
    }},
    {{
      "path": "data/mockTalents.ts", 
      "description": "Mock data array with diverse talent profiles for the app",
      "dependencies": ["types/talent.ts"],
      "priority": 2
    }},
    {{
      "path": "lib/hooks.ts",
      "description": "Custom React hooks for localStorage and talent browsing state management", 
      "dependencies": ["types/talent.ts"],
      "priority": 2
    }},
    {{
      "path": "components/TalentCard.tsx",
      "description": "Interactive card component for displaying talent profiles with like/dislike buttons",
      "dependencies": ["types/talent.ts"],
      "priority": 3
    }},
    {{
      "path": "app/page.tsx",
      "description": "Main application page integrating all components for the talent discovery app",
      "dependencies": ["components/TalentCard.tsx", "lib/hooks.ts", "data/mockTalents.ts"],
      "priority": 4
    }}
  ]
}}

GUIDELINES:
- Start with types and interfaces (priority 1)
- Then data and utilities (priority 2) 
- Then components (priority 3)
- Finally pages that use everything (priority 4)
- Include realistic descriptions of what each file should contain
- List dependencies accurately
- For simple apps, keep it minimal (3-5 files)
- For complex apps, include more components and pages
- Always include app/page.tsx as the main page
"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert NextJS architect. Always return valid JSON."},
                {"role": "user", "content": planning_prompt}
            ]
            
            response = self.generate_with_fallback(messages, context="file_plan")
            
            if response:
                # Parse the JSON response
                file_plan = self._parse_file_plan_response(response)
                print(f"✅ Generated plan for {len(file_plan)} files")
                return file_plan
            else:
                print("❌ Failed to generate file plan")
                return self._get_fallback_file_plan(app_idea)
                
        except Exception as e:
            print(f"❌ Error generating file plan: {str(e)}")
            return self._get_fallback_file_plan(app_idea)

    def _parse_file_plan_response(self, response: str) -> List[Dict[str, str]]:
        """Parse AI response for file plan."""
        try:
            import json
            import re
            
            # Find JSON in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                if 'files' in data:
                    return data['files']
                    
        except Exception as e:
            print(f"⚠️ Could not parse file plan response: {e}")
        
        return []

    def _get_fallback_file_plan(self, app_idea: str) -> List[Dict[str, str]]:
        """Get a fallback file plan if AI generation fails."""
        print("🔄 Using fallback file plan...")
        
        # Determine app type and provide appropriate fallback
        app_lower = app_idea.lower()
        
        if any(word in app_lower for word in ['tinder', 'swipe', 'card', 'profile']):
            # Talent/dating app style
            return [
                {
                    "path": "types/profile.ts",
                    "description": "TypeScript interfaces for user profiles and interactions",
                    "dependencies": [],
                    "priority": 1
                },
                {
                    "path": "data/mockData.ts",
                    "description": "Mock data for profiles",
                    "dependencies": ["types/profile.ts"],
                    "priority": 2
                },
                {
                    "path": "lib/hooks.ts",
                    "description": "Custom hooks for state management",
                    "dependencies": ["types/profile.ts"],
                    "priority": 2
                },
                {
                    "path": "components/ProfileCard.tsx",
                    "description": "Card component for displaying profiles",
                    "dependencies": ["types/profile.ts"],
                    "priority": 3
                },
                {
                    "path": "app/page.tsx",
                    "description": "Main application page",
                    "dependencies": ["components/ProfileCard.tsx", "lib/hooks.ts", "data/mockData.ts"],
                    "priority": 4
                }
            ]
        else:
            # Generic app fallback
            return [
                {
                    "path": "types/index.ts",
                    "description": "TypeScript interfaces and types",
                    "dependencies": [],
                    "priority": 1
                },
                {
                    "path": "app/page.tsx",
                    "description": "Main application page",
                    "dependencies": ["types/index.ts"],
                    "priority": 4
                }
            ]

    def generate_app_with_single_files(self, app_idea: str) -> Optional[str]:
        """
        Generate an app using the improved single-file approach with proper planning and context.
        
        This method:
        1. Enhances the initial request for better context
        2. Creates a smart, focused file plan (1 API call)
        3. Generates files one by one with context from previous files
        4. Keeps file count reasonable (5-8 files max)
        5. Ensures proper dependency order
        """
        print("🚀 Generating app using improved single-file approach...")
        print("   ✨ Step 0: Enhanced request processing")
        print("   🧠 Step 1: Smart planning with focused file list")
        print("   🔗 Step 2: Context-aware file generation")
        print("   📦 Step 3: Proper dependency ordering")
        
        # Step 0: Enhance the request for better context
        print("\n✨ Enhancing app idea with additional context...")
        try:
            from .request_enhancer import RequestEnhancer
            request_enhancer = RequestEnhancer()
            
            enhanced_request = request_enhancer.enhance_app_idea(app_idea)
            
            if enhanced_request and enhanced_request != app_idea:
                print(f"✅ App idea enhanced with technical specifications")
                print(f"   📝 Original: {app_idea[:80]}...")
                print(f"   🎯 Enhanced: {enhanced_request[:100]}...")
                app_idea = enhanced_request
            else:
                print("ℹ️ App idea enhancement not needed")
                
        except Exception as e:
            print(f"⚠️ App idea enhancement failed: {str(e)}")
            print("   Continuing with original app idea...")
        
        # Step 1: Generate a FOCUSED file plan (1 API call)
        print("\n📋 Creating focused file plan...")
        file_plan = self.generate_focused_file_plan(app_idea)
        
        if not file_plan:
            print("❌ Could not create file plan")
            return None
        
        # Limit to reasonable number of files
        if len(file_plan) > 8:
            print(f"⚠️ Reducing file count from {len(file_plan)} to 8 most essential files")
            file_plan = file_plan[:8]
        
        print(f"📋 Focused plan created with {len(file_plan)} essential files:")
        for i, file_info in enumerate(file_plan, 1):
            print(f"   {i}. 📄 {file_info['path']} - {file_info['description'][:60]}...")
        
        # Step 2: Generate files with proper context
        print(f"\n🎯 Generating {len(file_plan)} files with context...")
        success = self.generate_files_with_context(app_idea, file_plan)
        
        if success:
            print("✅ All files generated successfully!")
            return "SUCCESS"
        else:
            print("❌ Some files failed to generate")
            return None

    def generate_focused_file_plan(self, app_idea: str) -> List[Dict[str, str]]:
        """
        Generate a focused file plan that prioritizes essential files only.
        
        This creates a minimal, working app with just the essential files.
        """
        print("📋 Generating focused file plan...")
        
        focused_planning_prompt = f"""Create a MINIMAL file plan for a working NextJS app based on this idea:

APP IDEA: {app_idea}

CRITICAL REQUIREMENTS:
1. Keep it to 5-8 files maximum for efficiency
2. Focus on essential files only to create a working app
3. Start with types, then 1-2 key components, then main page
4. No complex features - just core functionality
5. Use shadcn/ui components

Return a JSON array with this EXACT format:

{{
  "files": [
    {{
      "path": "types/index.ts",
      "description": "Essential TypeScript interfaces for the app",
      "dependencies": [],
      "priority": 1
    }},
    {{
      "path": "components/MainComponent.tsx", 
      "description": "Primary component that handles core app functionality",
      "dependencies": ["types/index.ts"],
      "priority": 2
    }},
    {{
      "path": "app/page.tsx",
      "description": "Main page that integrates the components",
      "dependencies": ["components/MainComponent.tsx"],
      "priority": 3
    }}
  ]
}}

Focus on the MINIMUM viable files to create a working app. Quality over quantity."""

        try:
            messages = [
                {"role": "system", "content": "You are an expert NextJS architect. Always return focused, minimal JSON plans."},
                {"role": "user", "content": focused_planning_prompt}
            ]
            
            response = self.generate_with_fallback(messages, context="file_plan")
            
            if response:
                file_plan = self._parse_file_plan_response(response)
                print(f"✅ Generated focused plan for {len(file_plan)} files")
                return file_plan
            else:
                print("❌ Failed to generate focused file plan")
                return self._get_minimal_fallback_plan(app_idea)
                
        except Exception as e:
            print(f"❌ Error generating focused file plan: {str(e)}")
            return self._get_minimal_fallback_plan(app_idea)

    def generate_files_with_context(self, app_idea: str, file_plan: List[Dict[str, str]]) -> bool:
        """
        Generate files one by one with proper context from previously generated files.
        
        This ensures each file has context from dependencies and follows proper patterns.
        """
        print(f"📁 Generating {len(file_plan)} files with context...")
        
        generated_files = {}
        context_history = f"APP IDEA: {app_idea}\n\n"
        
        # Sort files by priority/dependencies
        sorted_files = sorted(file_plan, key=lambda x: x.get('priority', 99))
        
        for i, file_info in enumerate(sorted_files, 1):
            file_path = file_info['path']
            description = file_info['description']
            
            print(f"\n[{i}/{len(sorted_files)}] 🎯 Generating {file_path}...")
            
            # Build rich context from previous files
            rich_context = self._build_rich_context_for_file(
                file_info, generated_files, context_history, app_idea
            )
            
            # Generate the file with context
            content = self.generate_single_file_with_rich_context(
                file_path, description, rich_context
            )
            
            if content:
                generated_files[file_path] = content
                # Add this file to context for next files
                context_history += f"\n=== {file_path} ===\n{content[:500]}...\n"
                print(f"✅ Generated {file_path} ({len(content)} chars)")
            else:
                print(f"❌ Failed to generate {file_path}")
                return False
        
        # Write all files to disk
        print(f"\n📝 Writing {len(generated_files)} files to disk...")
        return self._write_generated_files(generated_files)

    def _build_rich_context_for_file(self, file_info: Dict[str, str], generated_files: Dict[str, str], 
                                   context_history: str, app_idea: str) -> str:
        """Build rich context for generating a specific file."""
        
        context = f"""RICH CONTEXT FOR FILE GENERATION

APP IDEA: {app_idea}

FILE TO GENERATE: {file_info['path']}
DESCRIPTION: {file_info['description']}

PREVIOUSLY GENERATED FILES:
{context_history}

DEPENDENCIES: {', '.join(file_info.get('dependencies', []))}

INTEGRATION REQUIREMENTS:
- This file should work seamlessly with the previously generated files
- Follow the same patterns and coding style established
- Use proper imports from the generated dependencies
- Maintain consistency with the overall app architecture"""

        return context

    def generate_single_file_with_rich_context(self, file_path: str, description: str, rich_context: str) -> Optional[str]:
        """Generate a single file with rich context from previously generated files."""
        
        file_type = self._get_file_type(file_path)
        
        contextual_prompt = f"""You are generating a specific file for a NextJS application with full context.

{rich_context}

SPECIFIC TASK:
Generate the content for: {file_path}

REQUIREMENTS:
1. Generate ONLY the raw file content (no explanations, no markdown)
2. Use the context from previously generated files to ensure consistency
3. Follow established patterns and imports from the context
4. Make it complete and functional
5. Use proper TypeScript and NextJS patterns

FILE TYPE GUIDELINES:
{self._get_file_type_guidelines(file_type)}

Return ONLY the file content that should go in {file_path}."""

        try:
            messages = [
                {"role": "system", "content": self.get_single_file_system_prompt()},
                {"role": "user", "content": contextual_prompt}
            ]
            
            response = self.generate_with_fallback(messages, context="single_file")
            
            if response:
                cleaned_content = self._clean_single_file_response(response)
                return cleaned_content
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error generating {file_path}: {str(e)}")
            return None

    def _get_minimal_fallback_plan(self, app_idea: str) -> List[Dict[str, str]]:
        """Fallback plan with minimal essential files."""
        return [
            {
                "path": "types/index.ts",
                "description": "Essential TypeScript interfaces",
                "dependencies": [],
                "priority": 1
            },
            {
                "path": "components/MainComponent.tsx",
                "description": "Primary component for the app",
                "dependencies": ["types/index.ts"],
                "priority": 2
            },
            {
                "path": "app/page.tsx",
                "description": "Main page that uses the component",
                "dependencies": ["components/MainComponent.tsx"],
                "priority": 3
            }
        ]

    def get_single_file_system_prompt(self):
        """Get system prompt specifically for single-file generation (no <new> tags)."""
        return """You are an expert NextJS frontend developer creating individual files for a NextJS application.

🚨 CRITICAL JSX SYNTAX RULES 🚨

1. **RETURN STATEMENT SYNTAX**: ALWAYS end JSX return statements with closing parenthesis `)`, NEVER with curly brace `}`
   
   ✅ CORRECT JSX RETURN:
   ```javascript
   return (
     <div className="container">
       <h1>Title</h1>
       <p>Content</p>
     </div>
   )  // ← ENDS WITH CLOSING PARENTHESIS
   ```
   
   ❌ WRONG JSX RETURN:
   ```javascript
   return (
     <div className="container">
       <h1>Title</h1>
     </div>
   }  // ← WRONG! NEVER USE CURLY BRACE TO END RETURN
   ```

2. **FRONTEND-ONLY FOCUS**: 
   - NO authentication systems (no NextAuth, no login/logout)
   - NO database connections (no Prisma, no MongoDB, no SQL)
   - NO backend API routes (no pages/api/ unless absolutely necessary)
   - NO server-side functionality beyond static generation
   - Use React state, localStorage, or sessionStorage for data management
   - Focus on rich UI/UX and interactive frontend features

3. **STYLING**: Use shadcn/ui components with Tailwind CSS:
   ✅ CORRECT: Import and use shadcn components: `import { Button } from "@/components/ui/button"`
   ✅ CORRECT: Use shadcn semantic variants: `<Button variant="outline" size="lg">Click me</Button>`
   ✅ CORRECT: Combine with Tailwind: `<Button className="w-full mt-4">Submit</Button>`

4. **OUTPUT FORMAT**: Return ONLY the raw file content - no explanations, no markdown code blocks, no <new> tags, just the actual code that should go in the file.

🎯 **GOAL**: Create a single, complete, functional file that integrates well with a NextJS application."""

    def generate_app_with_anti_truncation(self, app_idea: str) -> Optional[str]:
        """
        Generate app using anti-truncation strategy - shorter, focused prompt to prevent cutoffs.
        
        This approach uses the original single-call method but with a much shorter,
        more focused prompt to prevent AI response truncation.
        """
        print("🎯 Using anti-truncation single-call approach...")
        
        # Create a much shorter, focused prompt
        focused_prompt = f"""Create a NextJS app for: {app_idea}

REQUIREMENTS:
- Use <new filename="path/file.ext"> syntax
- Start with essential files only: types, 1-2 components, main page
- Keep responses under 300 lines total
- Use shadcn/ui components
- TypeScript + Tailwind CSS
- No auth, no backend, frontend-only

Generate ONLY the essential files to make a working app. Keep it minimal and focused."""

        try:
            messages = [
                {"role": "system", "content": self.get_focused_system_prompt()},
                {"role": "user", "content": focused_prompt}
            ]
            
            return self.generate_with_fallback(messages, context="create")
            
        except Exception as e:
            print(f"❌ Error in anti-truncation generation: {str(e)}")
            return None

    def get_focused_system_prompt(self):
        """Shorter system prompt to prevent truncation."""
        return """You are a NextJS developer. Create minimal, working apps using <new filename=""> syntax.

KEY RULES:
1. KEEP RESPONSES SHORT (under 300 lines total)
2. Create only essential files: 1-2 types, 1-2 components, main page
3. Use shadcn/ui: Button, Card, Input
4. Frontend-only, no auth, no backend
5. Working code that builds successfully

SYNTAX:
<new filename="types/index.ts">
// content here  
</new>

Focus on minimal viable functionality."""

    def build_and_fix_errors(self, auto_install_deps: bool = True, max_retries: int = 8) -> bool:
        """
        Build the app and automatically fix any errors found.
        
        Args:
            auto_install_deps: Whether to automatically install dependencies
            max_retries: Maximum number of fix attempts
            
        Returns:
            True if build succeeds or errors are fixed, False otherwise
        """
        print("🔨 Building and validating NextJS app...")
        
        try:
            import subprocess
            import re
            app_path = self.get_app_path()
            
            previous_errors = None
            consecutive_same_errors = 0
            
            for attempt in range(max_retries):
                # Try a build
                print(f"   📦 Running npm run build (attempt {attempt + 1}/{max_retries})...")
                result = subprocess.run(
                    ["npm", "run", "build"], 
                    cwd=app_path, 
                    capture_output=True, 
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    print("✅ Build successful!")
                    return True
                
                current_errors = result.stderr
                print("⚠️ Build failed, attempting to fix errors...")
                print(f"   📝 Build errors: {current_errors[:500]}...")
                
                # Check if we're making progress (different errors)
                if previous_errors and current_errors == previous_errors:
                    consecutive_same_errors += 1
                    print(f"   ⚠️ Same errors as previous attempt ({consecutive_same_errors}/3)")
                    if consecutive_same_errors >= 3:
                        print("   🛑 No progress after 3 attempts with same errors")
                        print("   🔍 Need more sophisticated debugging approach")
                        break
                else:
                    consecutive_same_errors = 0  # Reset counter
                    if previous_errors:
                        print("   ✅ Different errors detected - making progress")
                
                previous_errors = current_errors
                
                # Check for shadcn UI component issues
                missing_ui_components = self._extract_missing_shadcn_components(result.stderr)
                if missing_ui_components:
                    print(f"   🎨 Installing missing shadcn components: {', '.join(missing_ui_components)}")
                    shadcn_result = subprocess.run(
                        ["npx", "shadcn@latest", "add"] + missing_ui_components + ["--yes"], 
                        cwd=app_path, 
                        capture_output=True, 
                        text=True
                    )
                    if shadcn_result.returncode == 0:
                        print("✅ Shadcn components installed successfully!")
                        continue  # Retry build
                    else:
                        print(f"❌ Failed to install shadcn components: {shadcn_result.stderr}")
                
                # Use smart build error analyzer for all build errors
                print("   🧠 Using smart build error analysis...")
                try:
                    from .build_error_analyzer import SmartBuildErrorAnalyzer
                    from .smart_task_executor import SmartTaskExecutor
                    
                    # Analyze the error with full context
                    analyzer = SmartBuildErrorAnalyzer(app_path)
                    analysis = analyzer.analyze_build_error(result.stderr)
                    
                    print(f"   📋 Analysis: {analysis.error_summary}")
                    print(f"   🎯 Root cause: {analysis.root_cause}")
                    print(f"   📊 Confidence: {analysis.confidence:.1%}")
                    
                    if analysis.confidence > 0.4 and analysis.tasks:
                        # Execute the fix tasks
                        executor = SmartTaskExecutor(app_path)
                        success, task_results = executor.execute_analysis(analysis)
                        
                        if success:
                            print("✅ Smart build error fix applied and validated successfully!")
                            continue  # Retry build
                        else:
                            print("❌ Smart build error fix failed or didn't resolve the errors")
                            execution_summary = executor.get_execution_summary()
                            print(executor.get_execution_summary_text())
                            
                            # Check if this is an executor bug vs a real build error
                            failed_tasks = execution_summary.get('failed_tasks', [])
                            executor_bugs = []
                            
                            for task_info in failed_tasks:
                                error_msg = task_info.get('error', '').lower()
                                if any(bug_indicator in error_msg for bug_indicator in [
                                    'is not defined', 'import error', 'module not found', 
                                    'name error', 'attribute error'
                                ]):
                                    executor_bugs.append(task_info)
                            
                            if executor_bugs:
                                print("🐛 Detected executor bugs that need fixing:")
                                for bug in executor_bugs:
                                    print(f"   • {bug.get('task_id', 'unknown')}: {bug.get('error', 'unknown error')}")
                                print("🔧 System needs debugging before continuing with build fixes")
                                print("❌ Stopping build attempts until executor issues are resolved")
                                return False
                            else:
                                # Real build errors, continue with partial fixes
                                print("🔄 Continuing with partial fixes in case they help...")
                                continue
                    else:
                        print(f"❌ Low confidence analysis ({analysis.confidence:.1%}), trying fallback...")
                        
                except Exception as e:
                    print(f"❌ Smart error analysis failed: {e}")
                
                # Fallback: Check for specific dependency issues
                if "Cannot resolve module" in result.stderr or "Module not found" in result.stderr:
                    print("   🔧 Fallback: Attempting basic dependency fix...")
                    if auto_install_deps:
                        try:
                            from .dependency_manager import DependencyManager
                            dependency_manager = DependencyManager(app_path)
                            if dependency_manager.auto_manage_dependencies():
                                print("✅ Dependencies analyzed and installed successfully!")
                                continue  # Retry build
                            else:
                                print("⚠️ Dependency management completed with warnings")
                                continue  # Still retry build
                        except Exception as e:
                            print(f"❌ Error running dependency management: {e}")
                            # Fallback to basic npm install
                            dep_result = subprocess.run(
                                ["npm", "install"], 
                                cwd=app_path, 
                                capture_output=True, 
                                text=True
                            )
                            if dep_result.returncode == 0:
                                print("✅ Basic npm install completed!")
                                continue  # Retry build
                
                # If we get here, we couldn't fix the issue
                break
            
            # Provide detailed feedback based on what happened
            if consecutive_same_errors >= 3:
                print("❌ Build validation failed: Same errors persisting despite multiple fix attempts")
                print("💡 Suggestions:")
                print("   • The smart fixes may not be targeting the right issues")
                print("   • Manual code review may be needed")
                print("   • Check if the error messages point to specific files/lines")
            else:
                print(f"❌ Build validation failed after {max_retries} retry attempts")
                print("💡 The system tried multiple smart fixes but couldn't resolve all issues")
            
            # Show final error state for debugging
            if previous_errors:
                print("\n📋 Final error state:")
                print(f"   {previous_errors[:300]}{'...' if len(previous_errors) > 300 else ''}")
            
            return False
                
        except subprocess.TimeoutExpired:
            print("⚠️ Build timeout - skipping validation")
            return False
        except Exception as e:
            print(f"⚠️ Build validation error: {str(e)}")
            return False

    def _extract_missing_shadcn_components(self, error_output: str) -> List[str]:
        """Extract missing shadcn UI component names from build error output."""
        import re
        
        # Look for patterns like "Can't resolve '@/components/ui/checkbox'"
        pattern = r"Can't resolve '@/components/ui/(\w+)'"
        matches = re.findall(pattern, error_output)
        
        # Filter to known shadcn components
        known_components = {
            'button', 'input', 'card', 'checkbox', 'badge', 'textarea', 'select',
            'dialog', 'dropdown-menu', 'popover', 'tooltip', 'alert', 'progress',
            'tabs', 'accordion', 'avatar', 'calendar', 'command', 'form', 'label',
            'menubar', 'navigation-menu', 'radio-group', 'scroll-area', 'separator',
            'sheet', 'skeleton', 'slider', 'switch', 'table', 'toast', 'toggle'
        }
        
        missing_components = [comp for comp in matches if comp in known_components]
        return list(set(missing_components))  # Remove duplicates


# Backward compatibility alias
NextJSAppBuilder = MultiLLMAppBuilder
