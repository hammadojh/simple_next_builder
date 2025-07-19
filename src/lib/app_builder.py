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


class MultiLLMAppBuilder:
    def __init__(self, openai_api_key=None, openrouter_api_key=None):
        """Initialize the app builder with multiple LLM providers."""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        
        # Initialize OpenAI client if key is available
        self.openai_client = None
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            except ImportError:
                print("⚠️ OpenAI package not installed")
        
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
        elif context == "edit_fallback":
            # For fallback editing, expect <edit> tags
            edit_tags = re.findall(r'<edit filename=\"([^\"]+)\"', content)
            if not edit_tags:
                issues.append("No <edit> tags found in response for fallback editing")
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
        elif context == "edit_fallback" and '<edit filename=' in content:
            good_practices.append("Uses appropriate line-based edit tags")
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
        """Call OpenAI API."""
        if not self.openai_client:
            return None
        
        try:
            response = self.openai_client.chat.completions.create(
                model=config["model"],
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI error: {str(e)}")
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
        print("🚀 Generating NextJS application with multi-LLM fallback...")
        
        for i, config in enumerate(self.llm_providers, 1):
            try:
                print(f"🤖 Trying {config['name']} ({i}/{len(self.llm_providers)})...")
                
                # Call the appropriate provider
                if config["type"] == "openai" and self.openai_client:
                    response = self.call_openai(messages, config)
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
        
        try:
            # Change to app directory
            original_dir = os.getcwd()
            os.chdir(app_path)
            
            # Install dependencies if requested
            if auto_install_deps:
                print("📦 Installing dependencies...")
                result = subprocess.run(['npm', 'install'], 
                                      capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    print(f"⚠️ npm install warnings: {result.stderr}")
            
            # Try to build the app with enhanced error capture
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
            # Return to original directory
            os.chdir(original_dir)
    
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
        Get context by directly reading key files with enhanced structural analysis.
        
        🔧 FIXED: No longer includes full file content to prevent massive token consumption.
        Instead provides structural summaries and relevant excerpts.
        """
        print(f"📁 Getting direct file context from: {app_directory}")
        
        context_parts = []
        context_parts.append("CURRENT APP FILES FOR EDITING:")
        context_parts.append("=" * 60)
        context_parts.append(f"User Request: {user_request}")
        context_parts.append("=" * 60)
        
        from pathlib import Path
        import re
        app_path = Path(app_directory)
        
        # Read the key files directly with structural analysis
        key_files = ["app/page.tsx", "app/layout.tsx", "app/globals.css", "package.json"]
        files_found = 0
        total_chars = 0  # Track context size
        MAX_CONTEXT_CHARS = 50000  # Limit to ~12-15k tokens
        
        for file_path in key_files:
            full_path = app_path / file_path
            if full_path.exists():
                try:
                    content = full_path.read_text()
                    context_parts.append(f"\n📄 FILE: {file_path}")
                    context_parts.append("-" * 40)
                    
                    # Add structural analysis for TypeScript/JavaScript files
                    if file_path.endswith(('.tsx', '.ts', '.js', '.jsx')):
                        context_parts.append("🏗️ CODE STRUCTURE:")
                        
                        # Extract interfaces
                        interfaces = re.findall(r'interface\s+(\w+)\s*{[^}]*}', content, re.DOTALL)
                        if interfaces:
                            context_parts.append(f"  📋 Interfaces: {', '.join(interfaces)}")
                        
                        # Extract variable declarations
                        variables = re.findall(r'(?:const|let|var)\s+(\w+)\s*[=:]', content)
                        if variables:
                            context_parts.append(f"  📦 Variables: {', '.join(set(variables))}")
                        
                        # Extract function/component names
                        functions = re.findall(r'(?:function\s+(\w+)|export\s+default\s+function\s+(\w+)|const\s+(\w+)\s*=.*=>)', content)
                        func_names = [name for group in functions for name in group if name]
                        if func_names:
                            context_parts.append(f"  ⚙️ Functions/Components: {', '.join(set(func_names))}")
                        
                        # Extract useState declarations
                        usestate_vars = re.findall(r'const\s+\[(\w+),\s*set\w+\]\s*=\s*useState', content)
                        if usestate_vars:
                            context_parts.append(f"  🎛️ State Variables: {', '.join(usestate_vars)}")
                        
                        # Extract imports for better context
                        imports = re.findall(r'import.*from\s+[\'"]([^\'"]+)[\'"]', content)
                        if imports:
                            context_parts.append(f"  📥 Key Imports: {', '.join(set(imports))}")
                        
                        # 🔧 SMART EXCERPT: Include only relevant parts, not full content
                        context_parts.append("\n📝 RELEVANT EXCERPT:")
                        excerpt = self._get_relevant_excerpt(content, user_request, max_lines=20)
                        context_parts.append(excerpt)
                        
                    else:
                        # For non-code files (CSS, JSON), include a smaller excerpt
                        context_parts.append("📝 FILE EXCERPT:")
                        lines = content.split('\n')
                        if len(lines) <= 10:
                            context_parts.append(content)  # Small files can be included fully
                        else:
                            # Include first and last few lines for context
                            excerpt_lines = lines[:5] + ["... (content truncated) ..."] + lines[-3:]
                            context_parts.append('\n'.join(excerpt_lines))
                    
                    # Check if we're approaching context limit
                    current_context = "\n".join(context_parts)
                    total_chars = len(current_context)
                    
                    if total_chars > MAX_CONTEXT_CHARS:
                        context_parts.append(f"\n⚠️ Context limit reached ({total_chars:,} chars) - truncating remaining files")
                        break
                        
                    files_found += 1
                    
                except Exception as e:
                    context_parts.append(f"❌ Error reading {file_path}: {str(e)}")
        
        if files_found == 0:
            context_parts.append("❌ No key files found in the app directory")
            return ""
        
        # Add context size information
        final_context = "\n".join(context_parts)
        estimated_tokens = len(final_context) // 4  # Rough estimate: 4 chars per token
        context_parts.append(f"\n📊 Context size: {len(final_context):,} characters (~{estimated_tokens:,} tokens)")
        
        print(f"✅ Retrieved {files_found} files with structural analysis")
        print(f"📊 Context size: {len(final_context):,} characters (~{estimated_tokens:,} tokens)")
        
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
        
        # Generate line-based edit
        prompt = f"""You are editing a NextJS app using line-based edits.

CURRENT APP CONTEXT:
{semantic_context}

USER REQUEST:
{app_idea}

Use ONLY <edit filename="..." start_line="N" end_line="N"> tags.
Make minimal, surgical changes. Be extremely careful with line numbers.
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
        """Edit app using intent-based approach (minimizes AI role in diff generation)."""
        from .intent_editor import IntentBasedEditor, parse_ai_intent_response, get_intent_based_prompt
        from pathlib import Path
        
        print(f"📝 Editing existing app with intent-based approach: {self.app_name}")
        print(f"🎯 Requested changes: {app_idea}")
        
        # Get semantic context
        semantic_context = self.get_semantic_context_for_request(
            user_request=app_idea, 
            app_directory=self.get_app_path()
        )
        if not semantic_context:
            print("❌ Failed to get semantic context for edit request")
            return False
        
        # Generate structured intents instead of raw diffs
        print("🤖 Generating structured editing intents...")
        intent_prompt = get_intent_based_prompt() + f"""

CURRENT APP CONTEXT:
{semantic_context}

USER REQUEST:
{app_idea}

Generate JSON intents to implement these changes precisely and safely.
"""
        
        is_valid, response = self.make_openai_request(intent_prompt, context="intent")
        
        if not is_valid:
            print("❌ Failed to generate valid intent response")
            return False
        
        # Parse intents from AI response
        intents = parse_ai_intent_response(response)
        if not intents:
            print("❌ No valid intents found in AI response")
            return False
        
        print(f"📋 Parsed {len(intents)} editing intents:")
        for i, intent in enumerate(intents, 1):
            print(f"   {i}. {intent.action} in {intent.file_path}: {intent.context}")
        
        # Apply intents
        print("🔧 Applying structured intents...")
        editor = IntentBasedEditor(Path(self.get_app_path()))
        results = editor.apply_intent_list(intents)
        
        # Check results
        all_success = all(success for success, _ in results)
        
        if all_success:
            print("✅ All intents applied successfully!")
        else:
            print("⚠️ Some intents failed:")
            for success, message in results:
                if not success:
                    print(f"   ❌ {message}")
        
        if not all_success:
            print("🔄 Falling back to diff-based approach...")
            return self._edit_app_with_diffs(app_idea)
        
        # Build and run to verify changes work
        print("🔨 Building app to verify changes...")
        build_success = self.build_and_run(auto_install_deps=True)
        
        if build_success:
            print("🎉 App edited successfully with intent-based approach!")
            return True
        else:
            print("⚠️ App edited but has build errors - attempting automatic fixes...")
            return self.auto_fix_build_errors()
    
    def _edit_app_with_diffs(self, app_idea):
        """Edit app using traditional diff-based approach (with sanitization)."""
        from .diff_builder import DiffBuilder
        
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
        diff_builder = DiffBuilder(diff_file, self.get_app_path())
        success = diff_builder.build()
        
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
                
            from .diff_builder import DiffBuilder
            diff_builder = DiffBuilder(diff_file, self.get_app_path())
            success = diff_builder.build()
            
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
        
        # Generate line-based edit response
        legacy_prompt = """You are editing an existing NextJS application. Here is the semantic context:

""" + semantic_context + """

User wants to make the following changes:
""" + app_idea + """

FALLBACK MODE - USE LINE-BASED EDITS:
Please use the legacy <edit filename="..." start_line="N" end_line="N"> format.
Be extremely careful with line numbers and avoid overlapping edits.

Make minimal, targeted changes to implement the requested features."""

        is_valid, response = self.make_openai_request(legacy_prompt, context="edit_fallback")  # Use special context for edit fallback
        
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
        Fix infrastructure-related errors.
        
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
            
            # Try npm install to fix dependency issues
            print("🔧 Running npm install to fix dependencies...")
            result = subprocess.run(['npm', 'install'], 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✅ npm install completed successfully")
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
            from .diff_builder import DiffBuilder
            diff_builder = DiffBuilder(fix_file, self.get_app_path())
            fix_success = diff_builder.build()
            
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

FALLBACK MODE - USE LINE-BASED EDITS:
Use the legacy <edit filename="..." start_line="N" end_line="N"> format.
Make minimal, surgical fixes to resolve the exact errors shown above.
Be extremely careful with line numbers.

Focus on fixing:
- Syntax errors (missing commas, brackets, quotes)
- JSX return statement issues
- Import/export problems
- Type errors"""

        is_valid, response = self.make_openai_request(legacy_fix_prompt, context="edit_fallback")
        
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
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            response = self.generate_with_fallback(messages, context=context)
            if not response:
                return False, ""
            
            # 🔍 DEBUG: Save and print the raw AI response
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
            is_valid, validation_feedback = self.validate_response(response, context)
            
            if not is_valid:
                print(f"❌ Response validation failed: {validation_feedback}")
                print(f"🔍 Full response saved to {debug_file} for analysis")
                return False, response
            
            return True, response
            
        except Exception as e:
            print(f"❌ Error making AI request: {str(e)}")
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


# Backward compatibility alias
NextJSAppBuilder = MultiLLMAppBuilder
