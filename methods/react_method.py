"""
ReAct (Reasoning and Acting) Method for GDPR Compliance Detection - FIXED VERSION

Fixed Issues:
1. ✅ Added missing attribute initialization (api_base, api_key, temperature)
2. ✅ Fixed async/await mixing issue (changed to synchronous calls)
3. ✅ Simplified agent construction, no external react_agent dependency
4. ✅ Maintained compatibility with original method
"""

import sys
import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC

from methods.base_method import BaseMethod
from methods.react_tools import GDPR_TOOLS

# Import ReAct components
try:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, MessagesState
    from langgraph.prebuilt import ToolNode
    
    REACT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ReAct components not available: {e}")
    REACT_AVAILABLE = False


class ReActMethod(BaseMethod):
    """
    ReAct agent method that uses iterative reasoning and tool calling
    to detect GDPR violations.
    
    This is a FIXED version that:
    - Properly initializes all required attributes
    - Uses synchronous invoke instead of async
    - Simplifies the agent graph construction
    """
    
    def initialize(self):
        """Initialize the ReAct agent with GDPR tools."""
        if not REACT_AVAILABLE:
            print("ReAct components not available, using fallback mode")
            self.agent = None
            return
        
        # FIX 1: Initialize all required attributes
        self.model = self.config.get('model', 'gpt-4o')
        self.api_base = self.config.get('api_base', 'https://api.openai.com/v1')
        self.api_key = self.config.get('api_key')
        self.temperature = self.config.get('temperature', 0.0)
        self.max_iterations = self.config.get('max_iterations', 10)
        self.timeout = self.config.get('timeout', 300)
        
        print(f"Initializing ReAct agent with model: {self.model}")
        print(f"API base: {self.api_base}")
        print(f"Temperature: {self.temperature}")
        
        # Build GDPR-specific ReAct agent
        try:
            self.agent = self._build_gdpr_react_agent()
            print("✅ ReAct agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize ReAct agent: {e}")
            self.agent = None
    
    def _build_gdpr_react_agent(self):
        """
        Build a simplified ReAct agent using LangGraph.
        
        FIX 2: Use synchronous version to avoid async/await mixing
        """
        
        # Create LLM with bound tools
        llm_with_tools = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_base=self.api_base,
            openai_api_key=self.api_key
        ).bind_tools(GDPR_TOOLS)
        
        # Define call_model node (synchronous version)
        def call_model(state: MessagesState) -> Dict[str, List[AIMessage]]:
            """Call the LLM with GDPR tools - SYNCHRONOUS VERSION"""
            
            # Prepare system message
            system_message = self._get_system_prompt().format(
                system_time=datetime.now(tz=UTC).isoformat()
            )
            
            # Build message list
            messages = [
                {"role": "system", "content": system_message}
            ] + state["messages"]
            
            # Synchronous call (not async)
            response = llm_with_tools.invoke(messages)
            
            return {"messages": [response]}
        
        # Define routing function
        def should_continue(state: MessagesState) -> str:
            """Determine whether to continue or end."""
            last_message = state["messages"][-1]
            
            # If there are tool calls, continue to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            
            # Otherwise end
            return "__end__"
        
        # Create graph
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(GDPR_TOOLS))
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "__end__": "__end__"
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        # Compile graph
        return workflow.compile()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for GDPR analysis."""
        return """You are a GDPR compliance expert analyzing code for privacy violations.

**Your Task:**
Identify which GDPR articles are violated by the given code.

**Available Tools:**
1. `gdpr_lookup(article_number)` - Look up GDPR article definitions
2. `code_search(keyword)` - Search for sensitive API calls in code
3. `rule_check(code_snippet)` - Check code against formal rules

**Analysis Process:**
1. Examine the code for sensitive operations (data collection, network, storage)
2. Use `code_search` to find specific API patterns
3. Use `rule_check` to detect known violation patterns
4. Use `gdpr_lookup` to verify article requirements
5. Reason about which articles are violated

**CRITICAL OUTPUT FORMAT:**
After your analysis, you MUST end your response with ONLY the violated article numbers.
Format: Comma-separated integers on the last line
Example final lines:
"Based on the analysis above, the violated articles are: 6,7,32"
or 
"No violations found: 0"

Current time: {system_time}"""
    
    def _extract_articles_from_response(self, response: str) -> List[int]:
        """
        Extract GDPR article numbers from agent response.
        
        Looks for patterns in the text, prioritizing the last line.
        """
        if not response:
            return [0]
        
        # Strategy 1: Find pure digits in the last line
        lines = response.strip().split('\n')
        for line in reversed(lines[-3:]):  # Check last 3 lines
            line = line.strip()
            # Match lines with only digits and commas (e.g., "6,7,32")
            if re.match(r'^[\d,\s]+$', line):
                numbers = [int(n.strip()) for n in line.split(',') if n.strip().isdigit()]
                if numbers:
                    return numbers
        
        # Strategy 2: Find "violated articles are: X,Y,Z" pattern
        match = re.search(r'violated articles? (?:are|is)[:\s]+(\d+(?:\s*,\s*\d+)*)', response, re.IGNORECASE)
        if match:
            numbers_str = match.group(1)
            articles = [int(n.strip()) for n in numbers_str.split(',') if n.strip().isdigit()]
            if articles:
                return articles
        
        # Strategy 3: Find "Article X, Y, Z" pattern
        articles = re.findall(r'Article\s+(\d+)', response, re.IGNORECASE)
        if articles:
            return sorted(list(set(int(a) for a in articles)))
        
        # Strategy 4: Find all numbers (last resort)
        all_numbers = re.findall(r'\b(\d+)\b', response)
        if all_numbers:
            # Filter out obviously non-article numbers (>100)
            valid_articles = [int(n) for n in all_numbers if int(n) <= 99]
            if valid_articles:
                # Deduplicate and sort
                return sorted(list(set(valid_articles)))
        
        # Default: No violations found
        return [0]
    
    def _run_agent(self, prompt: str) -> List[int]:
        """
        Run the ReAct agent with a prompt and extract article numbers.
        
        FIX 3: Use synchronous invoke
        """
        if not self.agent:
            return self._fallback_analysis(prompt)
        
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=prompt)]
            }
            
            # Synchronous call
            result = self.agent.invoke(initial_state)
            
            # Extract final response
            final_message = result['messages'][-1]
            
            # Get text content
            if isinstance(final_message, AIMessage):
                response_text = final_message.content
            else:
                response_text = str(final_message)
            
            # Print for debugging
            print(f"\n[ReAct Response Preview]:\n{response_text[:500]}...\n")
            
            # Extract article numbers
            articles = self._extract_articles_from_response(response_text)
            print(f"[Extracted Articles]: {articles}\n")
            
            return articles
            
        except Exception as e:
            print(f"❌ ReAct agent error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_analysis(prompt)
    
    def _fallback_analysis(self, prompt: str) -> List[int]:
        """Simple fallback analysis if ReAct agent fails."""
        # Extract code from prompt
        code_match = re.search(r'```(?:java|kotlin|python|php|javascript)?\n(.*?)\n```', prompt, re.DOTALL)
        code = code_match.group(1) if code_match else prompt
        
        # Simple rule-based detection
        articles = set()
        
        # Device ID
        if re.search(r'getDeviceId|IMEI|ANDROID_ID|getSerialNumber', code, re.IGNORECASE):
            articles.update([6, 13])
        
        # Location
        if re.search(r'getLocation|getLatitude|getLongitude|LocationManager|GPS', code, re.IGNORECASE):
            articles.update([6, 7, 9, 13])
        
        # Camera/Microphone
        if re.search(r'Camera\.open|MediaRecorder|AudioRecord', code, re.IGNORECASE):
            articles.update([6, 7, 32])
        
        # Contacts
        if re.search(r'ContactsContract|getContacts', code, re.IGNORECASE):
            articles.update([6, 9, 13])
        
        # SMS
        if re.search(r'sendTextMessage|SmsManager|READ_SMS', code, re.IGNORECASE):
            articles.update([6, 13, 25])
        
        # Insecure transmission
        if re.search(r'http://(?!localhost)', code, re.IGNORECASE):
            articles.add(32)
        
        # Unencrypted storage
        if re.search(r'SharedPreferences|FileOutputStream', code, re.IGNORECASE):
            if not re.search(r'encrypt|cipher|Secure', code, re.IGNORECASE):
                articles.add(32)
        
        result = sorted(list(articles)) if articles else [0]
        print(f"[Fallback Analysis]: {result}")
        return result
    
    def predict_file_level(self, file_path: str, code: str, **kwargs) -> List[int]:
        """Analyze file-level GDPR violations using ReAct agent."""
        prompt = f"""Analyze the following file for GDPR compliance violations:

**File:** {file_path}

**Code:**
```
{code[:2000]}
```

Use your tools to perform thorough analysis and identify all violated GDPR articles."""
        
        return self._run_agent(prompt)
    
    def predict_module_level(self, file_path: str, module_name: str, 
                            code: str, **kwargs) -> List[int]:
        """Analyze module-level GDPR violations using ReAct agent."""
        prompt = f"""Analyze the following module/class for GDPR compliance violations:

**File:** {file_path}
**Module:** {module_name}

**Code:**
```
{code[:2000]}
```

Use your tools to perform thorough analysis and identify all violated GDPR articles."""
        
        return self._run_agent(prompt)
    
    def predict_line_level(self, file_path: str, line_spans: str, 
                          code: str, description: str, **kwargs) -> List[int]:
        """Analyze line-level GDPR violations using ReAct agent."""
        prompt = f"""Analyze the following code lines for GDPR compliance violations:

**File:** {file_path}
**Lines:** {line_spans}
**Description:** {description}

**Code:**
```
{code}
```

Use your tools to perform thorough analysis and identify all violated GDPR articles."""
        
        return self._run_agent(prompt)
    
    def predict_snippet(self, snippet: str, snippet_path: str = "", **kwargs) -> List[int]:
        """Analyze code snippet for GDPR violations using ReAct agent."""
        prompt = f"""Analyze the following code snippet for GDPR compliance violations:

**Snippet Path:** {snippet_path}

**Code:**
```
{snippet}
```

Use your tools to perform thorough analysis and identify all violated GDPR articles."""
        
        return self._run_agent(prompt)
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'agent') and self.agent:
            self.agent = None
