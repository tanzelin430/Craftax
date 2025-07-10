#!/usr/bin/env python3
"""
Register CraftaxAgentLoop with Verl framework
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and register
from craftax_agent_loop import CraftaxAgentLoop

def register_craftax_agent():
    """Register CraftaxAgentLoop with the Verl framework"""
    try:
        # Import verl agent registry
        from verl.experimental.agent_loop.agent_loop_registry import agent_loop_registry
        
        # Register our agent loop
        agent_loop_registry.register("craftax_agent", CraftaxAgentLoop)
        
        print("✅ Successfully registered CraftaxAgentLoop with Verl")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Verl registry: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to register CraftaxAgentLoop: {e}")
        return False

if __name__ == "__main__":
    register_craftax_agent()