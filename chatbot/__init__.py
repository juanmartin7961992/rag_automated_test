"""
Chatbot module. 

This module provides the ConfigurableChatbot class.
"""

from .core.chatbot import ConfigurableChatbot
from config.settings import ChatbotConfig

__all__ = ["ConfigurableChatbot", "ChatbotConfig"]

__version__ = "0.1.0"