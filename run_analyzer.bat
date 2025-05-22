@echo off
setlocal
set G_MESSAGES_DEBUG=none
set G_DEBUG=fatal-warnings
python gpt4o_stock_comparator.py
endlocal 