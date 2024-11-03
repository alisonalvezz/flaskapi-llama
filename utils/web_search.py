from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3, tavily_api_key='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')