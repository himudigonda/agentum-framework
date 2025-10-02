from .filesystem import read_file, write_file
try:
    from .speech import text_to_speech, transcribe_audio
except ImportError:

    def text_to_speech(*args, **kwargs):
        raise ImportError('text_to_speech requires Google Cloud dependencies. Run: pip install google-cloud-texttospeech langchain-google-community')

    def transcribe_audio(*args, **kwargs):
        raise ImportError('transcribe_audio requires Google Cloud dependencies. Run: pip install google-cloud-texttospeech langchain-google-community')
try:
    from .web_search import search_web_tavily
except ImportError:

    def search_web_tavily(*args, **kwargs):
        raise ImportError("search_web_tavily requires 'tavily-python' to be installed. Run: pip install tavily-python")
__all__ = ['search_web_tavily', 'write_file', 'read_file', 'transcribe_audio', 'text_to_speech']