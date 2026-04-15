from abc import ABC, abstractmethod


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path):
        """Parse a document and return list of documents."""
        pass
