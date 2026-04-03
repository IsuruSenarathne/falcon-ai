"""Service for formatting and displaying responses."""


class ResponseFormatter:
    """Responsible for formatting responses for display."""

    SEPARATOR = "-" * 33

    def format(self, response: str) -> str:
        """Format LLM response for display."""
        return response

    def display(self, response: str) -> None:
        """Display formatted response."""
        print(f"\n{self.SEPARATOR}")
        print(self.format(response))
        print(f"{self.SEPARATOR}\n")
