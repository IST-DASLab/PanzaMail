from typing import List, Tuple

from ..entities import Email, EmailInstruction
from ..retriever import DocumentRetriever
from .base import PromptBuilder
from .utils import load_preamble, load_user_preamble


class EmailPromptBuilder(PromptBuilder):
    def __init__(
        self,
        retriever: DocumentRetriever,
        system_preamble: str,
        user_preamble: str,
        rag_preamble: str,
        thread_preamble: str,
        number_rag_emails: int,
        rag_relevance_threshold: float,
        number_thread_emails: int,
    ):
        self.retriever = retriever
        self.system_preamble = system_preamble
        self.user_preamble = user_preamble
        self.rag_preamble = rag_preamble
        self.thread_preamble = thread_preamble
        self.number_rag_emails = number_rag_emails
        self.rag_relevance_threshold = rag_relevance_threshold
        self.number_thread_emails = number_thread_emails

        self.retriever.set_document_class(Email)

    def _create_rag_preamble_from_emails(self, emails: List[Email]) -> str:
        rag_context = self._create_rag_context_from_emails(emails)
        return self.rag_preamble.format(rag_context=rag_context)

    def _create_rag_context_from_emails(self, emails: List[Email]) -> str:
        """Creates a RAG context from a list of relevant e-mails.

        The e-mails are formatted as follows:

        E-MAIL CONTENT:
        <e-mail1 content>

        ---

        E-MAIL CONTENT:
        <e-mail2 content>

        ---
        ...
        """

        rag_context = ""
        for email in emails:
            rag_context += f"E-MAIL CONTENT:\n{email.email}\n\n---\n\n"

        return rag_context

    def _create_threading_preamble(self, thread: List[str]) -> str:
        threading_context = self._create_threading_context(thread)
        return self.thread_preamble.format(threading_context=threading_context)

    def _create_threading_context(self, thread: List[str]) -> str:
        """Creates a threading context from a list of relevant e-mails.

        The e-mails are formatted as follows:

        <e-mail1 content>

        ---

        <e-mail2 content>

        ---
        ...
        """

        threading_context = ""
        for email in thread:
            threading_context += f"{email}\n\n---\n\n"

        return threading_context

    @staticmethod
    def load_all_preambles(
        system_preamble_path: str,
        user_preamble_path: str,
        rag_preamble_path: str,
        thread_preamble_path: str,
    ) -> Tuple[str, str, str, str]:
        """Load all preambles from file."""
        system_preamble = load_preamble(system_preamble_path) if system_preamble_path else ""
        user_preamble = load_user_preamble(user_preamble_path) if user_preamble_path else ""
        rag_preamble = load_preamble(rag_preamble_path) if rag_preamble_path else ""
        thread_preamble = load_preamble(thread_preamble_path) if thread_preamble_path else ""
        return system_preamble, user_preamble, rag_preamble, thread_preamble

    def build_prompt(
        self,
        instruction: EmailInstruction,
    ) -> str:

        if self.number_thread_emails and not self.rag_preamble:
            raise ValueError("RAG preamble format must be provided if RAG is used.")

        if self.number_thread_emails and not self.thread_preamble:
            raise ValueError("Thread preamble format must be provided if thread is used.")

        if self.number_rag_emails > 0:
            relevant_emails = self.retriever.retrieve(
                instruction.instruction, self.number_rag_emails, self.rag_relevance_threshold
            )
            rag_prompt = self._create_rag_preamble_from_emails(relevant_emails).strip()
        else:
            rag_prompt = ""

        if self.number_thread_emails > 0:
            thread_prompt = self._create_threading_preamble(
                instruction.thread[: self.number_thread_emails]
            ).strip()
        else:
            thread_prompt = ""

        system_preamble = self.system_preamble.strip()
        user_preamble = self.user_preamble.strip()

        prompt = ""
        if system_preamble:
            prompt += f"{system_preamble}\n\n"
        if user_preamble:
            prompt += f"{user_preamble}\n\n"
        if rag_prompt:
            prompt += f"{rag_prompt}\n\n"
        if thread_prompt:
            prompt += f"{thread_prompt}\n\n"
        prompt += f"Instruction: {instruction.instruction}"

        return prompt
