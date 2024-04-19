from typing import List, Optional, Text

from langchain_core.documents import Document


def create_prompt(
    user_input: Text,
    system_preamble: Text,
    user_preamble: Text,
    rag_preamble: Optional[Text] = None,
    relevant_emails: Optional[List[Document]] = None,
) -> Text:

    if relevant_emails:
        assert rag_preamble, "RAG preamble format must be provided if similar emails are provided"
        rag_prompt = _create_rag_preamble_from_emails(rag_preamble, relevant_emails).strip()
    else:
        rag_prompt = ""

    system_preamble = system_preamble.strip()
    user_preamble = user_preamble.strip()

    prompt = ""
    if system_preamble:
        prompt += f"{system_preamble}\n\n"
    if user_preamble:
        prompt += f"{user_preamble}\n\n"
    if rag_prompt:
        prompt += f"{rag_prompt}\n\n"
    prompt += f"Instruction: {user_input}"

    return prompt


def _create_rag_preamble_from_emails(rag_preamble_format: Text, emails: List[Document]) -> Text:
    rag_context = _create_rag_context_from_emails(emails)
    return rag_preamble_format.format(rag_context=rag_context)


def _create_rag_context_from_emails(emails: List[Document]) -> Text:
    """Creates a RAG context from a list of relevant e-mails.

    The e-mails are formatted as follows:

    SUBJECT: <e-mail1 subject>
    E-MAIL CONTENT:
    <e-mail1 content>

    ---

    SUBJECT: <e-mail2 subject>
    E-MAIL CONTENT:
    <e-mail2 content>

    ---
    ...
    """

    rag_context = ""
    for email in emails:
        rag_context += (
            f"SUBJECT: {email.metadata['subject']}\n"
            f"E-MAIL CONTENT:\n{email.page_content}\n\n---\n\n"
        )

    return rag_context


def load_preamble(path):
    with open(path, "r") as file:
        return file.read().strip()


# The user preamble must be edited by the user in order to work as intended.
# Here, we perform additional checks to make sure that that happened; if not,
# We issue a warning to the user.
def load_user_preamble(path):
    with open(path, "r") as file:
        lines = [l for l in file.readlines() if not l.strip().startswith("#")]
        print(lines)
        preamble =  "".join(lines)
        if "CHANGE ME" in preamble:
            print("*" * 66 + "\n* WARNING: User prompt preamble not customized.                  *\n* Please edit the preamble at prompt_preambles/user_preamble.txt *\n" + "*" * 66) 
        return preamble

def load_all_preambles(system_preamble, user_preamble, rag_preamble):
    system_preamble = load_preamble(system_preamble) if system_preamble else ""
    user_preamble = load_user_preamble(user_preamble) if user_preamble else ""
    rag_preamble = load_preamble(rag_preamble) if rag_preamble else ""
    return system_preamble, user_preamble, rag_preamble
