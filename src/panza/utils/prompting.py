from typing import List, Optional, Text

from panza.utils.documents import Email

MISTRAL_PROMPT_START_WRAPPER = "[INST] "
MISTRAL_PROMPT_END_WRAPPER = " [/INST]"
MISTRAL_RESPONSE_START_WRAPPER = "<s>"
MISTRAL_RESPONSE_END_WRAPPER = "</s>"

LLAMA3_PROMPT_START_WRAPPER = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_PROMPT_END_WRAPPER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_RESPONSE_START_WRAPPER = ""
LLAMA3_RESPONSE_END_WRAPPER = "<|eot_id|>"

PHI3_PROMPT_START_WRAPPER = "<s><|user|> "
PHI3_PROMPT_END_WRAPPER = "<|end|><|assistant|> "
PHI3_RESPONSE_START_WRAPPER = ""
PHI3_RESPONSE_END_WRAPPER = "<|end|>"


def create_prompt(
    user_input: Text,
    system_preamble: Text,
    user_preamble: Text,
    rag_preamble: Optional[Text] = None,
    relevant_emails: Optional[List[Email]] = None,
    thread_preamble: Optional[Text] = None,
    thread_emails: Optional[List[Text]] = None,
) -> Text:

    if relevant_emails:
        assert rag_preamble, "RAG preamble format must be provided if similar emails are provided."
        rag_prompt = _create_rag_preamble_from_emails(rag_preamble, relevant_emails).strip()
    else:
        rag_prompt = ""

    if thread_emails:
        assert thread_preamble, "Thread preamble format must be provided if thread is provided."
        thread_prompt = _create_threading_preamble(thread_preamble, thread_emails).strip()
    else:
        thread_prompt = ""

    system_preamble = system_preamble.strip()
    user_preamble = user_preamble.strip()

    prompt = ""
    if system_preamble:
        prompt += f"{system_preamble}\n\n"
    if user_preamble:
        prompt += f"{user_preamble}\n\n"
    if rag_prompt:
        prompt += f"{rag_prompt}\n\n"
    if thread_prompt:
        prompt += f"{thread_prompt}\n\n"
    prompt += f"Instruction: {user_input}"

    return prompt


def _create_rag_preamble_from_emails(rag_preamble_format: Text, emails: List[Email]) -> Text:
    rag_context = _create_rag_context_from_emails(emails)
    return rag_preamble_format.format(rag_context=rag_context)


def _create_rag_context_from_emails(emails: List[Email]) -> Text:
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
            # f"SUBJECT: {email.metadata['subject']}\n"  # TODO(armand): Handle subject metadata
            f"E-MAIL CONTENT:\n{email.page_content}\n\n---\n\n"
        )

    return rag_context


def _create_threading_preamble(threading_preamble_format: Text, thread: List[Text]) -> Text:
    threading_context = _create_threading_context(thread)
    return threading_preamble_format.format(threading_context=threading_context)


def _create_threading_context(thread: List[Text]) -> Text:
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
        preamble = "".join(lines)
        if "CHANGE ME" in preamble:
            print(
                "*" * 66
                + "\n* WARNING: User prompt preamble not customized.                  *\n* Please edit the preamble at prompt_preambles/user_preamble.txt *\n"
                + "*" * 66
            )
        return preamble


def load_all_preambles(system_preamble, user_preamble, rag_preamble, thread_preamble):
    system_preamble = load_preamble(system_preamble) if system_preamble else ""
    user_preamble = load_user_preamble(user_preamble) if user_preamble else ""
    rag_preamble = load_preamble(rag_preamble) if rag_preamble else ""
    thread_preamble = load_preamble(thread_preamble) if thread_preamble else ""
    return system_preamble, user_preamble, rag_preamble, thread_preamble


def get_model_special_tokens(model_name):
    model_name = model_name.lower()
    if "llama" in model_name:
        prompt_start_wrapper = LLAMA3_PROMPT_START_WRAPPER
        prompt_end_wrapper = LLAMA3_PROMPT_END_WRAPPER
        response_start_wrapper = LLAMA3_RESPONSE_START_WRAPPER
        response_end_wrapper = LLAMA3_RESPONSE_END_WRAPPER
    elif "mistral" in model_name.lower():
        prompt_start_wrapper = MISTRAL_PROMPT_START_WRAPPER
        prompt_end_wrapper = MISTRAL_PROMPT_END_WRAPPER
        response_start_wrapper = MISTRAL_RESPONSE_START_WRAPPER
        response_end_wrapper = MISTRAL_RESPONSE_END_WRAPPER
    elif "phi" in model_name.lower():
        prompt_start_wrapper = PHI3_PROMPT_START_WRAPPER
        prompt_end_wrapper = PHI3_PROMPT_END_WRAPPER
        response_start_wrapper = PHI3_RESPONSE_START_WRAPPER
        response_end_wrapper = PHI3_RESPONSE_END_WRAPPER
    else:
        raise ValueError(f"Presets missing for prompting model {model_name}")

    return prompt_start_wrapper, prompt_end_wrapper, response_start_wrapper, response_end_wrapper
