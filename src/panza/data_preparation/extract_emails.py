import json
import mailbox
import re
from email.utils import parsedate_to_datetime
from email.message import Message
from mailbox import mboxMessage
from os import makedirs
from os.path import join, dirname

import langdetect

CLEAN_EMAILS = []
DISCARDED_EMAILS = {
    "non_english": [],
    "forwarded": [],
    "short": [],
    "empty": [],
    "cant_decode_utf8": [],
}

SHORT_EMAIL_THRESHOLD = 10  # words

FORWARDED_MESSAGE_TAG = "---------- Forwarded message ---------"


def extract_only_plain_text(msg_part):
    if msg_part.get_content_type() == "text/plain":
        body = msg_part.get_payload(decode=True)
        plain_text = body.decode()  # assuming the text is in UTF-8, handle other cases later
        return plain_text


def skip_forwarded_messages(plain_text):
    if FORWARDED_MESSAGE_TAG in plain_text:
        DISCARDED_EMAILS["forwarded"].append(plain_text)
        return ""
    else:
        return plain_text


def remove_date_time(email_body):
    # Regular expression pattern to match lines starting with "On " and ending with "> wrote: "
    # The pattern uses non-greedy matching (.*?) to find the shortest match that satisfies the condition
    pattern = re.compile(r"(^On.*wrote.*)|(^Am.*schrieb.*)", re.MULTILINE | re.DOTALL)

    match = pattern.search(email_body)
    if match:
        return (email_body[: match.start()] + email_body[match.end() :]).strip()
    else:
        return email_body


def remove_lines_starting_with_gt(text):
    lines = text.split("\n")
    filtered_lines = [
        line for line in lines if not line.startswith(">")
    ]  # Filter out lines starting with "> "
    return "\n".join(filtered_lines)


def count_words(s):
    return len(s.split())


def extract_by_quote_level(text):
    # Split the text into lines
    lines = text.split("\n")

    # Dictionary to store lines by quote level
    grouped_lines = {}

    for line in lines:
        # Count the number of '>' at the start of the line
        quote_level = len(re.match(r"^>*", line).group())

        # Remove leading '>' and spaces
        clean_line = re.sub(r"^>*\s*", "", line)

        # Add the clean line to the appropriate group
        if quote_level not in grouped_lines:
            grouped_lines[quote_level] = []
        grouped_lines[quote_level].append(clean_line)

    return grouped_lines


def filter_message(msg):
    try:
        plain_text = extract_only_plain_text(msg)
    except:
        DISCARDED_EMAILS["cant_decode_utf8"].append(msg)
        return None

    if plain_text is None:
        return None

    plain_text = skip_forwarded_messages(plain_text)
    email_with_thread = extract_by_quote_level(plain_text)
    email_with_thread = ["\n".join(an_email).strip() for an_email in email_with_thread.values()]

    # remove "On ... wrote:" lines
    email_with_thread = [remove_date_time(an_email) for an_email in email_with_thread]

    main_email = email_with_thread.pop(0)
    email_with_thread.reverse()  # chronological order

    # check length before detecting language
    if count_words(main_email) < SHORT_EMAIL_THRESHOLD:
        DISCARDED_EMAILS["short"].append(plain_text)
        return None
    try:
        if langdetect.detect(main_email) != "en":
            DISCARDED_EMAILS["non_english"].append(plain_text)
            return None
    except:
        # failed to detect language
        DISCARDED_EMAILS["non_english"].append(plain_text)
        return None

    if main_email.isspace() or main_email == "":
        DISCARDED_EMAILS["empty"].append(plain_text)
        return None

    return (main_email.strip(), [an_email.strip() for an_email in email_with_thread])


def extract_emails(mailbox_path, output_path, email_addresses, save_discarded_emails_path):

    MBOX_PATH = mailbox_path
    EMAIL = email_addresses

    mbox = mailbox.mbox(MBOX_PATH)
    n_emails = len(mbox)
    for i, message in enumerate(mbox):
        print(f"--> processing {i}/{n_emails} <--")
        # Filter messages sent from your email address
        if message["from"] and any(email in message["from"] for email in EMAIL):
            if message["Date"]:
                date = parsedate_to_datetime(message["Date"]).isoformat()
            else:
                print("Date was not found in the email. Skipping.")
                continue
            if message.is_multipart():
                for part in message.walk():
                    filtered_msg = filter_message(part)
                    if filtered_msg is not None:
                        print(filtered_msg)
                        main_email, thread = filtered_msg
                        CLEAN_EMAILS.append(
                            {
                                "email": main_email,
                                "thread": thread,
                                "subject": message["Subject"],
                                "date": date,
                            }
                        )
            else:
                filtered_msg = filter_message(message)
                if filtered_msg is not None:
                    print(filtered_msg)
                    main_email, thread = filtered_msg
                    CLEAN_EMAILS.append(
                        {
                            "email": main_email,
                            "thread": thread,
                            "subject": message["Subject"],
                            "date": date,
                        }
                    )

    print(f"\n---> [Cleaning stats] <---")
    print(f"# clean emails = {len(CLEAN_EMAILS)}")
    print(
        f"# discarded emails:"
        f"\n\t non_english = {len(DISCARDED_EMAILS['non_english'])}"
        f"\n\t empty = {len(DISCARDED_EMAILS['empty'])}"
        f"\n\t short (less than {SHORT_EMAIL_THRESHOLD} words)= {len(DISCARDED_EMAILS['short'])}"
        f"\n\t forwarded = {len(DISCARDED_EMAILS['forwarded'])}"
        f"\n\t cant_decode_utf8 = {len(DISCARDED_EMAILS['cant_decode_utf8'])}"
    )

    first_email = EMAIL[0]
    username = first_email[: first_email.find("@")]

    makedirs(dirname(output_path), exist_ok=True)

    # Save clean emails
    with open(join(output_path), "w", encoding="utf-8") as f:
        for item in CLEAN_EMAILS:
            json_record = json.dumps(item)
            f.write(json_record + "\n")

    # Save discarded emails
    if save_discarded_emails_path and save_discarded_emails_path != "":
        print(f"\n---> Processing Discarded Emails <---")
        makedirs(save_discarded_emails_path, exist_ok=True)
        for k, v in DISCARDED_EMAILS.items():
            print(f"--> processing {k} emails <--")
            output_path = join(save_discarded_emails_path, f"{username}_discarded_{k}.jsonl")
            with open(output_path, "w", encoding="utf-8") as f:
                discarded_emails = len(v)
                for i, item in enumerate(v):
                    print("\n\n\n\n\===========================")
                    if type(item) is Message or type(item) is mboxMessage:
                        item = item.get_payload()
                    print(f"--> processing {i}/{discarded_emails} <--")
                    json_record = json.dumps(item)
                    f.write(json_record + "\n")
