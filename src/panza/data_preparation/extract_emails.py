import argparse
import json
import mailbox
import re
from os import makedirs
from os.path import join

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


def extract_only_plain_text(msg_part):
    if msg_part.get_content_type() == "text/plain":
        body = msg_part.get_payload(decode=True)
        plain_text = body.decode()  # assuming the text is in UTF-8, handle other cases later
        return plain_text


def skip_forwarded_messages(plain_text):
    if "---------- Forwarded message ---------" in plain_text:
        DISCARDED_EMAILS["forwarded"].append(plain_text)
        return ""
    else:
        return plain_text


def remove_quoted_content(email_body):
    # Regular expression pattern to match lines starting with "On " and ending with "> wrote: "
    # The pattern uses non-greedy matching (.*?) to find the shortest match that satisfies the condition
    pattern = re.compile(r"(^On.*wrote.*)|(^Am.*schrieb.*)", re.MULTILINE | re.DOTALL)

    # Search for the pattern and truncate everything after it
    match = pattern.search(email_body)
    if match:
        # Truncate the email body up to the start of the matched pattern
        return email_body[: match.start()].strip()
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


def filter_message(msg):
    try:
        plain_text = extract_only_plain_text(msg)
    except:
        DISCARDED_EMAILS["cant_decode_utf8"].append(msg)
        return None

    if plain_text is None:
        return None

    plain_text = skip_forwarded_messages(plain_text)
    plain_text = remove_quoted_content(plain_text)
    # sometimes remove_quoted_content misses, so making sure we remove lines with ">" at the start
    plain_text = remove_lines_starting_with_gt(plain_text)

    # check length before detecting language
    if count_words(plain_text) < SHORT_EMAIL_THRESHOLD:
        DISCARDED_EMAILS["short"].append(plain_text)
        return None
    try:
        if langdetect.detect(plain_text) != "en":
            DISCARDED_EMAILS["non_english"].append(plain_text)
            return None
    except:
        # failed to detect language
        DISCARDED_EMAILS["non_english"].append(plain_text)
        return None

    if plain_text.isspace() or plain_text == "":
        DISCARDED_EMAILS["empty"].append(plain_text)
        return None

    return plain_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Process an MBOX file for PANZA project.")
    parser.add_argument("--mbox-path", help="Path to the MBOX file.")
    parser.add_argument("--output-path", help="Path to the directory to save the output files.")
    parser.add_argument(
        "--email",
        action="append",
        help="Email address(es) to filter the messages. Use the argument multiple times for multiple emails.",
    )
    parser.add_argument("--save-discarded-emails", action="store_true")
    args = parser.parse_args()

    MBOX_PATH = args.mbox_path
    EMAIL = args.email

    mbox = mailbox.mbox(MBOX_PATH)
    n_emails = len(mbox)
    for i, message in enumerate(mbox):
        print(f"--> processing {i}/{n_emails} <--")
        # Filter messages sent from your email address
        if message["from"] and any(email in message["from"] for email in EMAIL):
            if message.is_multipart():
                for part in message.walk():
                    filtered_msg = filter_message(part)
                    if filtered_msg is not None:
                        print(filtered_msg)
                        CLEAN_EMAILS.append({"email": filtered_msg, "subject": message["Subject"]})
            else:
                filtered_msg = filter_message(message)
                if filtered_msg is not None:
                    print(filtered_msg)
                    CLEAN_EMAILS.append({"email": filtered_msg, "subject": message["Subject"]})

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

    makedirs(args.output_path, exist_ok=True)

    # Save clean emails
    with open(join(args.output_path, username + "_clean.jsonl"), "w", encoding="utf-8") as f:
        for item in CLEAN_EMAILS:
            json_record = json.dumps(item)
            f.write(json_record + "\n")

    # Save discarded emails
    if args.save_discarded_emails:
        makedirs(join(args.output_path, "discarded"), exist_ok=True)
        for k, v in DISCARDED_EMAILS.items():
            output_path = join(
                args.output_path, "discarded", username + "_discarded_" + k + ".jsonl"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                for item in v:
                    json_record = json.dumps(item)
                    f.write(json_record + "\n")


if __name__ == "__main__":
    main()
