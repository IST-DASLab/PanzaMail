def load_preamble(path: str) -> str:
    with open(path, "r") as file:
        return file.read().strip()


def load_user_preamble(path: str) -> str:
    # The user preamble must be edited by the user in order to work as intended.
    # Here, we perform additional checks to make sure that that happened; if not,
    # We issue a warning to the user.
    with open(path, "r") as file:
        lines = [l for l in file.readlines() if not l.strip().startswith("#")]
        preamble = "".join(lines)
        if "CHANGE ME" in preamble:
            print(
                "*" * 66
                + "\n* WARNING: User prompt preamble not customized.                  *\n* Please edit the preamble at prompt_preambles/user_preamble.txt *\n"
                + "*" * 66
            )
        return preamble
