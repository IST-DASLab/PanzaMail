This part of the repo is for preparing new anonymous data.

The process should be something as follows:

1. Collect data. For gmail emails, this is done by first exporting them with [Google Takeout](https://takeout.google.com/),
then running the [email extraction code](https://github.com/IST-DASLab/PanzaMail/blob/main/src/panza/data_preparation/extract_emails.py)
to get a `.jsonl` file, each line of which is an email.
1. Then, the emails need to be anonymized. In the past, this was done manually, by preparing the files like [`isabel_notfilledin_210.jsonl`](https://github.com/IST-DASLab/PanzaMail/blob/Abantu/data_cleaning/isabel_notfilledin_210.jsonl)
   and [`isabel_madlibs.txt`](https://github.com/IST-DASLab/PanzaMail/blob/Abantu/data_cleaning/isabel_madlibs.txt), which contain the emails with
   private info replaced with placeholders, and a mapping from the placeholders to fake names/places. The script
   [`fill_in_blanks.py`](https://github.com/IST-DASLab/PanzaMail/blob/Abantu/data_cleaning/fill_in_blanks.py) could then be run to fill
   in the blanks and prepare the data.

   Our goal is to replace the manual identification of the PII with a script that uses an LLM. The script [`extract_personal_info_from_data.py`](https://github.com/IST-DASLab/PanzaMail/blob/Abantu/data_cleaning/extract_personal_info_from_data.py)
   is a first attempt at this, but so far it fails because Aya-Expanse-8B isn't a nearly powerful enough model to do what we want - it should
   be modified to call out to a more powerful model or API in order to get better PII identification.
