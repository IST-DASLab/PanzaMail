import json
import numpy as np

with open("data_for_mit/isabel_notfilledin_210.jsonl", 'r') as f:
    emails = [json.loads(line) for line in f.readlines()]

bio = ""
values = {}
with open("data_for_mit/isabel_madlibs.txt", 'r') as f:
    is_bio = False
    is_madlib = False
    for line in f.readlines():
        if line.strip() == "": continue
        if 'BACKSTORY' in line.strip():
            is_bio = True
            is_madlib = False
        if 'MADLIBS' in line.strip():
            is_madlib = True
            is_bio = False
        if line.strip().startswith("#"): continue
        if is_bio:
            bio += (" " + line.strip())
        if is_madlib:
            try:
                if "#" in line:
                    line = line.split("#")[0]
                k, v = line.strip().split(" -> ")
                if f"[{k}]" in values:
                    raise ValueError(f"key {k} already exists!")
                values[f"[{k}]"] = v
            except:
                raise ValueError("Parsing failed for", line)

for k, v in values.items():
    bio = bio.replace(k, v)

email_counts = []
for email in emails:
    pres = 0
    for k in values:
        if k in email["subject"] or k in email['email']:
            pres += 1
    email_counts.append(pres)

bc = np.bincount(email_counts)

counts = {k:0 for k in values}
for k, v in values.items():
    for email in emails:
        if email["subject"].count(k) > 0:
            counts[k] += email["subject"].count(k)
            email["subject"] = email["subject"].replace(k, v)
        if email["email"].count(k) > 0:
            counts[k] += email["email"].count(k)
            email["email"] = email["email"].replace(k, v)
        if "date" in email:
            email.pop("date")
print(counts)

res = {k: {"value": v, "count": counts[k] if k in counts else 0} for k, v in values.items()}

for email in emails:
    if "[" in email["subject"] + email["email"]:
        print(email)
print(bio)

print( [[k, v] for k, v in res.items() if v["count"] == 0])
    
print("facts per email", [[i, x] for i, x in enumerate(bc)])

abc = np.bincount([v["count"] for v in res.values()])
print("emails per fact", [[i, x] for i, x in enumerate(abc) if x > 0])

with open("data_for_mit/isabel_filledin_clean.jsonl", 'w') as f:
    for email in emails:
        f.write(json.dumps(email) + "\n")

with open("data_for_mit/isabel_backstory_filledin.txt", 'w') as f:
    f.write(bio)

# int(x) because numpy makes int64s and json refuses to serialize them.
with open("data_for_mit/isabel_stats.json", 'w') as f:
    json.dump({"facts_per_email": [[i, int(x)] for i, x in enumerate(bc)],
             "emails_per_fact": [[i, int(x)] for i, x in enumerate(abc) if x > 0]}, f)