import pandas as pd
from pathlib import Path

participants_path = Path("data/ds004504/participants.tsv")

new_rows = [
    {"participant_id": "sub-089", "Gender": "M", "Age": 66, "Group": "A", "MMSE": 15},
    {"participant_id": "sub-090", "Gender": "F", "Age": 71, "Group": "A", "MMSE": 18},
    {"participant_id": "sub-091", "Gender": "M", "Age": 62, "Group": "A", "MMSE": 20},
    {"participant_id": "sub-092", "Gender": "F", "Age": 58, "Group": "F", "MMSE": 22},
    {"participant_id": "sub-093", "Gender": "M", "Age": 64, "Group": "F", "MMSE": 19},
    {"participant_id": "sub-094", "Gender": "F", "Age": 55, "Group": "F", "MMSE": 24},
    {"participant_id": "sub-095", "Gender": "M", "Age": 68, "Group": "C", "MMSE": 30},
    {"participant_id": "sub-096", "Gender": "F", "Age": 61, "Group": "C", "MMSE": 30},
    {"participant_id": "sub-097", "Gender": "M", "Age": 73, "Group": "C", "MMSE": 30},
    {"participant_id": "sub-098", "Gender": "F", "Age": 59, "Group": "C", "MMSE": 30},
]

def main():
    if not participants_path.exists():
        raise FileNotFoundError(f"participants.tsv not found at: {participants_path}")

    df = pd.read_csv(participants_path, sep="\t")

    existing_ids = set(df["participant_id"].astype(str))
    to_add = [row for row in new_rows if row["participant_id"] not in existing_ids]

    if len(to_add) == 0:
        print("✅ All synthetic subjects already exist in participants.tsv")
        return

    new_df = pd.DataFrame(to_add)
    updated = pd.concat([df, new_df], ignore_index=True)

    updated.to_csv(participants_path, sep="\t", index=False)

    print(f"✅ Added {len(to_add)} new subjects into participants.tsv")
    print("Added IDs:", [r["participant_id"] for r in to_add])

if __name__ == "__main__":
    main()
