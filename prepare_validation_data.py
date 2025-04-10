import os
import random
import json

def split_dataset_interactive():
    print("Pfad zum finetuning datensatz (jsonl) eingeben:")
    input_path = input("Pfad: ").strip()

    if not os.path.isfile(input_path):
        print("❌ Datei nicht gefunden.")
        return

    # Verzeichnis und Dateiname
    base_dir = os.path.dirname(os.path.abspath(input_path))
    train_output = os.path.join(base_dir, "finetuning_data_split.jsonl")
    val_output = os.path.join(base_dir, "finetuning_validation_data_split.jsonl")

    # Lade Datensätze
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        print("❌ Zu wenige Einträge zum Aufteilen.")
        return

    # Split-Verhältnis
    val_ratio = 0.1
    val_count = int(len(lines) * val_ratio)
    random.shuffle(lines)

    val_set = lines[:val_count]
    train_set = lines[val_count:]

    # Schreibe die gesplitteten Dateien
    with open(train_output, "w", encoding="utf-8") as f_train:
        f_train.writelines(train_set)

    with open(val_output, "w", encoding="utf-8") as f_val:
        f_val.writelines(val_set)

    print(f"✅ Aufteilung abgeschlossen.")
    print(f"Trainingsdaten: {len(train_set)} → {train_output}")
    print(f"Validierungsdaten: {len(val_set)} → {val_output}")

if __name__ == "__main__":
    split_dataset_interactive()
