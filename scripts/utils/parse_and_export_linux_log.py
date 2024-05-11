import re
import csv
import sys

def parse_gruppo(testo, gruppo_id):
    dati = {'index': gruppo_id}
    match_iteration = re.search(r"Iteration (\d+)", testo)
    if not match_iteration:
        print(f"Errore: l'iterazione non è stata trovata nel gruppo {gruppo_id}. Controlla il formato del testo.")
        return dati

    iteration = match_iteration.group(1)
    
    sections = re.split(r'\btrain\b|\bval\b', testo)
    if len(sections) < 3:
        print(f"Errore: non è possibile dividere il testo in sezioni 'train' e 'val' nel gruppo {gruppo_id}.")
        return dati

    train_text = sections[1]
    val_text = sections[2]

    def parse_section(text, prefix):
        lines = re.findall(r"^\s+(.*?)\s+\│\s+([\d\.]+|nan)\s+\│\s+([\d\.]+|nan)\s+$", text, re.MULTILINE)
        for key, value, mean in lines:
            mean_or_value = '0' if 'nan' in mean else mean
            mean_or_value = value if 'nan' in mean_or_value or mean_or_value == '0' else mean_or_value
            modified_key = f"{prefix}_{key.replace(' ', '_').replace('/', '_')}"
            dati[modified_key] = mean_or_value

    parse_section(train_text, "dataset")
    parse_section(val_text, "val")

    dati['iteration'] = iteration

    return dati

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    gruppi = re.split(r'\[\d{2}:\d{2}:\d{2}\]', content)[1:]  # Divide il file nei gruppi
    all_data = []
    all_keys = set()
    valid_index = 1  # Inizia l'indice dei gruppi validi da 1

    for i, gruppo in enumerate(gruppi, start=1):
        gruppo_completo = f'[{gruppo}'
        dati_gruppo = parse_gruppo(gruppo_completo, valid_index)
        if len(dati_gruppo) > 2:  # Assicurati che ci siano più dati oltre all'index e iteration
            all_data.append(dati_gruppo)
            all_keys.update(dati_gruppo.keys())
            valid_index += 1  # Incrementa solo se il gruppo è valido

    if not all_data:
        print("Nessun dato valido da scrivere.")
        return

    with open('output.csv', 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=list(all_keys))
        dict_writer.writeheader()
        dict_writer.writerows(all_data)

# Controlla l'input da linea di comando e chiama la funzione
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_and_export.py <path_to_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    process_file(filepath)