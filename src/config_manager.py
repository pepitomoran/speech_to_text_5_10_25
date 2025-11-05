# filepath: src/config_manager.py
import csv

def read_simple_kv_csv(path):
    """
    Read a two-column CSV with header key,value and return a dict.
    Ignores empty lines.
    """
    cfg = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # skip header if it looks like header
        first = next(reader, None)
        if first is None:
            return cfg
        # If the first row is header like ['key','value'], continue reading
        if [c.strip().lower() for c in first] != ['key', 'value']:
            # treat first as data row
            k, v = first
            cfg[k.strip()] = v.strip()
        for row in reader:
            if not row or len(row) < 2:
                continue
            k, v = row[0].strip(), row[1].strip()
            if k:
                cfg[k] = v
    return cfg
