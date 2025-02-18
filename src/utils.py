import re

def minutes(x):
    if isinstance(x, int): return x
    assert isinstance(x, str)
    x = x.lower().strip()
    if x.isdigit() or x.endswith("min"):
        return int(x.replace("min", ""))
    match = re.match(r"^(\d+)([ymwdh])$", x)
    if not match:
        raise ValueError(f"Invalid time format: {x}")
    num, unit = int(match.group(1)), match.group(2)
    units = {'y': 525600, 'm': 43800, 'w': 10080, 'd': 1440, 'h': 60}
    assert unit in units.keys()
    return num * units.get(unit) 