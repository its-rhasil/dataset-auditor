from dataclasses import dataclass, field

@dataclass
class DataProfile:
    rows: int
    columns: int
    filename: str
    fingerprint: str = ""
    missing: dict = field(default_factory=dict)
    duplicates: dict = field(default_factory=dict)
    outliers: dict = field(default_factory=dict)
    cardinality: dict = field(default_factory=dict)
    imbalance: dict = field(default_factory=dict)
    leakage: dict = field(default_factory=dict)
    # target_candidates: dict = field(default_factory=dict)