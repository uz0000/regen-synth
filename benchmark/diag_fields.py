"""Check field_dict classification for hypothyroid."""
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
result = do_ingest("benchmark/data/hypothyroid.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=0))
for name, meta in sorted(result.field_dict.items()):
    print(f"  {name:25s} {meta.field_type.value:15s}", end="")
    print()
