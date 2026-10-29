"""Choose the ledger backend: NEUROTRACE_LEDGER=fabric, otherwise local JSON."""
import os


def get_ledger():
    if os.environ.get("NEUROTRACE_LEDGER", "local").lower() == "fabric":
        from ledger.fabric_ledger import FabricLedger
        return FabricLedger()
    from ledger.local_ledger import LocalLedger
    return LocalLedger()
