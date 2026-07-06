class ObjectCounter:
    def __init__(self):
        self.unique_ids = set()

    def update(self, track_ids):
        for tid in track_ids:
            self.unique_ids.add(int(tid))
        return len(self.unique_ids)