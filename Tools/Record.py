from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_datetime(zone = "Asia/Ho_Chi_Minh"):
    now = datetime.now(ZoneInfo(zone))
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second
    }
    
def dataset_agreegate(point):
    point['train_numsamples'] = len(point['train_ds'])
    point['test_numsamples'] = len(point['test_ds'])

